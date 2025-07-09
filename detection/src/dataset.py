import torch
import numpy as np

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

import albumentations as A
from transformers.image_processing_utils import BatchFeature
from collections.abc import Mapping
from typing import Any, Optional, Union

from functools import partial

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

FDS = None  # Cache FederatedDataset

image_square_size = 600
max_size = image_square_size
train_augment_and_transform = A.Compose(
    [
        A.Compose(
            [
                A.SmallestMaxSize(max_size=max_size, p=1.0),
                A.RandomSizedBBoxSafeCrop(height=max_size, width=max_size, p=1.0),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.Blur(blur_limit=7, p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
            ],
            p=0.1,
        ),
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)

def format_image_annotations_as_coco(
    image_id: str, categories: list[int], bboxes: list[tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (list[int]): list of categories/class labels corresponding to provided bounding boxes
        bboxes (list[tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, bbox in zip(categories, bboxes):
        area = bbox[2] * bbox[3]
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []

    for image_id, image, objects in zip(examples["id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bboxes"], category=objects["classes"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result

def collate_fn(batch: list[BatchFeature]) -> Mapping[str, Union[torch.Tensor, list[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

def load_data(partition_id: int,
              num_partitions: int,
              dataset_name: str,
              model_name: str,
              split: str = "train"):
    assert num_partitions == 10, "This custom partitioning assumes 10 clients."

    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    raw_ds = FDS.load_partition(partition_id, split)
    print(f"[Client {partition_id}] Got {len(raw_ds)} samples.")

    image_processor = AutoImageProcessor.from_pretrained(
        model_name,
        do_resize=True,
        size={"max_height": image_square_size, "max_width": image_square_size},
        do_pad=True,
        pad_size={"height": image_square_size, "width": image_square_size},
        use_fast=True,
    )

    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )

    trainset = raw_ds.with_transform(train_transform_batch)

    return trainset, image_processor, collate_fn


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
