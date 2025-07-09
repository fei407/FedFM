import random, numpy as np, torch, albumentations as A
from torch.utils.data import Dataset
from functools import partial
from transformers import DeformableDetrImageProcessor
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
import torch.nn.functional as F

FDS = None  # Cache FederatedDataset
PROCESSOR  = None

SCALES = [480,512,544,576,608,640,672,704,736,768,800]

def build_aug_pipelines():
    # 随机水平翻转
    aug_flip = A.HorizontalFlip(p=0.5)

    # 多尺度 resize
    aug_resize1 = A.Compose([
        A.OneOf([A.SmallestMaxSize(s) for s in SCALES], p=1.0),
        A.LongestMaxSize(max_size=1333),
    ])

    # 小尺寸随机裁剪后再 resize
    aug_resize2 = A.Compose([
        A.OneOf([
            A.SmallestMaxSize(400),
            A.SmallestMaxSize(500),
            A.SmallestMaxSize(600),
        ], p=1.0),
        A.RandomResizedCrop(size=(384, 600), scale=(0.5,1.0), ratio=(0.75,1.33), p=1.0),
        A.LongestMaxSize(max_size=1333),
    ])

    train_tf = A.Compose(
        [aug_flip, A.OneOf([aug_resize1, aug_resize2], p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
    )

    val_tf = A.Compose(
        [A.SmallestMaxSize(800), A.LongestMaxSize(max_size=1333)],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
    )

    return train_tf

def get_processor(model_name: str):
    global PROCESSOR
    if PROCESSOR is None:
        PROCESSOR = DeformableDetrImageProcessor.from_pretrained(model_name)
    return PROCESSOR

def xywh_to_cxcywh_norm(boxes, size):
    W, H = size
    return [[(x + w * 0.5) / W, (y + h * 0.5) / H, w / W, h / H] for x, y, w, h in boxes]

class TorchDetectionDataset(Dataset):
    def __init__(self, dataset, processor, transform):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = np.array(example["image"].convert("RGB"))
        boxes = example["objects"]["bboxes"]
        classes = example["objects"]["classes"]

        aug = self.transform(image=image, bboxes=boxes, class_labels=classes)
        img_aug = aug["image"]
        boxes_aug = aug["bboxes"]
        cls_aug = aug["class_labels"]

        h, w = img_aug.shape[:2]
        boxes_norm = np.asarray(
            xywh_to_cxcywh_norm(boxes_aug, (w, h)),
            dtype=np.float32
        ).reshape(-1, 4)

        enc = self.processor(images=img_aug, return_tensors="pt")

        return {
            "pixel_values": enc["pixel_values"][0],
            "labels": {
                "class_labels": torch.tensor(cls_aug, dtype=torch.long),
                "boxes": torch.tensor(boxes_norm, dtype=torch.float32)
            }
        }

def collate_fn(batch):
    pad_value: float = 0.0
    batch = [b for b in batch if b is not None]

    pixel_list = [b["pixel_values"] for b in batch]

    max_h = max(img.shape[-2] for img in pixel_list)  # H
    max_w = max(img.shape[-1] for img in pixel_list)  # W

    # 4) 逐张进行 padding
    padded_imgs = []
    for img in pixel_list:  # img: [3, H, W]
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # F.pad 的顺序：(left, right, top, bottom)
        img = F.pad(img, (0, pad_w,  # width  方向 → 右侧 pad
                          0, pad_h),  # height 方向 → 下侧 pad
                    value=pad_value)
        padded_imgs.append(img)

    # 5) 组装 batch
    pixel_values = torch.stack(padded_imgs)  # [B, 3, max_h, max_w]
    labels = [b["labels"] for b in batch]

    return {"pixel_values": pixel_values, "labels": labels}

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

    img_processor = get_processor(model_name)
    tf = build_aug_pipelines()

    raw_ds = TorchDetectionDataset(raw_ds, processor=img_processor, transform=tf)

    return raw_ds, img_processor, collate_fn


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
