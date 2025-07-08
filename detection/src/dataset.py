from transformers import AutoTokenizer
from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

FDS = None  # Cache FederatedDataset


def get_tokenizer(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
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
    client_dataset = FDS.load_partition(partition_id, "train")
    print(f"[Client {partition_id}] Got {len(client_dataset)} samples.")

    client_dataset = client_dataset.rename_column("output", "response")

    return client_dataset


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
