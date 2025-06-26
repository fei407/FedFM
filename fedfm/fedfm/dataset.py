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

    # Load the full dataset
    dataset = load_dataset(dataset_name, split="train")
    total_size = len(dataset)

    # Define ratio: [10,10,10,10,10,1,1,1,1,1]
    weights = [10 if i < 5 else 1 for i in range(num_partitions)]
    total_weight = sum(weights)

    # Compute actual size per partition
    partition_sizes = [int(w / total_weight * total_size) for w in weights]

    # Adjust for rounding loss: give remaining examples to the last partition
    size_diff = total_size - sum(partition_sizes)
    partition_sizes[-1] += size_diff

    # Compute start/end index for this partition
    start_idx = sum(partition_sizes[:partition_id])
    end_idx = start_idx + partition_sizes[partition_id]

    # Slice and return the subset
    client_dataset = dataset.select(range(start_idx, end_idx))
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
