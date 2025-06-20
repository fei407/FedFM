"""fedfm: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import numpy as np

import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets.utils.logging import disable_progress_bar

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

import evaluate
import tempfile

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()

accuracy_metric = evaluate.load("accuracy")

fds = None  # Cache FederatedDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def load_data(data_distribution: str, niid_alpha: float, partition_id: int, num_partitions: int, dataset_name: str, model_name: str):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if data_distribution == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif data_distribution == "niid":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=niid_alpha,
            )

        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return partition_train_test["train"], partition_train_test["test"], data_collator

def train(net, train_dataset, data_collator, tokenizer, learning_rate, epochs, device):
    """Train the model on the training set."""
    temp_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=temp_dir,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        logging_steps=100,
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model.to(device)
    train_output  = trainer.train()
    loss = train_output.training_loss

    return loss


def test(net, test_dataset, data_collator, device):
    """Validate the model using a LoRA-merged copy of `net` without affecting the original model."""
    temp_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=temp_dir,
        eval_strategy="no",
        save_strategy="no",
        per_device_eval_batch_size=4,
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.model.to(device)
    eval_result = trainer.evaluate()

    print(f"eval_result = {eval_result}")

    if "eval_accuracy" not in eval_result:
        raise ValueError("eval_accuracy not in eval_result")

    if "eval_loss" not in eval_result:
        raise ValueError("eval_loss not in eval_result")

    loss = eval_result["eval_loss"]
    accuracy = eval_result["eval_accuracy"]

    return loss, accuracy


def get_weights(net, peft_name):
    if peft_name == "fedfft":
        selected_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    elif peft_name == "fedit":
        selected_params = [val.cpu().numpy() for name, val in net.state_dict().items() if "lora" in name]
    else:
        raise ValueError(f"This [{peft_name}] is a undefined fine-tuning method! Please choose: 'fedfft', 'fedit' ,or 'flash'.")

    return selected_params


def set_weights(net, parameters):
    state_dict = net.state_dict()

    if len(parameters) == len(state_dict):
        selected_keys = list(state_dict.keys())
        print("Detected full model update")
    else:
        selected_keys = [k for k in state_dict if "lora" in k]
        print("Detected LoRA-only update")

    new_state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(selected_keys, parameters)})

    net.load_state_dict(new_state_dict, strict=False)