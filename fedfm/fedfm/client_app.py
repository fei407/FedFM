"""fedfm: A Flower / PEFT app."""

import torch
from transformers import AutoModelForSequenceClassification
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import get_weights, load_data, set_weights, test, train
from .utils import set_seed
from peft import LoraConfig, get_peft_model


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, train_dataset, test_dataset, data_collator, learning_rate, local_epochs, peft_name):
        self.net            = net
        self.train_dataset  = train_dataset
        self.test_dataset   = test_dataset
        self.data_collator  = data_collator
        self.learning_rate  = learning_rate
        self.local_epochs   = local_epochs
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.peft_name      = peft_name
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.train_dataset,
            self.data_collator,
            self.learning_rate,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net, self.peft_name),
            len(self.train_dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.test_dataset, self.data_collator, self.device)
        return loss, len(self.test_dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    # Load model and data
    partition_id            = context.node_config["partition-id"]
    num_partitions          = context.node_config["num-partitions"]

    dataset_name            = context.run_config["dataset-name"]
    data_distribution       = context.run_config["data-distribution"]
    niid_alpha              = context.run_config["niid-alpha"]
    model_name              = context.run_config["model-name"]
    num_labels              = context.run_config["num-labels"]

    peft_name               = context.run_config["peft-name"]
    peft_rank               = context.run_config["peft-rank"]
    peft_inserted_modules   = context.run_config["peft-inserted-modules"]

    local_epochs            = context.run_config["local-epochs"]
    learning_rate           = context.run_config["learning-rate"]

    train_dataset, test_dataset, data_collator = load_data(data_distribution, niid_alpha, partition_id, num_partitions, dataset_name, model_name)
    net = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    peft_inserted_modules = [m.strip() for m in peft_inserted_modules.split(",")]
    task_type = "SEQ_CLS"
    if peft_name == "fedfft":
        net = net
    elif peft_name == "fedit":
        peft_config = LoraConfig(
            r=peft_rank,
            lora_alpha=peft_rank,
            lora_dropout=0.05,
            target_modules=peft_inserted_modules,
            task_type=task_type,
        )
        net = get_peft_model(net, peft_config)
    else:
        raise ValueError(f"This [{peft_name}] is a undefined fine-tuning method! Please choose: 'fedfft', 'fedit' ,or 'flash'.")

    # Return Client instance
    return FlowerClient(net, train_dataset, test_dataset, data_collator, learning_rate, local_epochs, peft_name).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)
