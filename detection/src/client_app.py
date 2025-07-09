"""flowertune-llm: A Flower / FlowerTune app."""

import os, warnings, torch
from platform import processor
from typing import Dict, Tuple

from omegaconf import DictConfig

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar

from transformers import TrainingArguments, Trainer

from .dataset import (
    get_processor,
    load_data,
    replace_keys,
)

from .models import (
    cosine_annealing,
    get_model,
    set_local_parameters,
    get_local_parameters,
)

from .utils import print_trainable_params, set_seed

# Avoid warnings
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    def __init__(
            self,
             model_cfg: DictConfig,
             train_cfg: DictConfig,
             trainset,
             collate_fn,
             processor,
             num_rounds,
             rank_choices,
             group_id,
             peft_name,
             fl_method,
             scaling_method,
    ): # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.collate_fn  = collate_fn
        self.processor  = processor

        # instantiate model
        self.model = get_model(model_cfg, rank_choices, group_id, peft_name, scaling_method)
        self.group_id = group_id
        self.peft_name = peft_name
        self.fl_method = fl_method

        print_trainable_params(self.model)

        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_local_parameters(self.model, parameters, self.group_id, self.peft_name, self.fl_method)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config["save_path"]
        self.model.config.use_cache = False

        # Construct trainer
        trainer = Trainer(
            model          = self.model,
            args           = self.training_arguments,
            train_dataset  = self.trainset,
            tokenizer      = self.processor,
            data_collator  = self.collate_fn,
        )

        # Do local training
        results = trainer.train()

        return (
            get_local_parameters(self.model, self.group_id, self.peft_name),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def client_fn(context: Context):
    """Create a Flower client representing a single organization."""
    # Fix random seed
    set_seed(42)
    edge_devices = ["agx-orin", "orin-nano", "rpi-5"]

    # Read from config
    # edge_device = context.node_config["edge-device"]
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Read device name
    if partition_id == 0:
        edge_device = "agx-orin"
    elif 1 <= partition_id <= 4:
        edge_device = "orin-nano"
    elif 5 <= partition_id <= 9:
        edge_device = "rpi-5"
    else:
        edge_device = "agx-orin"

    # Get initial model weights
    rank_choices_str = cfg.fl.rank_choices
    rank_choices = [int(r) for r in rank_choices_str.split(",")]

    rank_choices_map = dict(zip(edge_devices, rank_choices))
    rank = rank_choices_map[edge_device]
    device_index = edge_devices.index(edge_device)
    group_id = f"group_{device_index}"

    # Print Device Information
    if cfg.fl.peft_name != "fft":
        print(f"INFO :      Device: {edge_device} | Group: {group_id} | Rank: {rank}")

    # Let's get the client partition
    client_trainset, processor, collate_fn  = load_data(partition_id, num_partitions, cfg.dataset.name, cfg.model.name)


    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        collate_fn,
        processor,
        num_rounds,
        rank_choices,
        group_id,
        cfg.fl.peft_name,
        cfg.fl.fl_method,
        cfg.fl.scaling_method,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
