"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

from omegaconf import DictConfig

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar

from trl import SFTConfig, SFTTrainer

from .dataset import (
    get_tokenizer,
    load_data,
    replace_keys,
)

from .models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
             tokenizer,
             num_rounds
    ): # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = SFTConfig(
            **train_cfg.training_arguments,
            dataset_text_field = "text",
            max_length = train_cfg.seq_length,
            completion_only_loss = True,
        )
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.tokenizer = tokenizer
        # instantiate model
        self.model = get_model(model_cfg)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        try:
            """Implement distributed fit function for a given client."""
            set_parameters(self.model, parameters)

            new_lr = cosine_annealing(
                int(config["current_round"]),
                self.num_rounds,
                self.train_cfg.learning_rate_max,
                self.train_cfg.learning_rate_min,
            )

            self.training_arguments.learning_rate = new_lr
            self.training_arguments.output_dir = config["save_path"]

            self.model.enable_input_require_grads()
            self.model.config.use_cache = False

            # Construct trainer
            trainer = SFTTrainer(
                model=self.model,
                args=self.training_arguments,
                train_dataset=self.trainset,
                processing_class=self.tokenizer,
            )

            # Do local training
            results = trainer.train()

            return (
                get_parameters(self.model),
                len(self.trainset),
                {"train_loss": results.training_loss},
            )

        except Exception as e:
            print("Training failed!:", str(e))
            raise e


def client_fn(context: Context):
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
    tokenizer = get_tokenizer(cfg.model.name)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
