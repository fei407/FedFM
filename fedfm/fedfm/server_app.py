"""flowertune-llm: A Flower / FlowerTune app."""

import os
from datetime import datetime

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig

from .utils import print_trainable_params

from .models import get_global_model, get_global_parameters, set_global_parameters
from .dataset import replace_keys

# From: https://github.com/adap/flower/tree/main/examples/flowertune-llm
# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, rank_choices, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model

        print(f"INFO :      server round: {server_round}")
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_global_model(model_cfg, rank_choices)

            print("[INFO] Inspecting parameters...")

            if isinstance(parameters, list):
                print(f"[INFO] parameters is a list of length: {len(parameters)}")
                for i, param in enumerate(parameters):
                    print(
                        f"  - param[{i}]: type={type(param)}, shape={param.shape if isinstance(param, np.ndarray) else 'N/A'}")
                    print(
                        f"    values (first 3 elements): {param.flatten()[:3] if isinstance(param, np.ndarray) else param}")
            else:
                print(f"[ERROR] Unexpected type: {type(parameters)} — expected List of NDArrays")
                return

            set_global_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")
            print(f"INFO :      model is saved at'{save_path}/peft_{server_round}'")

        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the client's
    fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    avg_loss = sum(losses) / sum(examples)

    print(f"INFO :      train loss: {avg_loss:.6f}")
    return {"train_loss": avg_loss}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    edge_devices = ["rpi-5", "orin-nano", "agx-orin"]

    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    rank_choices_str = cfg.model.lora.rank_choices
    rank_choices = [int(r) for r in rank_choices_str.split(",")]
    rank_nums_str = cfg.model.lora.rank_nums
    rank_nums = [int(r) for r in rank_nums_str.split(",")]

    rank_choices_map = dict(zip(edge_devices, rank_choices))
    rank_nums_map = dict(zip(edge_devices, rank_nums))
    for device_name in edge_devices:
        rank = rank_choices_map[device_name]
        num = rank_nums_map[device_name]
        print(f"INFO :      Fine-tuning on [{device_name}] with rank [{rank}], number of devices: [{num}]")

    init_model = get_global_model(cfg.model, rank_choices)
    init_model_parameters = get_global_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # Print model info
    # for name, param in init_model.named_parameters():
    #     print(f"Parameter: {name}, Shape: {param.shape}, Dtype: {param.dtype}, Trainable: {param.requires_grad}")
    # print_trainable_params(init_model)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        min_available_clients=2,
        min_fit_clients=2,
        evaluate_fn=get_evaluate_fn(
            cfg.model, rank_choices, cfg.train.save_every_round, num_rounds, save_path
        ),
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)