"""flowertune-llm: A Flower / FlowerTune app."""

import torch
import os
from datetime import datetime
from typing import Optional, Union

from flwr.common import (
    Context,
    Parameters,
    FitRes,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.config import unflatten_dict
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from logging import WARNING
from omegaconf import DictConfig

from .models import get_model, get_global_parameters, set_global_parameters
from .dataset import replace_keys
from .hetero import update_global_model, custom_aggregate
from .utils import set_seed

# From: https://github.com/adap/flower/tree/main/examples/flowertune-llm

class CustomFedAvg(FedAvg):
    """FedAvg with custom parameter aggregation logic."""
    def __init__(self, global_model, rank_choices, fl_method, peft_name, scaling_method, rmax, ridge_lamda, solve_method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model = global_model
        self.rank_choices = rank_choices
        self.fl_method = fl_method
        self.peft_name = peft_name
        self.scaling_method = scaling_method
        self.rmax = rmax
        self.ridge_lamda = ridge_lamda
        self.solve_method = solve_method

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        ## Custom Logic for Hetero-aggregation ##
        client_param_dicts = []
        num_examples_list = []

        group_id = "group_2"
        state_keys = [
            k.replace(f".{group_id}", "")
            for k in self.global_model.state_dict().keys()
            if "lora_" in k and group_id in k
        ]

        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            param_dict = {k: torch.tensor(v) for k, v in zip(state_keys, params)}
            client_param_dicts.append(param_dict)
            num_examples_list.append(fit_res.num_examples)

        # hetero-aggragation and construction
        aggregated_params = custom_aggregate(client_param_dicts, num_examples_list, self.global_model, self.fl_method, self.peft_name, self.scaling_method, self.rmax)
        update_global_model(self.global_model, aggregated_params, self.fl_method, self.peft_name, self.rank_choices, self.solve_method, self.ridge_lamda, self.scaling_method)

        aggregated_ndarrays = get_global_parameters(self.global_model, self.peft_name, self.fl_method)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        ## Custom Logic for Hetero-aggregation ##

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log.warning(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, rank_choices, save_every_round, total_round, save_path, peft_name, scaling_method, peft_init, fl_method):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model

        print(f"INFO :      server round: {server_round}")
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg, rank_choices, "group_2", peft_name, scaling_method, peft_init)

            set_global_parameters(model, parameters, peft_name, fl_method)

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
    set_seed(42)

    edge_devices = ["rpi-5", "orin-nano", "agx-orin"]

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))


    # Create output directory given current timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{cfg.fl.peft_name}_{cfg.fl.fl_method}_{timestamp}"
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Get initial model weights
    rank_choices_str = cfg.model.lora.rank_choices
    rank_choices = [int(r) for r in rank_choices_str.split(",")]
    rmax = max(rank_choices)
    rank_nums_str = cfg.model.lora.rank_nums
    rank_nums = [int(r) for r in rank_nums_str.split(",")]

    rank_choices_map = dict(zip(edge_devices, rank_choices))
    rank_nums_map = dict(zip(edge_devices, rank_nums))

    for device_name in edge_devices:
        rank = rank_choices_map[device_name]
        num = rank_nums_map[device_name]
        if cfg.fl.peft_name != "fft":
            print(f"INFO :      Fine-tuning on [{device_name}] with rank [{rank}], number of devices: [{num}]")

    global_model = get_model(cfg.model, rank_choices, "group_2", cfg.fl.peft_name, cfg.fl.scaling_method, cfg.fl.peft_init)

    init_model_ndarrays = get_global_parameters(global_model, cfg.fl.peft_name, cfg.fl.fl_method)
    init_model_parameters = ndarrays_to_parameters(init_model_ndarrays)

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        evaluate_fn=get_evaluate_fn(
            cfg.model, rank_choices, cfg.train.save_every_round, num_rounds, save_path, cfg.fl.peft_name, cfg.fl.scaling_method, cfg.fl.peft_init, cfg.fl.fl_method
        ),
        global_model=global_model,
        rank_choices=rank_choices,
        fl_method = cfg.fl.fl_method,
        peft_name = cfg.fl.peft_name,
        scaling_method = cfg.fl.scaling_method,
        rmax=rmax,
        ridge_lamda = cfg.fl.ridge_lamda,
        solve_method = cfg.fl.solve_method,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)