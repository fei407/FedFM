"""fedfm: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification

from fedfm.task import get_weights


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    min_available_clients = context.run_config["min_available_clients"]
    min_fit_clients = context.run_config["min_fit_clients"]
    min_evaluate_clients = context.run_config["min_evaluate_clients"]

    # Initialize global model
    model_name = context.run_config["model-name"]
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
