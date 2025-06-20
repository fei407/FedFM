"""fedfm: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from transformers import AutoModelForSequenceClassification

from .utils import set_seed, print_trainable_params
from peft import LoraConfig, get_peft_model

def aggregate_accuracy(results):
    accuracies = [metrics["accuracy"] * num_examples for num_examples, metrics in results]
    total_samples = sum(num_examples for num_examples, _ in results)

    aggregated_accuracy = sum(accuracies) / total_samples if total_samples > 0 else 0.0

    print(f"[DEBUG] Aggregated Accuracy: {aggregated_accuracy:.4f}")

    return {"accuracy": aggregated_accuracy}

def server_fn(context: Context):
    # Read from config
    set_seed(42)

    num_rounds              = context.run_config["num-server-rounds"]
    min_available_clients   = context.run_config["min-available-clients"]
    min_fit_clients         = context.run_config["min-fit-clients"]
    min_evaluate_clients    = context.run_config["min-evaluate-clients"]
    fraction_fit            = context.run_config["fraction-fit"]
    model_name              = context.run_config["model-name"]
    num_labels              = context.run_config["num-labels"]

    peft_name               = context.run_config["peft-name"]
    peft_rank               = context.run_config["peft-rank"]
    peft_inserted_modules   = context.run_config["peft-inserted-modules"]

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

    for name, param in net.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}, Dtype: {param.dtype}, Trainable: {param.requires_grad}")

    # Print trainbale_params
    print_trainable_params(net)

    initial_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        initial_parameters=initial_parameters,
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        # evaluate_metrics_aggregation_fn=aggregate_accuracy,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)