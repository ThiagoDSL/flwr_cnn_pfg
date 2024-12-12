"""pytorch-example: A Flower / PyTorch app."""

import torch
from pytorch_example.strategy import CustomFedAvg
from pytorch_example.task import (
    Net,
    get_weights,
    set_weights,
    validate,
    load_data,
)
from torch.utils.data import DataLoader
from pytorch_example.task import load_test_data
from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def gen_evaluate_fn(
    valloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        net.load_state_dict(torch.load("pytorch_example/traffic_sign_model.pth", weights_only=True), strict=True)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = validate(net, valloader, device=device)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.001
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


# TODO: change this
# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]
    batch_size = context.run_config["batch-size"]

    # Initialize model parameters
    net = Net()
    net.load_state_dict(torch.load("pytorch_example/traffic_sign_model.pth", weights_only=True), strict=True)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation
    # THE TEST DATA IS SEEN BY THE MODEL ONLY ONCE, FOR FINAL EVALUATION
    testloader = load_test_data(batch_size=batch_size)

    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
