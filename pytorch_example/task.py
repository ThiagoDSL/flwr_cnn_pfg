"""pytorch-example: A Flower / PyTorch app."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import time

import torch
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomAffine,
    ToTensor,
    Resize,
    Grayscale,
)

from flwr.common.typing import UserConfig

FM_NORMALIZATION = ((0.1307,), (0.3081,))
# TEST_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TEST_TRANSFORMS = Compose(
    [
        Grayscale(num_output_channels=1),
        Resize((32, 32)),
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(degrees=10, shear=0.1),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)
TRAIN_TRANSFORMS = Compose(
    [
        Grayscale(num_output_channels=1),
        Resize((32, 32)),
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(degrees=10, shear=0.1),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)


# class TrafficSignModel(nn.Module):
class Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""
    
    # We neet to specify the images format and the number of classes, this can be delegated
    # to the configuration files -> context.dimensions smth smth
    def __init__(self):
        super(Net, self).__init__()

        # 32x32 -> Conv -> Relu -> BN -> POOL -> 16x16
        # Dimensions: (Channels, Height, Width)
        # self.conv1 = nn.Conv2d(in_channels=dimensions[0], out_channels=32, kernel_size=(4,4), padding=2)  # padding=2 for 'same' padding
        # in_channels=1 means grayscale, out_channels=32 means image is 32x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4), padding=2)  # padding=2 for 'same' padding
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        # 16x16 -> (Conv -> Relu -> BN)*2 -> 8x8 
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), padding=2)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=2)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 8x8 -> (Conv -> Relu -> BN)*3 -> 4x4
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=2)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, padding=2)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, padding=2)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the flattened size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 32, 32)  # Batch size 1, single-channel, 32x32
            sample_output = self._forward_conv_layers(sample_input)
            flattened_size = sample_output.view(1, -1).size(1)

        # self.fc1 = nn.Linear(in_features=128 * (dimensions[1] // 8) * (dimensions[2] // 8), out_features=128)
        # self.fc1 = nn.Linear(in_features=flattened_size, out_features=128)
        # self.bn_fc1 = nn.BatchNorm1d(128)
        # self.dropout1 = nn.Dropout(0.7)

        # self.fc2 = nn.Linear(128, 128)
        # self.bn_fc2 = nn.BatchNorm1d(128)
        # self.dropout2 = nn.Dropout(0.5)
        
        # Replace BatchNorm1d with LayerNorm
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=128)
        self.bn_fc1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(0.7)

        self.fc2 = nn.Linear(128, 128)
        self.bn_fc2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.5)

        # GTSRB 43
        num_classes = 43
        self.fc3 = nn.Linear(128, num_classes)

    def _forward_conv_layers(self, x):
        """Forward pass through the convolutional layers only."""
        x = self.pool1(self.bn1(F.relu(self.conv1(x))))
        x = self.bn2_1(F.relu(self.conv2_1(x)))
        x = self.pool2(self.bn2_2(F.relu(self.conv2_2(x))))
        x = self.bn3_1(F.relu(self.conv3_1(x)))
        x = self.bn3_2(F.relu(self.conv3_2(x)))
        x = self.pool3(self.bn3_3(F.relu(self.conv3_3(x))))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten
        x = self.bn_fc1(F.relu(self.fc1(x)))
        x = self.dropout1(x)
        x = self.bn_fc2(F.relu(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    
    # trainloader handles one-hot encoding automatically
    # data augmentation was done on trainloader
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # TODO: search more about this
    # avg train loss might not be the best one for our purpose
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def validate(net, valloader, device):
    """Validate the model on the validation set."""
    net.to(device)
    net.eval()
    correct, loss = 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()#.to(device)
    with torch.no_grad():
        for batch in valloader:
            images = batch[0].to(device) if isinstance(batch, list) else batch["image"].to(device)
            labels = batch[1].to(device) if isinstance(batch, list) else batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(valloader.dataset)
    loss = loss / len(valloader)
    return loss, accuracy

def test(net, testloader, device):
    net.eval()
    correct = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = net(images)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def apply_train_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
    return batch


def apply_eval_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    # batch["image"] = [TEST_TRANSFORMS(img) for img in batch["image"]]
    batch["image"] = TEST_TRANSFORMS(batch["image"])
    return {"image": batch["image"], "label": batch["label"]}


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        # train_dataset = datasets.GTSRB(
        #     root="pytorch_example/data", split="train", download=True, transform=TRAIN_TRANSFORMS
        # )
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="kuchidareo/chinese_trafficsign_dataset",
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_val = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_val["train"].with_transform(apply_train_transforms)
    val_partition = partition_train_val["test"].with_transform(apply_train_transforms)
    
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_partition, batch_size=batch_size, shuffle=True)
    return trainloader, valloader


def load_test_data(batch_size: int):
    dataset = load_dataset("kuchidareo/chinese_trafficsign_dataset", split="train")

    # Apply transformation
    test_dataset = []
    for batch in dataset:
        test_dataset.append(apply_eval_transforms(batch))

    # Convert the dataset to PyTorch DataLoader
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return testloader


def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    # Assumes Linux / Git Bash style commands
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
