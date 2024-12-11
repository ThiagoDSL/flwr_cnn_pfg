"""vitexample: A Flower / PyTorch app with Vision Transformers."""

from collections import OrderedDict

import torch
import time
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
    Resize,
    CenterCrop,
)

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import datasets
from flwr_datasets.partitioner import IidPartitioner


def get_model(num_classes: int):
    """Return a pretrained ViT with all layers frozen except output head."""

    # Instantiate a pre-trained ViT-B on ImageNet
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # We're going to federated the finetuning of this model
    # using (by default) the Oxford Flowers-102 dataset. One easy way
    # to achieve this is by re-initializing the output block of the
    # ViT so it outputs 102 clases instead of the default 1k
    in_features = model.heads[-1].in_features
    model.heads[-1] = torch.nn.Linear(in_features, num_classes)

    # Disable gradients for everything
    model.requires_grad_(False)
    # Now enable just for output head
    model.heads.requires_grad_(True)

    return model


def set_vit_params(model, parameters):
    """Apply the parameters to model head."""
    finetune_layers = model.heads
    params_dict = zip(finetune_layers.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    finetune_layers.load_state_dict(state_dict, strict=True)


def get_vit_params(model):
    """Get parameters from model head as ndarrays."""
    finetune_layers = model.heads
    return [val.cpu().numpy() for _, val in finetune_layers.state_dict().items()]


def train_vit(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[0].to(device) if isinstance(batch, list) else batch["image"].to(device)
            labels = batch[1].to(device) if isinstance(batch, list) else batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader)


def test_vit(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device) if isinstance(batch, list) else batch["image"].to(device)
            labels = batch[1].to(device) if isinstance(batch, list) else batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


fds = None

# def get_dataset_partition(num_partitions: int, partition_id: int, dataset_name: str):
#     """Get Oxford Flowers datasets and partition it."""
#     global fds
#     if fds is None:
#         # Get dataset (by default Oxford Flowers-102) and create IID partitions
#         partitioner = IidPartitioner(num_partitions)
#         fds = FederatedDataset(
#             dataset=dataset_name, partitioners={"train": partitioner}
#         )

#     return fds.load_partition(partition_id)

def load_vit_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="tanganke/gtsrb",
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_val = partition.train_test_split(test_size=0.1, seed=42)

    train_partition = partition_train_val["train"].with_transform(apply_vit_train_transforms)
    val_partition = partition_train_val["test"].with_transform(apply_vit_eval_transforms)
    
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_partition, batch_size=batch_size, shuffle=True)
    return trainloader, valloader

def load_vit_test_data(batch_size: int):
    test_dataset = datasets.GTSRB(
        root="data", split="test", download=True, transform=apply_vit_eval_transforms
    )
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    return testloader

def apply_vit_eval_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            Resize((224, 224)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # batch["image"] = [transforms(img) for img in batch["image"]]
    return transforms(batch)


def apply_vit_train_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # batch["image"] = [transforms(img) for img in batch["image"]]
    return transforms(batch)