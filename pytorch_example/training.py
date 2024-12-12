import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_example.task import Net
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

# Define transformations

FM_NORMALIZATION = ((0.1307,), (0.3081,))
transform = Compose(
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

# Load GTSRB dataset
dataset = datasets.GTSRB(root="./data", split='train', transform=transform, download=True)
test_dataset = datasets.GTSRB(root="./data", split='test', transform=transform, download=True)

# Split into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.load_state_dict(torch.load("traffic_sign_model.pth", weights_only=True), strict=False)
model.to(device)
criterion = nn.CrossEntropyLoss()  # Use cross-entropy for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 25
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "traffic_sign_model.pth")
