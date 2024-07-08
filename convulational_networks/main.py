"""
Module for training a LeNet model on a specified dataset.

This script includes functionality for:
1. Extracting a dataset from a zip file.
2. Preparing data loaders for training and testing.
3. Defining the LeNet model.
4. Training the model and tracking performance metrics.
5. Saving the trained model and history.
"""

import os
import zipfile
from typing import Dict, List
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pandas as pd


@dataclass
class TrainConfig:
    """Configuration for training the model."""
    model: nn.Module
    dataloaders: Dict[str, DataLoader]
    criterion: nn.Module
    optimizer: optim.Optimizer
    device: torch.device
    num_epochs: int


class LeNet(nn.Module):
    """A class representing the LeNet model for image classification."""

    def __init__(self, num_classes: int):
        """Initialize the model architecture."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass of the model."""
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def extract_dataset(zip_path: str, extract_to: str):
    """Extract a dataset from a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def get_data_loaders(base_dir: str, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    """Prepare data loaders for training and testing."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(base_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    return {x: DataLoader(image_datasets[x], batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
            for x in ['train', 'test']}


def train_model(config: TrainConfig) -> Dict[str, List[float]]:
    """Train the model and return a history of loss and accuracy metrics."""
    dataset_sizes = {x: len(config.dataloaders[x].dataset) for x in ['train', 'test']}
    history: Dict[str, List[float]] = {'train_loss': [], 'train_acc': [],
                                       'test_loss': [], 'test_acc': []}

    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                config.model.train()
            else:
                config.model.eval()

            running_loss = 0.0
            correct = 0

            for inputs, labels in config.dataloaders[phase]:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = config.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = config.criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        config.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(correct) / dataset_sizes[phase] * 100
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

        print()

    return history


def save_model_and_history(model: nn.Module, history: Dict[str, List[float]],
                           model_path: str, history_path: str):
    """Save the model and training history to files."""
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)


def main():
    """Main function to execute the model training."""
    dataset_zip_path = 'pokemons.zip'
    extract_folder = 'pokemon_dataset'
    extract_dataset(dataset_zip_path, extract_folder)

    base_dir = os.path.join(extract_folder, 'pokemons')
    dataloaders = get_data_loaders(base_dir, 64, 4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet(num_classes=len(dataloaders['train'].dataset.classes))

    config = TrainConfig(
        model=model,
        dataloaders=dataloaders,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        device=device,
        num_epochs=25
    )

    history = train_model(config)
    save_model_and_history(model, history, 'lenet_model.pth', 'training_history.csv')


if __name__ == '__main__':
    main()
