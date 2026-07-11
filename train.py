"""Training loop for the defect classification model."""
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> List[float]:
    """Fine-tune `model` on `loader` and return per-epoch average loss.

    Kept deliberately simple (Adam + cross-entropy, no LR schedule) since
    the emphasis of this project is on data efficiency, not on squeezing
    out marginal optimizer gains.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    history: List[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")

    return history
