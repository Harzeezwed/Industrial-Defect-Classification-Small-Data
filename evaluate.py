"""Evaluation utilities: accuracy, classification report, confusion matrix."""
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    classes: Sequence[str],
    device: torch.device,
    plot: bool = True,
    title: str = "Defect Classification Confusion Matrix",
) -> float:
    """Run inference over `loader`, print metrics, and return overall accuracy."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    print("\nDetailed Classification Report:")
    print(
        classification_report(
            all_labels, all_preds, labels=range(len(classes)), target_names=classes
        )
    )

    if plot:
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues"
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.tight_layout()
        plt.savefig("results/confusion_matrix.png", dpi=150)
        plt.show()

    return accuracy


def plot_training_loss(history: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.xticks(range(1, len(history) + 1))
    plt.tight_layout()
    plt.savefig("results/training_loss.png", dpi=150)
    plt.show()
