"""
End-to-end entry point.

Usage:
    python -m src.main --data-root data/small_sample_20 --epochs 20

Run `python -m src.main --help` for all options.
"""
import argparse
import os

import torch

from src.dataset import load_dataloaders
from src.evaluate import evaluate_model, plot_training_loss
from src.model import get_model
from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Small-sample defect classification")
    parser.add_argument("--data-root", type=str, required=True, help="Path to ImageFolder-style dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, classes = load_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_split=args.train_split,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Classes detected: {classes}")

    model = get_model(num_classes=len(classes)).to(device)

    history = train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr)
    plot_training_loss(history)

    print("\nEvaluating on training data:")
    evaluate_model(model, train_loader, classes, device, title="Confusion Matrix (Train)")

    print("\nEvaluating on held-out test data:")
    evaluate_model(model, test_loader, classes, device, title="Confusion Matrix (Test)")

    torch.save(model.state_dict(), "results/model.pt")
    print("\nSaved trained weights to results/model.pt")


if __name__ == "__main__":
    main()
