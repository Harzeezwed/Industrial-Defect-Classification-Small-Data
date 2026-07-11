"""
Learning-curve analysis: accuracy vs. number of labeled examples per class.

This is the evaluation the challenge brief specifically asks for: showing
how quickly the model reaches usable accuracy as the number of labeled
defect images grows, rather than only reporting a single final score.

For each sample-size budget (e.g. 2, 4, 8, 16, ... images per class):
  1. Randomly draw that many labeled examples per class from the training
     pool (multiple random seeds to get a mean +/- std band).
  2. Fine-tune a fresh model on just that subset.
  3. Evaluate on the same held-out test set every time.
  4. Plot accuracy vs. sample size.
"""
import copy
from collections import defaultdict
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.evaluate import evaluate_model
from src.model import get_model
from src.train import train_model


def _subset_indices_per_class(dataset, samples_per_class: int, seed: int) -> List[int]:
    """Randomly pick up to `samples_per_class` indices for each class."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)

    # dataset is expected to be an ImageFolder (or a Subset thereof with
    # `.dataset.samples` / `.indices` available)
    samples = dataset.samples if hasattr(dataset, "samples") else dataset.dataset.samples
    indices = range(len(samples)) if hasattr(dataset, "samples") else dataset.indices

    for idx in indices:
        _, label = samples[idx]
        by_class[label].append(idx)

    chosen: List[int] = []
    for label, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        chosen.extend(idxs[:samples_per_class].tolist())

    return chosen


def run_learning_curve(
    train_pool,
    test_loader: DataLoader,
    classes: Sequence[str],
    device: torch.device,
    sample_sizes: Sequence[int] = (2, 4, 8, 16, 32),
    seeds: Sequence[int] = (0, 1, 2),
    epochs: int = 15,
    batch_size: int = 16,
) -> dict:
    """Sweep sample sizes, train+eval each, and return a results dict.

    `train_pool` should be the training-side dataset/subset to draw
    few-shot samples from (kept separate from `test_loader`, which stays
    fixed across every run for a fair comparison).
    """
    results = {"sample_sizes": list(sample_sizes), "mean_acc": [], "std_acc": []}

    for n in sample_sizes:
        accs = []
        for seed in seeds:
            indices = _subset_indices_per_class(train_pool, n, seed)
            subset = Subset(
                train_pool if not hasattr(train_pool, "dataset") else train_pool.dataset,
                indices,
            )
            loader = DataLoader(
                subset,
                batch_size=min(batch_size, max(2, len(subset) // 2 * 2)),
                shuffle=True,
                drop_last=len(subset) > batch_size,
            )

            model = get_model(len(classes)).to(device)
            train_model(model, loader, device, epochs=epochs)
            acc = evaluate_model(model, test_loader, classes, device, plot=False)
            accs.append(acc)

        results["mean_acc"].append(float(np.mean(accs)))
        results["std_acc"].append(float(np.std(accs)))
        print(f"n={n} samples/class -> accuracy {np.mean(accs) * 100:.1f}% (+/- {np.std(accs) * 100:.1f})")

    return results


def plot_learning_curve(results: dict, target_accuracy: float = 0.85) -> None:
    sizes = results["sample_sizes"]
    mean_acc = np.array(results["mean_acc"]) * 100
    std_acc = np.array(results["std_acc"]) * 100

    plt.figure(figsize=(9, 6))
    plt.plot(sizes, mean_acc, marker="o", label="Mean accuracy")
    plt.fill_between(sizes, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, label="+/- 1 std")
    plt.axhline(target_accuracy * 100, color="red", linestyle="--", label=f"Target ({target_accuracy*100:.0f}%)")
    plt.xlabel("Labeled images per class")
    plt.ylabel("Classification accuracy (%)")
    plt.title("Accuracy vs. Number of Labeled Examples (Few-Shot Learning Curve)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/learning_curve.png", dpi=150)
    plt.show()
