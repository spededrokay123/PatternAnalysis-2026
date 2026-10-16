"""
Training, validation and testing for ADNI classification.

Usage:
    python3 train.py --args
"""

import argparse
import os
import torch
from dataset import get_dataloaders
from modules import BaselineCNN


def get_device():
    """Use an NVIDIA GPU if available, then Apple, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name):
    """Return an initialised model from its name."""
    if name == "baseline":
        return BaselineCNN()
    raise ValueError(f"Unknown model: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on ADNI AD vs NC")
    parser.add_argument("--model", default="baseline", choices=["baseline"])
    parser.add_argument(
        "--data_root", default=os.path.expanduser("~/comp3710_data/AD_NC")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="outputs")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training set. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a validation or test loader.
    Returns (mean loss, accuracy, AD probabilities, true labels).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        all_labels.append(labels.cpu())

    return (
        total_loss / total,
        correct / total,
        torch.cat(all_probs),
        torch.cat(all_labels),
    )


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")
