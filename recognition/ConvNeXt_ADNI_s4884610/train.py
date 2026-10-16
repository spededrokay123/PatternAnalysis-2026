"""
Training, validation and testing for ADNI classification.

Usage:
    python3 train.py --args
"""

import argparse
import time
import os
import torch
import torch.nn as nn
from dataset import get_dataloaders
from modules import BaselineCNN, ConvNeXt
import matplotlib.pyplot as plt


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
    if name == "convnext":
        return ConvNeXt()
    raise ValueError(f"Unknown model: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on ADNI AD vs NC")
    parser.add_argument("--model", default="baseline", choices=["baseline", "convnext"])
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


def plot_history(history, out_path, title):
    """Save loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid()

    ax_acc.plot(epochs, history["train_acc"], marker="o", label="Train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", label="Validation")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"{time.time() - start:.0f}s"
        )

    # Plot the loss and accuracy itself.
    plot_history(
        history,
        os.path.join(args.out_dir, f"{args.model}_curves.png"),
        f"{args.model} training curves",
    )

    # Final Validation of baseline on Rangpur, only gets run once.
    """
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Best val accuracy {best_val_acc:.3f} | Test loss {test_loss:.4f} | Test accuracy {test_acc:.3f}")
    """
