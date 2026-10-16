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
