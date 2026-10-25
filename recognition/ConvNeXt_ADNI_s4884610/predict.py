"""
Evaluation of the trained ADNI classifier.

Usage:
    python3 predict.py --model convnext                    # full evaluation
    python3 predict.py --model convnext --image scan.jpeg  # one image
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from dataset import get_dataloaders, get_transforms
from train import get_device, build_model

CLASS_NAMES = ["NC", "AD"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ADNI model")
    parser.add_argument("--model", default="convnext", choices=["baseline", "convnext"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--data_root", default=os.path.expanduser("~/comp3710_data/AD_NC")
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fig_dir", default="images")
    parser.add_argument("--out_dir", default="outputs")
    return parser.parse_args()


def load_model(name, checkpoint, device):
    """Build the model and load the trained weights."""
    model = build_model(name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_logits(model, loader, device):
    """
    Run the model over a loader and collect its logits for the temperature scaling
    Returns (logits [N, 2], labels [N])
    """
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)
