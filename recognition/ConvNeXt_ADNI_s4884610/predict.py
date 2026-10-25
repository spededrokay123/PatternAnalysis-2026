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


def compute_metrics(probs, labels):
    """Computes the Classification metrics from the AD probs"""
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_AD": precision_score(labels, preds),
        "recall_AD": recall_score(labels, preds),
        "recall_NC": recall_score(labels, preds, pos_label=0),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "auroc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def plot_confusion_and_roc(probs, labels, save_path, title):
    """Produces confusion matrix and ROC curves side by side."""
    preds = (probs >= 0.5).astype(int)
    false_pos_rate, true_pos_rate, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    figure, (confusion_panel, roc_panel) = plt.subplots(1, 2, figsize=(10, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(labels, preds), display_labels=CLASS_NAMES
    ).plot(ax=confusion_panel, colorbar=False)
    confusion_panel.set_title("Confusion matrix")

    roc_panel.plot(false_pos_rate, true_pos_rate, label=f"AUROC {auroc:.3f}")
    roc_panel.plot([0, 1], [0, 1], "k--")
    roc_panel.set_xlabel("False positive rate")
    roc_panel.set_ylabel("True positive rate")
    roc_panel.set_title("ROC curve")
    roc_panel.legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)


def wait_for_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_resources(model, device, n_runs=100):
    """Measures the number of parameters, latency per slices in ms, and the peak VRAM in GB"""

    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 1, 240, 256, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(10):  # warm up gpu resources
            model(x)
        wait_for_device(device)
        start = time.time()
        for _ in range(n_runs):
            model(x)
        wait_for_device(device)
        latency_ms = (time.time() - start) / n_runs * 1000

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {"params": n_params, "latency_ms": latency_ms, "peak_vram_gb": peak_vram_gb}


def fit_temperature(logits, labels):
    """
    Temperature scaling finds a number T that minimises the cross-entropy
    of softmax(logits / T) on the validation set.
    """
    best_t, best_loss = 1.0, float("inf")
    for t in torch.arange(0.5, 5.01, 0.01):
        loss = torch.nn.functional.cross_entropy(logits / t, labels).item()
        if loss < best_loss:
            best_t, best_loss = t.item(), loss
    return best_t


def expected_calibration_error(confidences, correct, n_bins=10):
    """
    Computes the average gap between confidence and accuracy which is weighted by how many
    predictions fall in each confidence interval
    """

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.sum() > 0:
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
    return ece


def plot_reliability(conf_before, conf_after, correct, path, title):
    """Produces reliability diagrams before and after temperature scaling."""
    edges = np.linspace(0.5, 1.0, 11)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        (axes[0], conf_before, "Before temperature scaling"),
        (axes[1], conf_after, "After temperature scaling"),
    ]
    for ax, conf, name in panels:
        accs = []
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (conf > low) & (conf <= high)
            accs.append(correct[in_bin].mean() if in_bin.any() else 0.0)
        ece = expected_calibration_error(conf, correct)

        ax.bar(centres, accs, width=0.05, edgecolor="black", label="Accuracy")
        ax.plot([0.5, 1], [0.5, 1], "k--", label="Perfect calibration")
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} (ECE {ece:.3f})")
        ax.legend(loc="upper left")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
