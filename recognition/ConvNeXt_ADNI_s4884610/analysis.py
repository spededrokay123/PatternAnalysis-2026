"""
Extra analysis of trained ADNI models (not part of the model)
    - scan-level accuracy (average of the 20 slices of each scan)
    - test scans from subjects seen in training vs unseen subjects
    - horizontal-flip test-time augmentation
    - an ensemble that averages all the given runs
    - an AD threshold chosen on the validation set (never on test)

Usage:
    python3 analysis.py --data_root <path>/AD_NC \
        --run baseline:outputs_pc_baseline_bn \
        --run convnext:outputs_pc_reg
"""



import argparse
import json
import os
from collections import Counter
import torch
from dataset import get_dataloaders, get_patient_id
from train import get_device, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Extra analysis of ADNI models")
    parser.add_argument(
        "--data_root", default=os.path.expanduser("~/comp3710_data/AD_NC")
    )
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default=os.path.join("outputs", "analysis.json"))
    return parser.parse_args()


@torch.no_grad()
def get_probs(model, loader, device, flip=False):
    """
    AD probability for every slice
    With flip boolean true, average with the mirrored slice
    """
    all_probs = []
    for images, _ in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)[:, 1]
        if flip:
            flipped = torch.flip(images, dims=[3])  # mirror left to right
            probs = (probs + torch.softmax(model(flipped), dim=1)[:, 1]) / 2
        all_probs.append(probs.cpu())
    return torch.cat(all_probs)


def slice_accuracy(probs, samples, threshold=0.5):
    correct = 0
    for prob, (_, label, _) in zip(probs.tolist(), samples):
        if (prob > threshold) == label:
            correct += 1
    return correct / len(samples)


def scan_level(probs, samples):
    """
    Group slices by scan
    Returns {scan_id: (mean AD prob, label, subject)}
    """
    slice_probs, labels, subjects = {}, {}, {}
    for prob, (path, label, subject) in zip(probs.tolist(), samples):
        scan = get_patient_id(path)
        slice_probs.setdefault(scan, []).append(prob)
        labels[scan] = label
        subjects[scan] = subject

    scans = {}
    for scan in slice_probs:
        mean_prob = sum(slice_probs[scan]) / len(slice_probs[scan])
        scans[scan] = (mean_prob, labels[scan], subjects[scan])
    return scans


def scan_accuracy(scans, scan_ids, threshold=0.5):
    if len(scan_ids) == 0:
        return 0.0
    correct = 0
    for scan in scan_ids:
        mean_prob, label, _ = scans[scan]
        if (mean_prob > threshold) == label:
            correct += 1
    return correct / len(scan_ids)


def choose_threshold(val_scans):
    """Pick the AD threshold with the best val scan accuracy"""

    best_t, best_acc = 0.5, scan_accuracy(val_scans, list(val_scans), 0.5)
    for i in range(20, 81):
        t = i / 100
        acc = scan_accuracy(val_scans, list(val_scans), t)
        if acc > best_acc or (acc == best_acc and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_acc = t, acc
    return best_t


def summarise(val_probs, test_probs, val_samples, test_samples, seen_subjects):
    """All the numbers for one set of predictions."""
    val_scans = scan_level(val_probs, val_samples)
    test_scans = scan_level(test_probs, test_samples)

    seen = [s for s in test_scans if test_scans[s][2] in seen_subjects]
    unseen = [s for s in test_scans if test_scans[s][2] not in seen_subjects]
    threshold = choose_threshold(val_scans)

    return {
        "val_slice_acc": slice_accuracy(val_probs, val_samples),
        "val_scan_acc": scan_accuracy(val_scans, list(val_scans)),
        "test_slice_acc": slice_accuracy(test_probs, test_samples),
        "test_scan_acc": scan_accuracy(test_scans, list(test_scans)),
        "test_scan_acc_seen_subjects": scan_accuracy(test_scans, seen),
        "test_scan_acc_unseen_subjects": scan_accuracy(test_scans, unseen),
        "val_chosen_threshold": threshold,
        "test_scan_acc_at_val_threshold": scan_accuracy(
            test_scans, list(test_scans), threshold
        ),
    }


def split_counts(scans, seen_subjects):
    """How many test scans (and of which class) are from seen or unseen subjects."""
    counts = {"seen": Counter(), "unseen": Counter()}
    for _, label, subject in scans.values():
        group = "seen" if subject in seen_subjects else "unseen"
        counts[group]["AD" if label == 1 else "NC"] += 1
    return {group: dict(c) for group, c in counts.items()}


if __name__ == "__main__":
    args = parse_args()

    if not args.run:
        raise SystemExit("Give at least one --run <model>:<out_dir>")
    device = get_device()

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    # each sample is (image path, label, subject ID)
    train_samples = train_loader.dataset.samples
    val_samples = val_loader.dataset.samples
    test_samples = test_loader.dataset.samples

    # a test subject is "seen" if any of their scans was in train or val
    seen_subjects = {s[2] for s in train_samples} | {s[2] for s in val_samples}

    # data only results
    results = {
        # how many test scans of each class come from seen / unseen subjects.
        "test_scan_counts": split_counts(
            scan_level(torch.zeros(len(test_samples)), test_samples), seen_subjects
        ),
        "runs": {},
    }

    # actual model results
    flip_val, flip_test = [], []  # flipped predictions of every run, for the ensemble
    for run in args.run:
        # example "convnext:outputs_pc_reg" to model and folder
        model_name, out_dir = run.split(":", 1)
        model = build_model(model_name)
        checkpoint = os.path.join(out_dir, f"{model_name}_best.pt")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        model.eval()

        # Ad probability per slice
        val_probs = get_probs(model, val_loader, device)
        test_probs = get_probs(model, test_loader, device)
        val_flip = get_probs(model, val_loader, device, flip=True)
        test_flip = get_probs(model, test_loader, device, flip=True)
        flip_val.append(val_flip)
        flip_test.append(test_flip)

        name = f"{model_name}:{out_dir}"

        # accuracy summaries
        results["runs"][name] = summarise(
            val_probs, test_probs, val_samples, test_samples, seen_subjects
        )
        results["runs"][name + " +flip"] = summarise(
            val_flip, test_flip, val_samples, test_samples, seen_subjects
        )

    # average the flipped AD probabilities of all runs
    if len(args.run) > 1:
        ensemble_val = sum(flip_val) / len(flip_val)
        ensemble_test = sum(flip_test) / len(flip_test)
        results["runs"]["ensemble of all runs +flip"] = summarise(
            ensemble_val, ensemble_test, val_samples, test_samples, seen_subjects
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # short printed summary (full numbers in the JSON).
    print(json.dumps(results["test_scan_counts"]))
    print(f"{'run':45s} {'test slice':>10s} {'test scan':>10s} {'seen':>6s} {'unseen':>7s}")
    for name, r in results["runs"].items():
        print(
            f"{name:45s} {r['test_slice_acc']:10.3f} {r['test_scan_acc']:10.3f} "
            f"{r['test_scan_acc_seen_subjects']:6.3f} {r['test_scan_acc_unseen_subjects']:7.3f}"
        )
    print(f"Saved {args.out}")
