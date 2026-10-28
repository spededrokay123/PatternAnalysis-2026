"""
Extra analysis of trained ADNI models (not part of the model)
    - scan-level accuracy (average of the 20 slices of each scan)
    - test scans from subjects seen in training vs unseen subjects
    - horizontal-flip test-time augmentation
    - an ensemble that averages all the given runs
    - an AD threshold chosen on the validation set (never on test)
    - whether unseen test subjects come from scanner sites missing from training
    - acquisition year and MRI sequence per class, read from the raw file
      name in meta_data_with_label.json

Usage:
    python3 analysis.py --data_root <path>/AD_NC \
        --run baseline:outputs_pc_baseline_bn \
        --run convnext:outputs_pc_reg
"""



import argparse
import json
import os
import re
from collections import Counter
import torch
from dataset import get_dataloaders, get_patient_id, load_subject_map
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


def site_analysis(train_samples, test_samples, seen_subjects):
    """
    Subject IDs are formatted as 068_S_0473, 068 is the scanner site
    Counts unseen test scans whose site has no training scans
    """
    train_sites = {subject[:3] for _, _, subject in train_samples}
    test_scans = scan_level(torch.zeros(len(test_samples)), test_samples)

    unseen_scans, new_site_scans = 0, 0
    for _, _, subject in test_scans.values():
        if subject not in seen_subjects:
            unseen_scans += 1
            if subject[:3] not in train_sites:
                new_site_scans += 1
    return {
        "unseen_test_scans": unseen_scans,
        "unseen_test_scans_from_sites_not_in_train": new_site_scans,
    }


def load_scan_info(data_root):
    """
    Map each scan ID too (acquisition year, MRI sequence) from the metadata "raw" name
    """
    meta_path = os.path.join(
        os.path.dirname(os.path.normpath(data_root)), "meta_data_with_label.json"
    )
    with open(meta_path) as f:
        meta = json.load(f)

    info = {}
    for scan, entry in meta.items():
        year = re.search(r"_br_raw_(\d{4})", entry["raw"])
        sequence = re.search(r"_MR_(.+?)_+br_raw", entry["raw"])
        info[scan] = (
            year.group(1) if year else "?",
            sequence.group(1) if sequence else "?",
        )
    return info


def era(year):
    """Provides scan acquisition era"""
    if year == "?":
        return "unknown"
    return "<=2015" if int(year) <= 2015 else ">=2016"


def acquisition_table(samples, scan_info, keep):
    """Count scans per class by acquisition era and by MRI sequence"""
    scan_label = {}
    for path, label, subject in samples:
        if keep(subject):
            scan_label[get_patient_id(path)] = label

    table = {}
    for cls_label, cls_name in ((0, "NC"), (1, "AD")):
        eras, years, sequences = Counter(), Counter(), Counter()
        for scan, label in scan_label.items():
            if label == cls_label:
                year, sequence = scan_info.get(scan, ("?", "?"))
                eras[era(year)] += 1
                years[year] += 1
                sequences[sequence] += 1
        table[cls_name] = {
            "eras": dict(eras),
            "years": dict(sorted(years.items())),
            "sequences": dict(sequences.most_common()),
        }
    return table


def failure_details(eval_path, test_samples, seen_subjects, scan_info):
    """
    Subject, seen/unseen, acquisition year and sequence of the failure cases that
    predict.py saved
    """
    if not os.path.exists(eval_path):
        return []
    with open(eval_path) as f:
        failure_files = json.load(f).get("failure_cases", [])

    scan_subject = {get_patient_id(path): subject for path, _, subject in test_samples}
    details = []
    for path in failure_files:
        scan = get_patient_id(path)
        subject = scan_subject.get(scan, "?")
        year, sequence = scan_info.get(scan, ("?", "?"))
        details.append({
            "file": os.path.basename(path),
            "subject": subject,
            "seen_subject": subject in seen_subjects,
            "year": year,
            "sequence": sequence,
        })
    return details


def accuracy_by_era(test_probs, test_samples, scan_info):
    """Test scan level accuracy for each (class, acquisition era) group"""
    scans = scan_level(test_probs, test_samples)
    groups = {}
    for scan, (_, label, _) in scans.items():
        year, _ = scan_info.get(scan, ("?", "?"))
        key = f"{'AD' if label == 1 else 'NC'} {era(year)}"
        groups.setdefault(key, []).append(scan)
    return {
        key: {"scans": len(ids), "scan_acc": scan_accuracy(scans, ids)}
        for key, ids in sorted(groups.items())
    }


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

    # maps a scan ID to (acquisition year, MRI sequence)
    scan_info = load_scan_info(args.data_root)


    # count how many scans are mapped to a subject ID
    all_samples = train_samples + val_samples + test_samples
    all_scans = {get_patient_id(path) for path, _, _ in all_samples}
    subject_map = load_subject_map(args.data_root)
    mapped = sum(1 for scan in all_scans if scan in subject_map)

    # data only results
    results = {
        "scans_mapped_to_subject": {"mapped": mapped, "total": len(all_scans)},
        # scans per class by acquisition era / year / sequence  for each split
        "acquisition": {
            "train": acquisition_table(train_samples, scan_info, lambda s: True),
            "val": acquisition_table(val_samples, scan_info, lambda s: True),
            "test_seen": acquisition_table(
                test_samples, scan_info, lambda s: s in seen_subjects
            ),
            "test_unseen": acquisition_table(
                test_samples, scan_info, lambda s: s not in seen_subjects
            ),
        },
        # how many test scans of each class come from seen / unseen subjects.
        "test_scan_counts": split_counts(
            scan_level(torch.zeros(len(test_samples)), test_samples), seen_subjects
        ),
        "sites": site_analysis(train_samples, test_samples, seen_subjects),
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

        # accuracy split by class and acquisition era
        results["runs"][name]["test_acc_by_era"] = accuracy_by_era(
            test_probs, test_samples, scan_info
        )

        # who the failure cases picked by predict.py are (needs predict.py to be run first)
        results["runs"][name]["failure_cases"] = failure_details(
            os.path.join(out_dir, f"{model_name}_eval.json"),
            test_samples,
            seen_subjects,
            scan_info,
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
    print(json.dumps(results["test_scan_counts"]), json.dumps(results["sites"]))
    print(f"scans mapped to a subject: {results['scans_mapped_to_subject']}")
    for group, table in results["acquisition"].items():
        print(
            f"scans by acquisition era, {group}: "
            f"NC {table['NC']['eras']} | AD {table['AD']['eras']}"
        )
        print(f"    years: NC {table['NC']['years']} | AD {table['AD']['years']}")
    for name, r in results["runs"].items():
        for case in r.get("failure_cases", []):
            print(f"failure case, {name}: {case}")
    for name, r in results["runs"].items():
        if "test_acc_by_era" in r:
            parts = [
                f"{k}: {v['scan_acc']:.3f} ({v['scans']})"
                for k, v in r["test_acc_by_era"].items()
            ]
            print(f"test scan acc by era, {name}: " + " | ".join(parts))
    print(f"{'run':45s} {'test slice':>10s} {'test scan':>10s} {'seen':>6s} {'unseen':>7s}")
    for name, r in results["runs"].items():
        print(
            f"{name:45s} {r['test_slice_acc']:10.3f} {r['test_scan_acc']:10.3f} "
            f"{r['test_scan_acc_seen_subjects']:6.3f} {r['test_scan_acc_unseen_subjects']:7.3f}"
        )
    print(f"Saved {args.out}")
