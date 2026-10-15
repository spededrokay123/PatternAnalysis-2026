"""
Data loading and preprocessing for ADNI AD vs NC classification

 Expected layout:
      <data_root>/train/AD/*.jpeg
      <data_root>/train/NC/*.jpeg
      <data_root>/test/AD/*.jpeg
      <data_root>/test/NC/*.jpeg

Filenames are <patientID>_<sliceNumber>.jpeg, with 20 slices per patient
"""

import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

CLASSES = {"NC": 0, "AD": 1}
IMG_EXTS = (".jpeg", ".jpg", ".png")


def get_patient_id(filename):
    """Return the patient ID from a filename of the above format"""
    return os.path.basename(filename).split("_")[0]


def gather_samples(data_root, split):
    """List every (image_path, label, patient_id) in data_root/split/{AD,NC}."""
    samples = []
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(data_root, split, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(IMG_EXTS):
                samples.append(
                    (os.path.join(class_dir, fname), label, get_patient_id(fname))
                )
    return samples


def split_by_patient(samples, val_fraction=0.15, seed=42):
    """
    Split samples into train/val by patient to esnure that each
    patient is only in one split. Uses stratified sampling on the label so the AD/NC ratio
    is preserved in both splits.
    """
    patient_label = {pid: lbl for _, lbl, pid in samples}  # one label per patient
    patients = sorted(patient_label)
    labels = [patient_label[p] for p in patients]

    train_pids, val_pids = train_test_split(
        patients, test_size=val_fraction, stratify=labels, random_state=seed
    )

    val_pids = set(val_pids)
    train = [s for s in samples if s[2] not in val_pids]
    val = [s for s in samples if s[2] in val_pids]
    return train, val


class ADNIDataset(Dataset):
    """Returns (image_tensor, label) for each MRI slice."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert("L")  # force grey scale
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class PerImageStandardise:
    """Scale each image to zero mean and variance to 1 to ensure same scale of intensity
    from scan to scan.
    """

    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-8)


def get_transforms(train):
    """Transforms the training images with flips, rotations, scaling, etc, while
    both the training and test gets turned into tensors and standardised
    """
    steps = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
        ]
    steps += [transforms.ToTensor(), PerImageStandardise()]
    return transforms.Compose(steps)


def get_dataloaders(
    data_root, batch_size=32, val_fraction=0.15, num_workers=4, seed=42
):
    """Build train/val/test DataLoaders based on patient split."""
    train_samples, val_samples = split_by_patient(
        gather_samples(data_root, "train"), val_fraction, seed
    )
    test_samples = gather_samples(data_root, "test")

    train_ds = ADNIDataset(train_samples, get_transforms(train=True))
    val_ds = ADNIDataset(val_samples, get_transforms(train=False))
    test_ds = ADNIDataset(test_samples, get_transforms(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
