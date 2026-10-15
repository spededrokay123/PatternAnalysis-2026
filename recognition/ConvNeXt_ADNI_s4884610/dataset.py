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
