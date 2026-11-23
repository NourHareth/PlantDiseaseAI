import numpy as np
from PIL import Image

from plantdisease.pipeline import class_names_for, make_dataset, read_manifest, write_manifest


def _fake_images(root, n_per_class=3):
    """Leaf = green square in the middle; background = flat grey. Masks like PlantVillage's."""
    rows = []
    for label in ("Tomato___healthy", "Tomato___Late_blight"):
        (root / "color" / label).mkdir(parents=True)
        (root / "seg" / label).mkdir(parents=True)
        for i in range(n_per_class):
            photo = np.full((64, 64, 3), 128, np.uint8)
            photo[16:48, 16:48] = (40, 160, 40)
            segmented = np.zeros_like(photo)
            segmented[16:48, 16:48] = photo[16:48, 16:48]
            Image.fromarray(photo).save(root / "color" / label / f"{i}.png")
            Image.fromarray(segmented).save(root / "seg" / label / f"{i}.png")
            rows.append({"path": f"color/{label}/{i}.png", "mask_path": f"seg/{label}/{i}.png",
                         "label": label, "split": "train", "group": f"{label}{i}"})
    return rows


def test_manifest_round_trip(tmp_path):
    rows = [{"path": "a b/c.jpg", "mask_path": "", "label": "X", "split": "test", "group": "g1"}]
    write_manifest(tmp_path / "m.csv", rows)
    assert read_manifest(tmp_path / "m.csv") == rows
    assert read_manifest(tmp_path / "m.csv", split="train") == []


def test_class_names_sort_like_keras():
    rows = [{"label": n} for n in ("Tomato___healthy", "Tomato___Late_blight", "Tomato___Bacterial_spot")]
    assert class_names_for(rows) == ["Tomato___Bacterial_spot", "Tomato___Late_blight", "Tomato___healthy"]


def test_dataset_batches_and_labels(tmp_path):
    rows = _fake_images(tmp_path)
    names = class_names_for(rows)
    images, labels = next(iter(make_dataset(rows, tmp_path, names, 32, 6)))
    assert images.shape == (6, 32, 32, 3)
    assert float(images.numpy().min()) >= 0.0 and float(images.numpy().max()) <= 1.0
    assert sorted(labels.numpy().tolist()) == [0, 0, 0, 1, 1, 1]


def test_background_replacement_changes_only_the_background(tmp_path):
    rows = _fake_images(tmp_path, n_per_class=2)
    names = class_names_for(rows)
    plain, _ = next(iter(make_dataset(rows, tmp_path, names, 64, 4)))
    swapped, _ = next(iter(make_dataset(rows, tmp_path, names, 64, 4, background_prob=1.0, seed=3)))
    plain, swapped = plain.numpy(), swapped.numpy()
    leaf = (slice(None), slice(22, 42), slice(22, 42))  # well inside the leaf (mask edges are feathered)
    corner = (slice(None), slice(0, 10), slice(0, 10))
    assert np.allclose(plain[leaf], swapped[leaf], atol=0.02)
    assert np.abs(plain[corner] - swapped[corner]).mean() > 0.05
