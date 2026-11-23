from collections import defaultdict

import numpy as np
from PIL import Image

from scripts.prepare_data import leaf_key, prepare_plantdoc, prepare_plantvillage


def _save(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (24, 24, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def test_leaf_key_strips_photo_index():
    assert leaf_key("ab12___GHLB2 Leaf 121.3.JPG") == "GHLB2 Leaf 121"
    assert leaf_key("ab12___GHLB2 Leaf 121.JPG") == "GHLB2 Leaf 121"
    assert leaf_key("ab12___RS_HL 0105.JPG") == "RS_HL 0105"


def test_plantvillage_split_keeps_groups_together(tmp_path):
    root = tmp_path / "plantvillage"
    seed = 0
    for label in ("Tomato___healthy", "Tomato___Late_blight"):
        for i in range(40):
            for photo in ("", ".1", ".2") if i < 5 else ("",):  # 5 leaves photographed 3 times
                name = f"u{seed:04d}___Leaf {i}{photo}"
                _save(root / "raw" / "color" / label / f"{name}.png", seed)
                _save(root / "raw" / "segmented" / label / f"{name}_final_masked.jpg", seed + 10_000)
                seed += 1
    # an exact duplicate under another name, and a duplicate with a conflicting label
    dup = (root / "raw" / "color" / "Tomato___healthy" / "u0000___Leaf 0.png").read_bytes()
    (root / "raw" / "color" / "Tomato___healthy" / "zzzz___Copy 1.png").write_bytes(dup)
    (root / "raw" / "color" / "Tomato___Late_blight" / "yyyy___Conflict 1.png").write_bytes(dup)

    rows, audit = prepare_plantvillage(root, seed=42)
    assert audit["images"] == len(rows) == 2 * (40 + 5 * 2) + 2
    assert audit["multi_image_groups"] >= 10
    assert audit["exact_duplicate_files"] == 2
    assert audit["label_conflict_groups"] == 1
    splits_per_group = defaultdict(set)
    for row in rows:
        splits_per_group[row["group"]].add(row["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    conflict_rows = [r for r in rows if "Conflict" in r["path"] or "Copy" in r["path"]]
    assert {r["split"] for r in conflict_rows} == {"train"}  # ambiguous duplicates never reach evaluation
    assert all(r["mask_path"] for r in rows if "Copy" not in r["path"] and "Conflict" not in r["path"])


def test_plantdoc_drops_test_images_that_duplicate_train(tmp_path):
    root = tmp_path / "plantdoc"
    _save(root / "train" / "Tomato leaf" / "a.png", 1)
    _save(root / "train" / "Tomato leaf late blight" / "b.png", 2)
    _save(root / "test" / "Tomato leaf" / "c.png", 3)
    (root / "test" / "Tomato leaf late blight").mkdir(parents=True)
    (root / "test" / "Tomato leaf late blight" / "dup.png").write_bytes((root / "train" / "Tomato leaf" / "a.png").read_bytes())
    (root / "test" / "Tomato leaf" / "broken.jpg").write_bytes(b"not an image")

    rows, audit = prepare_plantdoc(root)
    assert audit["test_images_duplicating_train_removed"] == 1
    assert audit["unreadable_files_skipped"] == 1
    assert {(r["split"], r["label"]) for r in rows} == {
        ("train", "Tomato___healthy"), ("train", "Tomato___Late_blight"), ("test", "Tomato___healthy"),
    }
