"""Build reproducible, leakage-aware split manifests and a data audit.

    python -m scripts.prepare_data

PlantVillage: 70/15/15 train/val/test, stratified by class. Images are first
grouped so that a group never straddles two splits:

* repeat photos of the same leaf (``... Leaf 121``, ``... Leaf 121.1``, ...),
* byte-identical files (MD5),
* visually identical files (equal 64-bit difference hash, dHash).

PlantDoc: its own train/test split is kept; test images that duplicate a train
image are dropped from the test set.

Writes ``data/splits/plantvillage_tomato.csv``, ``plantdoc_tomato.csv`` and
``audit.json`` (all small and versioned in git).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.prepare_data)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/prepare_data.py)

import numpy as np
from PIL import Image

from plantdisease import config
from plantdisease.datasets import IMAGE_EXTENSIONS
from plantdisease.pipeline import write_manifest

SPLIT_FRACTIONS = {"test": 0.15, "val": 0.15}  # the rest is train

PLANTDOC_TO_PLANTVILLAGE = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}


# ---------------------------------------------------------------------------
# Hashing and grouping
# ---------------------------------------------------------------------------
def fingerprints(path: Path) -> tuple[str, int]:
    """(MD5 of the bytes, 64-bit dHash of the picture)."""
    data = path.read_bytes()
    with Image.open(path) as img:
        grey = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (grey[:, 1:] > grey[:, :-1]).flatten()
    return hashlib.md5(data).hexdigest(), int("".join("1" if b else "0" for b in bits), 2)


def compute_fingerprints(paths: list[Path]) -> list[tuple[str, int]]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(fingerprints, paths, chunksize=64))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def leaf_key(filename: str) -> str:
    """'<uuid>___GHLB2 Leaf 121.3.JPG' -> 'GHLB2 Leaf 121' (photo index removed)."""
    tag = filename.split("___", 1)[-1].rsplit(".", 1)[0]
    return re.sub(r"\.\d+$", "", tag)


# ---------------------------------------------------------------------------
# PlantVillage
# ---------------------------------------------------------------------------
def prepare_plantvillage(root: Path, seed: int) -> tuple[list[dict], dict]:
    color_dir = root / "raw" / "color"
    rows = []
    for class_dir in sorted(p for p in color_dir.iterdir() if p.is_dir() and p.name.startswith("Tomato___")):
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask = root / "raw" / "segmented" / class_dir.name / f"{image.stem}_final_masked.jpg"
            rows.append({
                "path": image.relative_to(root).as_posix(),
                "mask_path": mask.relative_to(root).as_posix() if mask.exists() else "",
                "label": class_dir.name,
            })
    if not rows:
        raise SystemExit(f"No PlantVillage images under {color_dir}. Run scripts/download_data.py first.")
    print(f"PlantVillage: {len(rows)} images, hashing ...", flush=True)
    prints = compute_fingerprints([root / r["path"] for r in rows])

    uf = UnionFind(len(rows))
    by_key: dict[tuple[str, str], int] = {}
    by_md5: dict[str, int] = {}
    by_dhash: dict[int, int] = {}
    md5_dupes = dhash_dupes = 0
    for i, (row, (md5, dhash)) in enumerate(zip(rows, prints)):
        key = (row["label"], leaf_key(Path(row["path"]).name))
        if key in by_key:
            uf.union(i, by_key[key])
        by_key.setdefault(key, i)
        if md5 in by_md5:
            uf.union(i, by_md5[md5])
            md5_dupes += 1
        by_md5.setdefault(md5, i)
        if dhash in by_dhash:
            uf.union(i, by_dhash[dhash])
            dhash_dupes += 1
        by_dhash.setdefault(dhash, i)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        members[uf.find(i)].append(i)
    conflicting = [g for g in members.values() if len({rows[i]["label"] for i in g}) > 1]

    # Stratified, group-aware assignment. Label-conflicting groups go to train,
    # so evaluation splits never contain ambiguous duplicates.
    rng = random.Random(seed)
    by_class: dict[str, list[list[int]]] = defaultdict(list)
    for group in members.values():
        if group in conflicting:
            for i in group:
                rows[i]["split"] = "train"
            continue
        by_class[rows[group[0]]["label"]].append(group)
    for label, groups in sorted(by_class.items()):
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        targets = {name: round(fraction * total) for name, fraction in SPLIT_FRACTIONS.items()}
        filled = Counter()
        for group in groups:
            split = next((s for s in ("test", "val") if filled[s] < targets[s]), "train")
            filled[split] += len(group)
            for i in group:
                rows[i]["split"] = split
    group_ids = {i: f"g{root_id}" for root_id, group in members.items() for i in group}
    for i, row in enumerate(rows):
        row["group"] = group_ids[i]

    # Verify: no group crosses splits
    splits_per_group = defaultdict(set)
    for row in rows:
        splits_per_group[row["group"]].add(row["split"])
    assert all(len(s) == 1 for s in splits_per_group.values()), "a group crosses splits"

    sizes = Counter(len(g) for g in members.values())
    audit = {
        "images": len(rows),
        "classes": len({r["label"] for r in rows}),
        "images_with_mask": sum(1 for r in rows if r["mask_path"]),
        "groups": len(members),
        "multi_image_groups": sum(1 for g in members.values() if len(g) > 1),
        "images_in_multi_image_groups": sum(len(g) for g in members.values() if len(g) > 1),
        "group_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "exact_duplicate_files": md5_dupes,
        "visual_duplicates_dhash": dhash_dupes,
        "label_conflict_groups": len(conflicting),
        "label_conflict_images": sum(len(g) for g in conflicting),
        "split_counts": {
            label: dict(Counter(r["split"] for r in rows if r["label"] == label))
            for label in sorted({r["label"] for r in rows})
        },
        "totals": dict(Counter(r["split"] for r in rows)),
        "seed": seed,
    }
    return rows, audit


# ---------------------------------------------------------------------------
# PlantDoc
# ---------------------------------------------------------------------------
def prepare_plantdoc(root: Path) -> tuple[list[dict], dict]:
    rows = []
    for split in ("train", "test"):
        for folder, label in PLANTDOC_TO_PLANTVILLAGE.items():
            directory = root / split / folder
            if not directory.is_dir():
                continue
            for image in sorted(directory.iterdir()):
                if image.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({"path": image.relative_to(root).as_posix(), "label": label, "split": split})
    if not rows:
        raise SystemExit(f"No PlantDoc images under {root}. Run scripts/download_data.py first.")

    # Web images: keep only files both PIL (app/evaluation) and TensorFlow (training) can decode,
    # so one odd file cannot crash a training run hours in.
    import tensorflow as tf

    readable, unreadable = [], 0
    for row in rows:
        try:
            with Image.open(root / row["path"]) as img:
                img.verify()
            tf.io.decode_image(tf.io.read_file(str(root / row["path"])), channels=3, expand_animations=False)
            readable.append(row)
        except Exception:
            unreadable += 1
    rows = readable
    prints = compute_fingerprints([root / r["path"] for r in rows])
    train_md5 = {p[0] for r, p in zip(rows, prints) if r["split"] == "train"}
    train_dhash = {p[1] for r, p in zip(rows, prints) if r["split"] == "train"}
    kept, leaked = [], 0
    for row, (md5, dhash) in zip(rows, prints):
        if row["split"] == "test" and (md5 in train_md5 or dhash in train_dhash):
            leaked += 1
            continue
        row["group"] = f"d{dhash:016x}"
        kept.append(row)
    audit = {
        "images": len(kept),
        "unreadable_files_skipped": unreadable,
        "test_images_duplicating_train_removed": leaked,
        "split_counts": {
            label: dict(Counter(r["split"] for r in kept if r["label"] == label))
            for label in sorted({r["label"] for r in kept})
        },
        "totals": dict(Counter(r["split"] for r in kept)),
        "class_mapping": PLANTDOC_TO_PLANTVILLAGE,
    }
    return kept, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    audit = {}
    pv_rows, audit["plantvillage"] = prepare_plantvillage(args.raw_dir / "plantvillage", args.seed)
    write_manifest(args.out_dir / config.PLANTVILLAGE_MANIFEST.name, pv_rows)
    print(json.dumps({k: v for k, v in audit["plantvillage"].items() if k != "split_counts"}, indent=2))

    plantdoc_root = args.raw_dir / "plantdoc"
    if plantdoc_root.exists():
        pd_rows, audit["plantdoc"] = prepare_plantdoc(plantdoc_root)
        write_manifest(args.out_dir / config.PLANTDOC_MANIFEST.name, pd_rows)
        print(json.dumps({k: v for k, v in audit["plantdoc"].items() if k not in ("split_counts", "class_mapping")}, indent=2))

    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote manifests and audit to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
