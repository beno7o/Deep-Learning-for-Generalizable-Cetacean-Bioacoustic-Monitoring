"""Unpack the Watkins full-cuts zip and build a clip index.

Usage:
    python -m src.preprocess.prepare_watkins --zip <watkins_full_cuts.zip> --root "D:/UNI&WORK/Watkins"

Expects in <root>:
    old/metadata_2025.csv   2025 annotations (category, filename, duration_sec, sample_rate)
    old/splits_2025/{train,val,test}.csv   2025 EfficientNet split

Produces:
    raw/<zip species>/<year>/<file>.wav   files exactly as in the zip (never edited)
    raw/class_labels_indices_whales.csv
    old/index.csv   one row per audio file: filename, zip_species, year, path, old_category
                (several joined with "+" if the old dataset had the file under several
                species), n_old_labels, label_status, old_split
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

TOP = "Watkins_Marine_Mammal_Sound_Database/"


def norm(name: str) -> str:
    """Make zip and old species names comparable (Dall'sPorpoise == Dall_sPorpoise)."""
    return name.replace("'", "_")


def extract(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [i for i in z.infolist() if not i.is_dir() and i.filename.startswith(TOP)]
        done = skipped = 0
        for info in members:
            target = raw_dir / info.filename[len(TOP):]
            # Resume support: skip files already fully written.
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            done += 1
            if done % 1000 == 0:
                print(f"extracted {done} (skipped {skipped}) of {len(members)}", flush=True)
        print(f"extracted {done}, already present {skipped}, total {len(members)}", flush=True)


def build_index(root: Path, raw_dir: Path) -> pd.DataFrame:
    rows = []
    for wav in raw_dir.rglob("*.wav"):
        rel = wav.relative_to(raw_dir)
        species, year = rel.parts[0], rel.parts[1]
        rows.append({
            "filename": wav.name,
            "zip_species": species,
            "year": year,
            "path": rel.as_posix(),
            "size_bytes": wav.stat().st_size,
        })
    idx = pd.DataFrame(rows)

    # The old dataset copied some multi-species clips into several species folders,
    # so one filename can have several old labels (and land in several splits).
    # Collapse to one row per filename, joining multiple values with "+".
    join = lambda s: "+".join(sorted(set(s)))
    meta = pd.read_csv(root / "old" / "metadata_2025.csv")
    meta = meta.groupby("filename").agg(
        old_category=("category", join), n_old_labels=("category", "nunique")).reset_index()
    split_rows = []
    for split in ["train", "val", "test"]:
        s = pd.read_csv(root / "old" / "splits_2025" / f"{split}.csv")
        split_rows.append(pd.DataFrame({"filename": s.filename, "old_split": split}))
    splits = pd.concat(split_rows).groupby("filename").old_split.agg(join).reset_index()

    idx = idx.merge(meta, on="filename", how="outer").merge(splits, on="filename", how="left")

    def status(r):
        if pd.isna(r.zip_species):
            return "missing_audio"
        if pd.isna(r.old_category):
            return "not_in_old_dataset"
        old = r.old_category.split("+")
        if r.zip_species in old:
            return "match" if len(old) == 1 else "match_one_of_several"
        if norm(r.zip_species) in {norm(o) for o in old}:
            return "match_name_spelling"
        return "label_differs"

    idx["label_status"] = idx.apply(status, axis=1)
    cols = ["filename", "zip_species", "year", "path", "size_bytes",
            "old_category", "n_old_labels", "label_status", "old_split"]
    return idx[cols].sort_values(["zip_species", "filename"], na_position="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()

    raw_dir = args.root / "raw"
    extract(args.zip, raw_dir)
    idx = build_index(args.root, raw_dir)
    out = args.root / "old" / "index.csv"
    idx.to_csv(out, index=False)
    print(f"wrote {out} ({len(idx)} rows)")
    print(idx.label_status.value_counts().to_string())
    print(idx.groupby("old_split", dropna=False).label_status.value_counts().to_string())


if __name__ == "__main__":
    main()
