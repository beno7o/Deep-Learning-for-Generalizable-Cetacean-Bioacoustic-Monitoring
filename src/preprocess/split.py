"""Train / val / test splits.

1. Main split (44 species, clip-level): stratified 70/15/15, seed 42.
   Same method as 2025, but on the cleaned data. Clips from the same recording
   can end up in train AND test, so this measures "recognise species in
   recordings the model has partly seen".

2. Grouped split (34 species, by recording): every recording (first 5 chars of
   the filename) is entirely in ONE of train/val/test, across all species (some
   tapes contain several species). Only species with
   >= 3 recordings can be in all three sets, so 10 species are left out.
   This measures "recognise species in recordings the model has never heard".

Usage: python -m src.preprocess.split
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

TARGET = np.array([0.70, 0.15, 0.15])  # train, val, test share of clips
NAMES = ["train", "val", "test"]


def split_clip_level(df: pd.DataFrame, seed: int = config.SEED):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["category"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["category"], random_state=seed)
    return train_df, val_df, test_df


def _best_assignment(rec_sizes: np.ndarray, fixed: np.ndarray, rng: np.random.Generator,
                     n_random: int = 4000):
    """Assign each recording to train/val/test (0/1/2) so every set gets >= 1 recording
    and the clip shares are as close as possible to 70/15/15.
    fixed[i] >= 0 means recording i was already assigned (it also holds another species)."""
    total = rec_sizes.sum()
    free = np.where(fixed < 0)[0]

    def score(assign):
        shares = np.bincount(assign, weights=rec_sizes, minlength=3) / total
        return float(((shares - TARGET) ** 2).sum())

    if len(free) <= 9:  # small: try every assignment of the free recordings (3^9 = 19683)
        free_parts = (np.array(a) for a in itertools.product(range(3), repeat=len(free)))
    else:               # large: random search
        free_parts = (rng.choice(3, size=len(free), p=TARGET) for _ in range(n_random))

    best, best_key = None, None
    for part in free_parts:
        a = fixed.copy()
        a[free] = part
        # prefer assignments that put this species in all 3 sets, then the closest to 70/15/15
        key = (len(set(a.tolist())) < 3, score(a))
        if best_key is None or key < best_key:
            best, best_key = a, key
    return best


def split_by_recording(df: pd.DataFrame, min_recordings: int = config.MIN_RECORDINGS_GROUPED,
                       seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    rec_counts = df.groupby("category").recording.nunique()
    keep_species = rec_counts[rec_counts >= min_recordings].index
    dropped = rec_counts[rec_counts < min_recordings]
    sub = df[df.category.isin(keep_species)]

    # Some Watkins tapes hold several species, so a recording is assigned ONCE for all
    # species. Species with the fewest recordings choose first (they have the least freedom).
    rec_split: dict[str, int] = {}
    order = sub.groupby("category").recording.nunique().sort_values(kind="stable").index
    for species in order:
        sizes = sub[sub.category == species].groupby("recording").size()
        fixed = np.array([rec_split.get(r, -1) for r in sizes.index])
        a = _best_assignment(sizes.values.astype(float), fixed, rng)
        for rec, s in zip(sizes.index, a):
            rec_split[rec] = int(s)
    split_col = [NAMES[rec_split[r]] for r in sub.recording]
    sub = sub.assign(split=split_col)
    parts = tuple(sub[sub.split == n].drop(columns="split") for n in NAMES)
    return parts, dropped


def check(parts, grouped: bool):
    tr, va, te = parts
    keys = [set(p.filename) for p in parts]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]), "clip in two sets"
    if grouped:  # a recording (tape) may hold several species: check across species
        recs = [set(p.recording) for p in parts]
        assert not (recs[0] & recs[1] or recs[0] & recs[2] or recs[1] & recs[2]), "recording in two sets"
    for p, n in zip(parts, NAMES):
        assert p.category.nunique() == tr.category.nunique(), f"{n} is missing species"


def summary(parts) -> pd.DataFrame:
    out = pd.concat([p.groupby("category").agg(**{f"{n}_clips": ("filename", "size"),
                                                  f"{n}_recordings": ("recording", "nunique")})
                     for p, n in zip(parts, NAMES)], axis=1).fillna(0).astype(int)
    return out


def write(parts, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, n in zip(parts, NAMES):
        p.sort_values(["category", "filename"]).to_csv(out_dir / f"{n}.csv", index=False)
    summary(parts).to_csv(out_dir / "split_summary.csv")


def make_all_splits():
    df = pd.read_csv(config.ANNOT_DIR / "audio_annotations.csv")

    main = split_clip_level(df)
    check(main, grouped=False)
    write(main, config.MAIN_SPLIT_DIR)
    print("main (44 species, clip-level):", [len(p) for p in main])

    grouped, dropped = split_by_recording(df)
    check(grouped, grouped=True)
    write(grouped, config.GROUPED_SPLIT_DIR)
    dropped.rename("recordings").to_csv(config.GROUPED_SPLIT_DIR / "species_left_out.csv")
    n = sum(len(p) for p in grouped)
    print(f"grouped ({grouped[0].category.nunique()} species, by recording):",
          [len(p) for p in grouped], "shares:", [round(len(p) / n, 3) for p in grouped])
    print("left out of grouped split:", dropped.to_dict())
    return main, grouped, dropped


if __name__ == "__main__":
    make_all_splits()
