"""Build the clean Watkins annotation file.

Input : old/index.csv (one row per audio file, from prepare_watkins.py) + raw wavs
Output: Annotations/audio_annotations.csv  clean clips (one row each)
        Annotations/excluded.csv           every dropped clip with its reason

Rules (first matching reason wins):
    no_audio               old 2025 clip with no audio file in the download
    species_not_in_44      one of the 10 rare species not in the 2025 dataset
    multi_species          2025 data had the file under 2 species (2 species audible)
    label_conflict         2025 label differs from the download (519 White-beaked vs White-sided etc.)
    unreadable             soundfile cannot read it or it has 0 samples
    too_short              shorter than config.MIN_DURATION_S
    duplicate_audio        byte-identical copy of a clip that is kept
    duplicate_conflicting  byte-identical files with different species (all copies dropped)
Flags (kept): sr_flag = low (< LOW_SR_HZ) / high (> HIGH_SR_HZ) / ok

Usage: python -m src.preprocess.clean
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import config


def norm_name(name: str) -> str:
    """One spelling per species: Dall'sPorpoise -> Dall_sPorpoise (2025 style)."""
    return name.replace("'", "_")


def audio_info(path):
    """Return (sample_rate, frames, channels, sha1) or None if unreadable."""
    try:
        info = sf.info(str(path))
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return info.samplerate, info.frames, info.channels, h.hexdigest()
    except Exception:
        return None


def build_clean_annotations(write: bool = True):
    idx = pd.read_csv(config.OLD_DIR / "index.csv")
    old_classes = set(
        pd.read_csv(config.OLD_DIR / "metadata_2025.csv").category.map(norm_name))

    excluded, candidates = [], []
    for r in idx.itertuples():
        base = {"filename": r.filename, "zip_species": r.zip_species,
                "old_category": r.old_category, "path": r.path}
        if r.label_status == "missing_audio":
            excluded.append({**base, "reason": "no_audio"})
        elif r.label_status == "not_in_old_dataset" and norm_name(r.zip_species) not in old_classes:
            excluded.append({**base, "reason": "species_not_in_44"})
        elif r.label_status == "match_one_of_several":
            excluded.append({**base, "reason": "multi_species", "details": r.old_category})
        elif r.label_status == "label_differs":
            excluded.append({**base, "reason": "label_conflict",
                             "details": f"2025: {r.old_category} / download: {r.zip_species}"})
        else:
            candidates.append(r)

    # Read header + hash of every remaining file (I/O bound -> threads).
    paths = [config.RAW_DIR / r.path for r in candidates]
    with ThreadPoolExecutor(max_workers=8) as ex:
        infos = list(tqdm(ex.map(audio_info, paths), total=len(paths), desc="reading audio", mininterval=30))

    rows = []
    for r, info in zip(candidates, infos):
        base = {"filename": r.filename, "zip_species": r.zip_species,
                "old_category": r.old_category, "path": r.path}
        if info is None or info[1] == 0:
            excluded.append({**base, "reason": "unreadable"})
            continue
        sr, frames, ch, sha1 = info
        dur = frames / sr
        if dur < config.MIN_DURATION_S:
            excluded.append({**base, "reason": "too_short", "details": f"{dur:.3f} s"})
            continue
        rows.append({
            "category": norm_name(r.zip_species),
            "filename": r.filename,
            "path": r.path,
            "recording": r.filename[:5],
            "year": r.year,
            "duration_sec": dur,
            "sample_rate": sr,
            "channels": ch,
            "sha1": sha1,
        })
    df = pd.DataFrame(rows).sort_values(["category", "filename"]).reset_index(drop=True)

    # Exact duplicates (same bytes, different filename).
    keep_mask = pd.Series(True, index=df.index)
    for sha, grp in df.groupby("sha1"):
        if len(grp) < 2:
            continue
        if grp.category.nunique() > 1:
            for i in grp.index:
                keep_mask[i] = False
                excluded.append({"filename": df.at[i, "filename"], "zip_species": df.at[i, "category"],
                                 "path": df.at[i, "path"], "reason": "duplicate_conflicting",
                                 "details": "+".join(sorted(grp.filename))})
        else:
            for i in grp.index[1:]:
                keep_mask[i] = False
                excluded.append({"filename": df.at[i, "filename"], "zip_species": df.at[i, "category"],
                                 "path": df.at[i, "path"], "reason": "duplicate_audio",
                                 "details": f"same as {grp.filename.iloc[0]}"})
    df = df[keep_mask].reset_index(drop=True)

    df["sr_flag"] = "ok"
    df.loc[df.sample_rate < config.LOW_SR_HZ, "sr_flag"] = "low"
    df.loc[df.sample_rate > config.HIGH_SR_HZ, "sr_flag"] = "high"

    exc = pd.DataFrame(excluded)
    if write:
        config.ANNOT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.ANNOT_DIR / "audio_annotations.csv", index=False)
        exc.to_csv(config.ANNOT_DIR / "excluded.csv", index=False)

    print(f"clean clips: {len(df)} | species: {df.category.nunique()} | recordings: {df.recording.nunique()}")
    print("excluded by reason:")
    print(exc.reason.value_counts().to_string())
    print("sample-rate flags:", df.sr_flag.value_counts().to_dict())
    return df, exc


if __name__ == "__main__":
    build_clean_annotations()
