"""Audio -> normalised mel spectrogram tensors (.norm.pt).

Same idea as the 2025 pipeline (44 kHz, 128 mels, power->dB, first 400 frames),
with two fixes:
  * fixed n_fft = 1024 / hop = 512 (2025 changed n_fft with clip length)
  * only the first ~5.2 s is loaded (the model only sees the first 400 frames
    anyway), so very long clips (up to 24 min) don't slow everything down
Per-sample z-score normalisation is applied before saving (2025 normalize_pt.py).

Output: Spectrograms/<category>/<filename>.norm.pt, shape [1, 128, T] float32
One file per clip; the split CSVs decide which set a clip belongs to.

Usage: python -m src.preprocess.preprocess
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config


def spec_path(category: str, filename: str) -> Path:
    return config.SPEC_DIR / category / f"{filename}.norm.pt"


def compute_spectrogram(audio_path) -> torch.Tensor:
    y, _ = librosa.load(str(audio_path), sr=config.TARGET_SR, mono=True, duration=config.LOAD_SECONDS)
    mel = librosa.feature.melspectrogram(y=y, sr=config.TARGET_SR, n_fft=config.N_FFT,
                                         hop_length=config.HOP_LENGTH, n_mels=config.N_MELS, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    t = torch.from_numpy(mel_db).float().unsqueeze(0)  # [1, n_mels, T]
    return (t - t.mean()) / (t.std() + 1e-6)


def _work(args):
    audio_path, out_path = args
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(compute_spectrogram(audio_path), out_path)
        return None
    except Exception as exc:
        return f"{audio_path}: {exc}"


def generate_all(workers: int = 12, overwrite: bool = False):
    df = pd.read_csv(config.ANNOT_DIR / "audio_annotations.csv")
    jobs = [(config.RAW_DIR / p, spec_path(c, f)) for c, f, p in zip(df.category, df.filename, df.path)]
    todo = [j for j in jobs if overwrite or not j[1].exists()]
    print(f"{len(jobs)} clips, {len(jobs) - len(todo)} already done, {len(todo)} to compute")
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for err in tqdm(ex.map(_work, todo, chunksize=16), total=len(todo), desc="spectrograms", mininterval=30):
            if err:
                errors.append(err)
    print(f"done, {len(errors)} errors")
    for e in errors[:20]:
        print("  ", e)
    return errors


if __name__ == "__main__":
    generate_all()
