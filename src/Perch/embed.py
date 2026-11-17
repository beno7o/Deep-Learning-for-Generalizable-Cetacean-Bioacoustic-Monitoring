"""Perch 2.0 embeddings for every clean Watkins clip (runs in the .venv-perch environment).

Model: official Perch v2 CPU model from Kaggle (google/bird-vocalization-classifier/
tensorFlow2/perch_v2_cpu/1), loaded through Perch-Hoplite (`perch_v2_cpu` preset).
Hoplite does the model's own preprocessing: 5 s windows at 32 kHz, peak-normalised to 0.25.

Per clip:
  * audio resampled to 32 kHz mono (librosa), at most the first 60 s (12 windows)
  * cut into 5 s windows (hop 5 s); the last window is zero-padded
  * emb_first = embedding of the first window  -> comparable with EfficientNet,
                which only sees the first ~4.65 s
  * emb_mean  = mean over all windows          -> the Perch 2.0 paper's protocol

Output (resumable, one file per chunk of 1000 clips):
  <DATA_ROOT>/Embeddings/perch_v2/chunk_XXX.npz
and, when complete, embeddings.npz with filename, category, n_windows, emb_first, emb_mean.

Usage: .venv-perch\\Scripts\\python -m src.Perch.embed
"""
import time
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import pandas as pd

from src import config

SR = 32000
WIN = 5 * SR
MAX_WINDOWS = 12
CHUNK = 1000
BATCH = 32
OUT_DIR = config.DATA_ROOT / "Embeddings" / "perch_v2"


def load_windows(path) -> np.ndarray:
    """[n_windows, 160000] float32, zero-padded last window."""
    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=MAX_WINDOWS * 5.0)
    n = max(1, int(np.ceil(len(y) / WIN)))
    y = np.pad(y, (0, n * WIN - len(y)))
    return y.reshape(n, WIN).astype(np.float32)


def load_model():
    from perch_hoplite.zoo import model_configs
    return model_configs.load_model_by_name("perch_v2_cpu")


def embed_chunk(model, df: pd.DataFrame) -> dict:
    paths = [config.RAW_DIR / p for p in df.path]
    with ThreadPoolExecutor(max_workers=6) as ex:  # decoding/resampling overlaps with inference
        windows = list(ex.map(load_windows, paths))
    counts = np.array([len(w) for w in windows])
    allw = np.concatenate(windows)
    embs = []
    for i in range(0, len(allw), BATCH):
        out = model.batch_embed(allw[i:i + BATCH])           # [B, 1, 1, 1536]
        embs.append(np.asarray(out.embeddings).reshape(-1, 1536))
    embs = np.concatenate(embs)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    first = embs[starts]
    mean = np.stack([embs[s:s + c].mean(0) for s, c in zip(starts, counts)])
    return {"filename": df.filename.values, "category": df.category.values,
            "n_windows": counts, "emb_first": first.astype(np.float32), "emb_mean": mean.astype(np.float32)}


def embed_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.ANNOT_DIR / "audio_annotations.csv")
    n_chunks = int(np.ceil(len(df) / CHUNK))
    model = None
    t0 = time.time()
    for k in range(n_chunks):
        out = OUT_DIR / f"chunk_{k:03d}.npz"
        if out.exists():
            continue
        if model is None:
            model = load_model()
        t = time.time()
        res = embed_chunk(model, df.iloc[k * CHUNK:(k + 1) * CHUNK])
        np.savez(out, **res)
        print(f"chunk {k + 1}/{n_chunks}: {len(res['filename'])} clips, {res['n_windows'].sum()} windows, "
              f"{time.time() - t:.0f}s (total {(time.time() - t0) / 60:.1f} min)", flush=True)

    parts = [np.load(OUT_DIR / f"chunk_{k:03d}.npz", allow_pickle=True) for k in range(n_chunks)]
    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    assert len(merged["filename"]) == len(df), "missing embeddings"
    np.savez(OUT_DIR / "embeddings.npz", **merged)
    print("saved", OUT_DIR / "embeddings.npz", merged["emb_first"].shape)
    return merged


if __name__ == "__main__":
    embed_all()
