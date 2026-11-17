"""Train / val / test dataloaders for one split folder (train.csv, val.csv, test.csv)."""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import SpectrogramCSVDataset

_CACHE: dict = {}


def _dataset(csv, classes, cache_in_memory, **kw):
    """Reuse the RAM cache across runs in the same session (loading from USB is slow)."""
    ds = SpectrogramCSVDataset(csv, classes, cache_in_memory=False, **kw)
    if cache_in_memory:
        key = str(csv)
        if key not in _CACHE:
            _CACHE[key] = [ds._load(p).half() for p, _ in ds.samples]
        ds._cache = _CACHE[key]
    return ds


def get_dataloaders(split_dir: Path, batch_size: int = 16, num_workers: int = 0,
                    pin_memory: bool = True, cache_in_memory: bool = True,
                    train_transform=None, random_offset: bool = False, seed: int = config.SEED):
    """Classes = sorted species in train.csv (identical in val/test by construction).

    train_transform / random_offset apply to the TRAIN set only (augmentation).
    num_workers=0 is the right default here: everything is cached in RAM, and
    worker processes on Windows would each copy the cache.
    """
    split_dir = Path(split_dir)
    classes = sorted(pd.read_csv(split_dir / "train.csv").category.unique())
    train_ds = _dataset(split_dir / "train.csv", classes, cache_in_memory,
                        transform=train_transform, random_offset=random_offset, seed=seed)
    val_ds = _dataset(split_dir / "val.csv", classes, cache_in_memory)
    test_ds = _dataset(split_dir / "test.csv", classes, cache_in_memory)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    g = torch.Generator().manual_seed(seed)  # same shuffle order for the same seed
    return (DataLoader(train_ds, shuffle=True, generator=g, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))
