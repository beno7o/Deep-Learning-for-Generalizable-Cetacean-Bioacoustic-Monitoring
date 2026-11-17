"""Train / val / test dataloaders for one split folder (train.csv, val.csv, test.csv)."""
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import SpectrogramCSVDataset


def get_dataloaders(split_dir: Path, batch_size: int = 16, num_workers: int = 0,
                    pin_memory: bool = True, cache_in_memory: bool = True, train_transform=None):
    """Classes = sorted species in train.csv (identical in val/test by construction).

    num_workers=0 is the right default here: everything is cached in RAM, and
    worker processes on Windows would each copy the cache.
    """
    split_dir = Path(split_dir)
    classes = sorted(pd.read_csv(split_dir / "train.csv").category.unique())
    train_ds = SpectrogramCSVDataset(split_dir / "train.csv", classes, transform=train_transform,
                                     cache_in_memory=cache_in_memory)
    val_ds = SpectrogramCSVDataset(split_dir / "val.csv", classes, cache_in_memory=cache_in_memory)
    test_ds = SpectrogramCSVDataset(split_dir / "test.csv", classes, cache_in_memory=cache_in_memory)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))
