"""Load normalised spectrogram .pt files listed in a split CSV.

Difference from REPSOL/2025 (folder-driven SpectrogramPTDataset): the split CSV
decides which clips are in a set, so the same Spectrograms/ folder serves both
the main (44 species) and the grouped (34 species) split without copying files.

Pads/crops every spectrogram to a fixed width (400 frames, like 2025).
Optionally caches everything in RAM as float16 (the external USB drive is slow).
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src import config


class SpectrogramCSVDataset(Dataset):
    def __init__(self, csv_path, classes, spec_dir=config.SPEC_DIR, transform=None,
                 target_width: int | None = config.TARGET_WIDTH, cache_in_memory: bool = True):
        self.df = pd.read_csv(csv_path)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.target_width = target_width
        self.samples = [(spec_dir / c / f"{f}.norm.pt", self.class_to_idx[c])
                        for c, f in zip(self.df.category, self.df.filename)]
        self.filenames = list(self.df.filename)
        self._cache = None
        if cache_in_memory:
            self._cache = [self._load(p).half() for p, _ in
                           tqdm(self.samples, desc=f"caching {csv_path.name}", mininterval=30)]

    def _load(self, path) -> torch.Tensor:
        t = torch.load(path, weights_only=True)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if self.target_width is not None:
            w = t.shape[-1]
            if w < self.target_width:
                t = torch.nn.functional.pad(t, (0, self.target_width - w))
            else:
                t = t[..., : self.target_width]
        return t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        t = self._cache[idx] if self._cache is not None else self._load(path)
        t = t.float()
        if self.transform:
            t = self.transform(t)
        return t, label
