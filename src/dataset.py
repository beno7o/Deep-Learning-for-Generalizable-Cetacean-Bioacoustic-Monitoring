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
                 target_width: int | None = config.TARGET_WIDTH, cache_in_memory: bool = True,
                 random_offset: bool = False, seed: int = config.SEED):
        """random_offset (training only): instead of always the first 400 frames, take a
        random 400-frame window from longer clips, and place shorter clips at a random
        position in the zero-padded window."""
        self.df = pd.read_csv(csv_path)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.target_width = target_width
        self.random_offset = random_offset
        self.g = torch.Generator().manual_seed(seed)
        self.samples = [(spec_dir / c / f"{f}.norm.pt", self.class_to_idx[c])
                        for c, f in zip(self.df.category, self.df.filename)]
        self.filenames = list(self.df.filename)
        self._cache = None
        if cache_in_memory:
            self._cache = [self._load(p).half() for p, _ in
                           tqdm(self.samples, desc=f"caching {csv_path.name}", mininterval=30)]

    def _load(self, path) -> torch.Tensor:
        """Unpadded spectrogram [1, n_mels, T] (T may differ per clip)."""
        t = torch.load(path, weights_only=True)
        return t.unsqueeze(0) if t.dim() == 2 else t

    def _fix_width(self, t: torch.Tensor) -> torch.Tensor:
        if self.target_width is None:
            return t
        w, tw = t.shape[-1], self.target_width
        if w >= tw:
            start = int(torch.randint(0, w - tw + 1, (1,), generator=self.g)) if self.random_offset else 0
            return t[..., start:start + tw]
        left = int(torch.randint(0, tw - w + 1, (1,), generator=self.g)) if self.random_offset else 0
        return torch.nn.functional.pad(t, (left, tw - w - left))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        t = self._cache[idx] if self._cache is not None else self._load(path)
        t = self._fix_width(t.float())
        if self.transform:
            t = self.transform(t)
        return t, label
