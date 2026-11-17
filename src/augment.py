"""Training-time augmentation for normalised mel spectrograms [1, n_mels, T].

Aimed at recording-condition differences (DL Lab 5: "simulate the conditions you
will meet"), and only transforms that keep the label's meaning:

* random low-pass  - blank all mel bins above a random cutoff (set to the clip's
                     minimum, like a lower original sample rate). Makes the
                     "empty band = sample rate" shortcut useless.
* SpecAugment      - frequency and time masks (Park et al., 2019; also used in REPSOL).
* gaussian noise   - small additive noise on the normalised spectrogram.

Not used: frequency flips/shifts (they change what the call sounds like). Random
time offset is done in the dataset (random_offset=True) because it needs the
unpadded width.
"""
import torch


class SpectrogramAugment:
    def __init__(self, p_lowpass: float = 0.5, min_cutoff_frac: float = 0.25,
                 freq_masks: int = 2, max_freq_width: int = 12,
                 time_masks: int = 2, max_time_width: int = 40,
                 p_noise: float = 0.5, noise_std: float = 0.05, seed: int | None = None):
        self.p_lowpass, self.min_cutoff_frac = p_lowpass, min_cutoff_frac
        self.freq_masks, self.max_freq_width = freq_masks, max_freq_width
        self.time_masks, self.max_time_width = time_masks, max_time_width
        self.p_noise, self.noise_std = p_noise, noise_std
        self.g = torch.Generator()
        if seed is not None:
            self.g.manual_seed(seed)

    def _rand(self):
        return torch.rand(1, generator=self.g).item()

    def _randint(self, lo: int, hi: int) -> int:  # inclusive lo, exclusive hi
        return int(torch.randint(lo, max(hi, lo + 1), (1,), generator=self.g).item())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        _, n_mels, n_t = x.shape
        fill = x.min()

        # low mel index = low frequency (librosa), so a low-pass blanks the top rows
        if self._rand() < self.p_lowpass:
            cutoff = self._randint(int(self.min_cutoff_frac * n_mels), n_mels)
            x[:, cutoff:, :] = fill

        for _ in range(self.freq_masks):
            w = self._randint(0, self.max_freq_width + 1)
            f0 = self._randint(0, n_mels - w)
            x[:, f0:f0 + w, :] = fill

        for _ in range(self.time_masks):
            w = self._randint(0, self.max_time_width + 1)
            t0 = self._randint(0, n_t - w)
            x[:, :, t0:t0 + w] = fill

        if self._rand() < self.p_noise:
            x = x + self.noise_std * torch.randn(x.shape, generator=self.g)
        return x
