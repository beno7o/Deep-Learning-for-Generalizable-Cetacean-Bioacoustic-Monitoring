# Deep Learning for Generalizable Cetacean Bioacoustic Monitoring

Önálló labor project, BME MSc Artificial Intelligence. Supervisor: Dr. Mohammed Al-Radhi.

The project investigates bioacoustic foundation models (Perch 2.0, later AVES) for classifying whale and dolphin
vocalizations, and how well they generalize to recordings from different datasets, locations, hydrophones and
recording conditions. Step 1 (this repo so far) rebuilds a clean **EfficientNet baseline on the Watkins Marine
Mammal Sound Database**. Perch 2.0 is compared against it on exactly the same splits.

A dated log of everything done is in [`notes.md`](notes.md).

## Status

| Step | State |
| --- | --- |
| Watkins audio recovered, audited against the 2025 dataset | done |
| Cleaning + two splits (44 species clip-level / 34 species by recording) | done |
| Dataset exploration | done |
| EfficientNet_01 (main split) / EfficientNet_02 (grouped split) | done, see Results |
| Perch 2.0 embeddings on the same splits (WSL2) | next |
| DCLDE killer whale dataset (generalization) | later |

## Folder structure (REPSOL layout)

```
onlab-cetacean/
├── README.md                     # this file: current state, decisions, results
├── notes.md                      # dated log
├── Preprocess.ipynb              # ⭐ clean → splits → spectrograms (run first)
├── Dataset_exploration.ipynb     # EDA of the cleaned dataset (run after Preprocess)
├── Models/                       # one notebook per training run
│   ├── EfficientNet_01.ipynb     # main split: 44 species, clip-level
│   ├── EfficientNet_02.ipynb     # grouped split: 34 species, by recording
│   └── Evaluation.ipynb          # ⭐ compares all runs (+ the 2025 result)
├── Models_output/                # checkpoints (*_best_NN.pth, auto-numbered) + training histories
├── outputs/
│   ├── exploration/              # dataset exploration figures
│   └── evaluation/               # metrics, reports, confusion matrices, per-class F1, predictions
└── src/
    ├── config.py                 # all paths + settings (data root, spectrogram params, seed)
    ├── dataset.py                # SpectrogramCSVDataset (split CSV → .norm.pt, RAM cache)
    ├── dataloaders.py            # train/val/test loaders for one split folder
    ├── evaluate.py               # predictions, metrics, saved reports
    ├── plot_analysis.py          # learning curves, confusion matrix, per-class F1
    ├── EfficientNet/
    │   ├── model.py              # EfficientNet-B0 for 1-channel spectrograms
    │   └── train.py              # Trainer (2025 recipe)
    └── preprocess/
        ├── prepare_watkins.py    # unzip download → raw/, build old/index.csv (2025 vs download)
        ├── clean.py              # → Annotations/audio_annotations.csv + excluded.csv
        ├── split.py              # → main + grouped splits
        └── preprocess.py         # → Spectrograms/<category>/<file>.wav.norm.pt
```

Data lives on the external drive (not in git), REPSOL's `Data/` equivalent:

```
D:\UNI&WORK\Watkins\              # config.DATA_ROOT (override with env var WATKINS_ROOT)
├── raw\<species>\<year>\*.wav    # original download, never edited (15,248 files, 9.86 GB)
├── Annotations\
│   ├── audio_annotations.csv     # clean clips, one row each
│   ├── excluded.csv              # every dropped clip + reason
│   ├── train.csv val.csv test.csv split_summary.csv    # main split (44 species)
│   └── grouped_34\               # grouped split (34 species) + species_left_out.csv
├── Spectrograms\<category>\*.norm.pt   # one per clean clip, shared by both splits
└── old\                          # 2025 metadata + splits, index.csv (reference only)
```

## Pipeline

1. `Preprocess.ipynb`: cleaning → splits → spectrograms
2. `Dataset_exploration.ipynb`
3. `Models/EfficientNet_01.ipynb`, `Models/EfficientNet_02.ipynb`
4. `Models/Evaluation.ipynb`

Command-line equivalents: `python -m src.preprocess.clean`, `python -m src.preprocess.split`,
`python -m src.preprocess.preprocess`, `python -m src.EfficientNet.train --split main --run EfficientNet_01`.

## Data

**Source.** Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution, "all cuts".
The official WHOI site has been offline (maintenance page) for about a year, so the audio comes from an unofficial
archive.org mirror (`watkins-marine-mammal-sound-database-full-cuts`, 6.7 GB zip, 15,248 WAVs, 54 species folders).
WHOI's terms allow personal/academic use but not reposting on third-party sites; this copy is used for academic
work only and is not redistributed. Credit: *Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution.*

**Problems found in the 2025 dataset** (Bens-Internship-2025: 44 species, 15,407 rows, 94.7 % test accuracy):
- 299 clips stored under **two species** (two species audible), e.g. long-finned pilot whale + sperm whale.
- **Leakage:** 64 identical files in both train and test; 317 of 321 test recordings also in train.
- 519 clips labelled *White-beaked dolphin* in 2025 are *White-sided dolphin* in the download (cannot be verified
  while WHOI is offline). The 2025 model's main errors were on White-sided dolphin, possibly label noise.

**Cleaning rules** (`src/preprocess/clean.py`; first matching reason is recorded in `excluded.csv`):

| Rule | Action |
| --- | --- |
| 10 rare species not in the 2025 set (1–7 clips each) | drop (`species_not_in_44`) |
| Spelling variants (`Dall'sPorpoise` / `Dall_sPorpoise`) | one name per species (2025 style) |
| Clip had two species in 2025 | drop (`multi_species`) |
| 2025 label ≠ download label | drop (`label_conflict`) |
| Unreadable / empty file | drop (`unreadable`) |
| Shorter than 0.2 s | drop (`too_short`) |
| Byte-identical file under another name | keep one (`duplicate_audio`); drop all if labels differ (`duplicate_conflicting`) |
| Sample rate < 4 kHz or > 96 kHz | keep, flag in `sr_flag` |

A *recording* = the first 5 characters of the Watkins filename (tape/session), e.g. `61025` in `6102500A.wav`.

## Splits and why there are two

| Split | Species | Method | What it measures |
| --- | --- | --- | --- |
| **main** (`Annotations/`) | 44 | clip-level, stratified 70/15/15, seed 42 (same method as 2025) | recognising species in recordings the model has **partly seen** |
| **grouped** (`Annotations/grouped_34/`) | 34 | by recording: each recording entirely in train, val *or* test (across all species, since some tapes hold several species), ≈70/15/15 per species | recognising species in recordings the model has **never heard** |

**Why the main split keeps all 44 species:** comparability with the 2025 work and with most Watkins papers, which
also split by clip, and every species keeps enough clips.

**Why a second, 34-species split:** clips from one recording share far more than the species: often the same
animals, hydrophone, boat noise, tape hiss and sample rate. With a clip-level split the model can score well by
recognising *the recording* instead of *the species*. The project's research question is generalization to new
recordings, so we also need a split where no recording is shared. A species needs at least 3 recordings to be in
train, val and test without sharing one; **10 species have only 1–2 recordings** (Tucuxi, Heaviside's, Dusky,
Dall's porpoise, Leopard seal, Boutu, Southern right whale, Long-beaked common dolphin, Rough-toothed dolphin,
Melon-headed whale), so they are left out of this split only. **The gap between the two results shows how much
the model relies on recognising recordings.**

## Model: EfficientNet (2025 recipe, unchanged)

- Input: mel spectrogram, 44 kHz mono, n_fft 1024, hop 512, 128 mels, power → dB, per-clip z-score,
  first 400 frames (≈4.65 s; shorter clips zero-padded). *Change vs 2025:* fixed n_fft (2025 varied it with clip length).
- Model: ImageNet EfficientNet-B0, 1 → 3 channels, new 44/34-class head, full fine-tuning.
- Training: batch 16, AdamW lr 1e-3 wd 1e-4, class-weighted cross-entropy, ReduceLROnPlateau (×0.5, patience 2),
  grad clip 1.0, max 10 epochs, early stopping patience 3, best = lowest val loss. Seed 42, mixed precision on GPU.

## Results (2026-11-17)

**Cleaning:** 15,248 downloaded files → **14,098 clean clips, 44 species, 378 recordings**; 1,151 excluded
(label_conflict 535, multi_species 299, too_short 272, species_not_in_44 31, duplicate_audio 13, no_audio 1).
Sample-rate flags: 670 low (< 4 kHz), 1,874 high (> 96 kHz).

**Splits:** main 9,868 / 2,115 / 2,115 (297 recordings shared between train and test).
Grouped 9,502 / 2,102 / 2,057 (≈70/15/15, **0 shared recordings**, all 34 species in every set).

| Run | Split | Classes | Test clips | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- |
| EfficientNet 2025 (old, for reference) | 2025 clip-level, leaky, uncleaned | 44 | 2,312 | 0.947 | 0.91 | 0.95 |
| **EfficientNet_01** | main: clip-level | 44 | 2,115 | **0.971** | **0.944** | 0.970 |
| **EfficientNet_02** | grouped: by recording | 34 | 2,057 | **0.606** | **0.436** | 0.610 |

On the 34 species shared by both splits, mean per-class F1 drops from **0.95 (seen recordings) to 0.44 (unseen
recordings)**. 7 species get F1 = 0 on unseen recordings (e.g. Atlantic spotted dolphin, Gray whale, Narwhal,
West Indian manatee); large, well-recorded species stay high (Fin whale 0.95, Bowhead 0.92, Humpback 0.86).

**Interpretation**
- With clean data and the same clip-level split, EfficientNet does *better* than 2025 (0.971 vs 0.947), so the
  cleaning removed noise rather than an advantage.
- **Most of that accuracy depends on having heard the recording before.** Split by recording, accuracy falls to
  0.61. EfficientNet_02 reaches 96 % train accuracy but val loss rises from epoch 1: it memorises recordings.
  This is the baseline that Perch 2.0 must beat, and the core of the generalization question.
- **Likely shortcut, the sample rate.** Everything is resampled to 44 kHz, so recordings made at a lower rate show a
  black (empty) band at the top of the spectrogram whose height *is* the original sample rate (see
  `outputs/exploration/example_spectrograms.png`). Sample rate is tied to recording equipment, hence to
  recording and often species. Possible follow-up: restrict all clips to a common frequency band.
- Caveat: grouped val/test sets contain few recordings per species (often 1), so per-species results are noisy;
  val accuracy (≈39–51 %) and test accuracy (61 %) differ noticeably.
- EfficientNet_01 was still improving at epoch 10 (max epochs of the 2025 recipe).

**Speed:** ~36 s per epoch on the RTX 5050 (first epoch ~2 min incl. warm-up) vs ~17 min per epoch for the 2025
run on a GTX 1050. Whole pipeline (cleaning → spectrograms → both trainings) ≈ 35 min.

Figures: `outputs/evaluation/` (learning curves, confusion matrices, per-class F1, `all_runs_comparison.png`,
`f1_seen_vs_unseen.png`) and `outputs/exploration/`.

## Environment

Windows 11 (Smart App Control off), Python 3.12 venv in `.venv/`, PyTorch 2.11 + CUDA 12.8 on an RTX 5050 Laptop GPU.

```bash
python -m venv .venv
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install pandas scikit-learn soundfile librosa matplotlib seaborn tqdm jupyter ipykernel
.venv\Scripts\python -m ipykernel install --user --name onlab-cetacean
```

Perch 2.0 (TensorFlow) will run in WSL2; set `WATKINS_ROOT=/mnt/d/UNI&WORK/Watkins` there.
