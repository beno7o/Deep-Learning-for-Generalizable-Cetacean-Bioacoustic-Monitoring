# Deep Learning for Generalizable Cetacean Bioacoustic Monitoring

Önálló labor project, BME MSc Artificial Intelligence. Supervisor: Dr. Mohammed Al-Radhi.

The project investigates bioacoustic foundation models (Perch 2.0, later AVES) for classifying whale and dolphin
vocalizations, and how well they generalize to recordings from different datasets, locations, hydrophones and
recording conditions. So far: a clean **EfficientNet baseline** and **Perch 2.0 embeddings + linear probe** on the
**Watkins Marine Mammal Sound Database**, compared on exactly the same splits.

A dated log of everything done is in [`notes.md`](notes.md).

## Headline result

| Test set | EfficientNet-B0 (fine-tuned spectrogram CNN) | Perch 2.0 (frozen) + logistic regression |
| --- | --- | --- |
| **main**: 44 species, recordings partly seen in training | **acc 0.971 / macro-F1 0.944** | acc 0.941 / macro-F1 0.924 |
| **grouped**: 34 species, **unseen recordings** | acc 0.658 / macro-F1 0.484 ± 0.02 (best tuned, 3 seeds) | **acc 0.722–0.724 / macro-F1 0.543–0.568** |

When test recordings have been partly heard during training, the fine-tuned EfficientNet is slightly better.
**On recordings never heard before, frozen Perch 2.0 embeddings with a simple linear classifier generalize
clearly better**: +6 points accuracy, +0.06–0.08 macro-F1 over the best tuned EfficientNet. With only
**32 clips per species**, Perch (macro-F1 0.51) already beats EfficientNet trained on all ~9,500 clips.

## Status

| Step | State |
| --- | --- |
| Watkins audio recovered, audited against the 2025 dataset | done |
| Cleaning + two splits (44 species clip-level / 34 species by recording) | done |
| Dataset exploration | done |
| EfficientNet: 2025 recipe on both splits + tuning on the grouped split (02b–06) | done |
| Perch 2.0 embeddings, linear probe, few-shot, PCA/UMAP, HDBSCAN | done |
| Next ideas: see "Open questions / next steps" | |
| DCLDE killer whale dataset (cross-dataset generalization) | later |

## Folder structure (REPSOL layout)

```
onlab-cetacean/
├── README.md                     # this file: current state, decisions, results
├── notes.md                      # dated log
├── Preprocess.ipynb              # ⭐ clean → splits → spectrograms (run first)
├── Dataset_exploration.ipynb     # EDA of the cleaned dataset
├── Perch_embeddings.ipynb        # Perch 2.0 embeddings of all clean clips (Perch kernel)
├── Models/                       # one notebook per training run
│   ├── EfficientNet_01.ipynb     # main split, 2025 recipe
│   ├── EfficientNet_02.ipynb     # grouped split, 2025 recipe (1 seed)
│   ├── EfficientNet_02b.ipynb    # grouped, 2025 recipe, 3 seeds (reference for tuning)
│   ├── EfficientNet_03.ipynb     # grouped, + lr 1e-4 / weight decay 1e-2
│   ├── EfficientNet_04.ipynb     # grouped, 03 + augmentation
│   ├── EfficientNet_05.ipynb     # grouped, 03 + checkpoint selection on val macro-F1  ⭐ best EfficientNet
│   ├── EfficientNet_06.ipynb     # grouped, 04 + checkpoint selection on val macro-F1
│   ├── Perch2_01.ipynb           # Perch 2.0 linear probe, few-shot, PCA/UMAP, HDBSCAN (Perch kernel)
│   └── Evaluation.ipynb          # ⭐ final comparison of all runs
├── Models_output/                # checkpoints (*_best_NN.pth, auto-numbered) + training histories
├── outputs/
│   ├── exploration/              # dataset exploration figures
│   ├── evaluation/               # metrics, reports, confusion matrices, per-class F1, predictions,
│   │                             #   experiments.csv (every seed), final_comparison.csv/.png
│   └── perch/                    # Perch figures (PCA/UMAP, few-shot, Perch vs EfficientNet)
└── src/
    ├── config.py                 # all paths + settings (data root, spectrogram params, seed)
    ├── dataset.py                # SpectrogramCSVDataset (split CSV → .norm.pt, RAM cache, random offset)
    ├── dataloaders.py            # train/val/test loaders for one split folder
    ├── augment.py                # random low-pass, SpecAugment, noise
    ├── evaluate.py               # predictions, metrics, saved reports
    ├── plot_analysis.py          # learning curves, confusion matrix, per-class F1
    ├── EfficientNet/
    │   ├── model.py              # EfficientNet-B0 for 1-channel spectrograms
    │   ├── train.py              # Trainer (2025 recipe by default; every change is a parameter)
    │   └── experiment.py         # run one configuration with seeds 42/43/44 → experiments.csv
    ├── Perch/
    │   ├── embed.py              # Perch 2.0 embeddings via Perch-Hoplite (resumable chunks)
    │   └── probe.py              # linear probe + few-shot on the embeddings
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
├── Embeddings\perch_v2\embeddings.npz  # Perch 2.0: emb_first / emb_mean (14,098 × 1536)
└── old\                          # 2025 metadata + splits, index.csv (reference only)
```

## Pipeline

1. `Preprocess.ipynb`: cleaning → splits → spectrograms
2. `Dataset_exploration.ipynb`
3. `Models/EfficientNet_01 … 06.ipynb` (kernel *Python (onlab-cetacean)*)
4. `Perch_embeddings.ipynb`, then `Models/Perch2_01.ipynb` (kernel *Python (onlab-perch, Perch 2.0)*)
5. `Models/Evaluation.ipynb`

Command-line equivalents: `python -m src.preprocess.clean`, `python -m src.preprocess.split`,
`python -m src.preprocess.preprocess`, `python -m src.EfficientNet.train --split main --run EfficientNet_01`,
`.venv-perch\Scripts\python -m src.Perch.embed`.

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

Result: 15,248 files → **14,098 clean clips, 44 species, 378 recordings**; 1,151 excluded (label_conflict 535,
multi_species 299, too_short 272, species_not_in_44 31, duplicate_audio 13, no_audio 1). Sample-rate flags:
670 low, 1,874 high.

A *recording* = the first 5 characters of the Watkins filename (tape/session), e.g. `61025` in `6102500A.wav`.

## Splits and why there are two

| Split | Species | Method | Sizes (train / val / test) | What it measures |
| --- | --- | --- | --- | --- |
| **main** (`Annotations/`) | 44 | clip-level, stratified 70/15/15, seed 42 (same method as 2025) | 9,868 / 2,115 / 2,115 | recognising species in recordings the model has **partly seen** |
| **grouped** (`Annotations/grouped_34/`) | 34 | by recording: each recording entirely in train, val *or* test (across all species, since some tapes hold several species), ≈70/15/15 per species | 9,502 / 2,102 / 2,057 | recognising species in recordings the model has **never heard** |

**Why the main split keeps all 44 species:** comparability with the 2025 work and with most Watkins papers, which
also split by clip, and every species keeps enough clips.

**Why a second, 34-species split:** clips from one recording share far more than the species: often the same
animals, hydrophone, boat noise, tape hiss and sample rate. With a clip-level split the model can score well by
recognising *the recording* instead of *the species*. The project's research question is generalization to new
recordings, so we also need a split where no recording is shared. A species needs at least 3 recordings to be in
train, val and test without sharing one; **10 species have only 1–2 recordings** (Tucuxi, Heaviside's, Dusky,
Dall's porpoise, Leopard seal, Boutu, Southern right whale, Long-beaked common dolphin, Rough-toothed dolphin,
Melon-headed whale), so they are left out of this split only. **The gap between the two results shows how much
a model relies on recognising recordings.**

Caveat: the grouped val/test sets often contain only one recording per species, so per-species results and
validation-based choices are noisy. That is why tuning runs use 3 seeds.

## EfficientNet

**Input:** mel spectrogram, 44 kHz mono, n_fft 1024, hop 512, 128 mels, power → dB, per-clip z-score, first 400
frames (≈4.65 s; shorter clips zero-padded). *Change vs 2025:* fixed n_fft (2025 varied it with clip length).
**Model:** ImageNet EfficientNet-B0, 1 → 3 channels, new head, full fine-tuning.

**2025 recipe (EfficientNet_01/02/02b):** batch 16, AdamW lr 1e-3 wd 1e-4, class-weighted cross-entropy,
ReduceLROnPlateau (×0.5, patience 2), grad clip 1.0, max 10 epochs, early stopping patience 3, best checkpoint =
lowest val loss. Seed 42, mixed precision.

| Run | Split | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| EfficientNet 2025 (old, for reference) | 2025 clip-level, leaky, uncleaned | 0.947 | 0.91 |
| **EfficientNet_01** | main | **0.971** | **0.944** |
| EfficientNet_02 | grouped (seed 42) | 0.606 | 0.436 |

With clean data and the same clip-level split EfficientNet does *better* than in 2025, so the cleaning removed noise
rather than an advantage. But **most of that accuracy depends on having heard the recording before**: split by
recording, accuracy falls to ≈0.61 (mean per-class F1 on the 34 shared species 0.95 → 0.44).

### Tuning on the grouped split (DL course Labs 3 and 5; one change at a time, 3 seeds each)

Selection rule, fixed in advance: best mean **validation** macro-F1. Test numbers are only reported.

| Run | Change | Val macro-F1 | Test macro-F1 | Test acc | Best epoch |
| --- | --- | --- | --- | --- | --- |
| 02b | 2025 recipe, 3 seeds | 0.251 ± 0.03 | 0.447 ± 0.04 | 0.626 ± 0.07 | 1.7 |
| 03 | lr 1e-3 → 1e-4, weight decay 1e-4 → 1e-2, max 15 epochs / patience 4 | 0.296 ± 0.01 | 0.423 ± 0.04 | 0.577 ± 0.05 | 1.3 |
| 04 | 03 + augmentation (random low-pass, SpecAugment, noise, random time offset) | 0.162 ± 0.02 | 0.259 ± 0.02 | 0.400 ± 0.03 | 1.0 |
| **05** | **03 + checkpoint / early stopping / LR schedule on val macro-F1** | **0.364 ± 0.01** | **0.484 ± 0.02** | **0.658 ± 0.01** | 9.7 |
| 06 | 04 + checkpoint selection on val macro-F1 | 0.344 ± 0.02 | 0.451 ± 0.05 | 0.668 ± 0.07 | 6.0 |

**What was learned**
- **The key fix was the selection metric.** On unseen recordings the validation *loss* rises from epoch 1 (the model
  becomes over-confident on its mistakes) while validation *accuracy/F1* keep improving. Early stopping on loss
  therefore kept the epoch-1 model (best epoch ≈ 1 in 02b/03/04). Selecting on validation macro-F1 lets training
  continue (best epoch ≈ 10) → **EfficientNet_05: test macro-F1 0.447 → 0.484, and much more stable across seeds.**
- Lower lr + stronger weight decay helped on validation but not clearly on test (within seed noise).
- Augmentation did not help, even with the selection fix (06 < 05 on validation). The random low-pass was meant to
  break the sample-rate shortcut; either the shortcut is not the main problem or the augmentation is too strong.
- Stopped tuning here: more runs on the same small validation set would mostly fit its noise (winner's curse).

## Perch 2.0

**Setup.** Official **Perch v2 CPU model** from Kaggle via **Perch-Hoplite** (`perch_v2_cpu` preset), TensorFlow 2.21
on CPU, native Windows (no WSL2 needed). Hoplite's preprocessing: 32 kHz, 5 s windows (hop 5 s), peak-normalised.
1536-dim embedding per window. All 14,098 clips (21,000 windows) in 54 min. Two clip embeddings:
`first` = first 5 s window (comparable with EfficientNet's first 4.65 s), `mean` = mean of up to 12 windows (paper protocol).

**Classifier (linear probe):** standardise with train statistics, logistic regression with balanced class weights,
C chosen on validation macro-F1 from {0.001 … 10}, test evaluated once.

| Split | Embedding | Accuracy | Precision (macro) | Recall (macro) | Macro F1 | ROC-AUC (ovr) |
| --- | --- | --- | --- | --- | --- | --- |
| main | first | 0.941 | 0.941 | 0.915 | 0.924 | 0.999 |
| main | mean | 0.943 | 0.937 | 0.918 | 0.924 | 0.999 |
| grouped | first | 0.722 | 0.601 | 0.544 | 0.543 | 0.936 |
| grouped | mean | 0.724 | 0.605 | 0.574 | 0.568 | 0.947 |

**Few-shot** (k clips per species from train, 5 random draws, full test set; as in the Perch 2.0 paper):

| k | grouped macro-F1 | grouped ROC-AUC | main macro-F1 | main ROC-AUC |
| --- | --- | --- | --- | --- |
| 4 | 0.34 | 0.90 | 0.47 | 0.96 |
| 8 | 0.39 | 0.92 | 0.59 | 0.98 |
| 16 | 0.45 | 0.93 | 0.70 | 0.99 |
| 32 | 0.51 | 0.94 | 0.78 | 0.99 |

**What the embedding captures** (`outputs/perch/pca_umap.png`, `outputs/evaluation/perch_hdbscan.csv`)
- Without labels, UMAP separates several species into clear islands (killer whale, fin whale, sperm whale,
  long-finned pilot whale, pantropical spotted dolphin).
- The large-scale layout also follows the original **sample rate** (low-rate baleen/killer-whale recordings vs
  high-rate dolphin recordings): partly biology, partly equipment.
- **HDBSCAN** (no labels; PCA-50 or UMAP-10): clusters agree with **recordings** about as much as with **species**
  (e.g. PCA-50: NMI species 0.79 vs recording 0.79; UMAP-10: 0.67 vs 0.76). The embedding encodes recording
  conditions as well as calls, which matters for the later unknown-sound clustering.

## Comparison and interpretation

See `Models/Evaluation.ipynb`, `outputs/evaluation/final_comparison.png`.

- **Seen recordings:** fine-tuned EfficientNet ≥ frozen Perch (0.944 vs 0.924 macro-F1).
- **Unseen recordings:** frozen Perch > best tuned EfficientNet (0.543–0.568 vs 0.484 macro-F1; 0.72 vs 0.66 acc),
  with a linear classifier that trains in seconds. This matches the Perch 2.0 paper's claim, now on a test where
  recordings are held out, which the paper did not do.
- Per species the picture is mixed: Perch is better on 18 of 34 species; 5 (Perch) and 6 (EfficientNet) species
  get F1 ≈ 0. The two models fail on different species.
- Everything on unseen recordings is far below the seen-recording numbers: generalization to new recording
  conditions, not species recognition per se, is the hard part.

## Open questions / next steps

- Combine both: fine-tune Perch, or a small MLP / ensemble of Perch + EfficientNet (they fail on different species).
- Recording-balanced sampling (a few long recordings dominate training), per-frequency background removal.
- Cross-dataset test: train on Watkins, test on DCLDE killer whale (and later DCLDE sites against each other).
- Compare more embedding models available in Hoplite (BirdNET, SurfPerch, AVES), as in the Perch 2.0 paper.
- Verify the 519 White-beaked / White-sided labels if the WHOI site comes back.

## Environment

Windows 11 (Smart App Control off), i7-14650HX, 16 GB RAM, RTX 5050 Laptop GPU (8 GB).

**EfficientNet** (`.venv`, kernel *Python (onlab-cetacean)*): Python 3.12, PyTorch 2.11 + CUDA 12.8.
```bash
python -m venv .venv
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install pandas scikit-learn soundfile librosa matplotlib seaborn tqdm jupyter ipykernel
.venv\Scripts\python -m ipykernel install --user --name onlab-cetacean
```

**Perch 2.0** (`.venv-perch`, kernel *Python (onlab-perch, Perch 2.0)*): separate environment so TensorFlow and
PyTorch don't conflict. TensorFlow runs on CPU on native Windows (no GPU support there).
```bash
python -m venv .venv-perch
.venv-perch\Scripts\pip install "perch-hoplite[tf]" umap-learn seaborn
.venv-perch\Scripts\python -m ipykernel install --user --name onlab-perch
```
Versions: perch-hoplite 1.0.2, TensorFlow 2.21.0, umap-learn 0.6.0, scikit-learn 1.9.1. The model (~390 MB) is
downloaded to `%USERPROFILE%\.cache\kagglehub` on first use.

**Speed:** EfficientNet ≈ 36–55 s per epoch on the RTX 5050 (vs ~17 min per epoch on the 2025 GTX 1050);
Perch embedding ≈ 0.15 s per 5 s window on CPU.
