# Notes (dated log)

Short entries, newest at the bottom. This file feeds the supervisor updates; `README.md` has the current state.

## 2026-11-05: Setup
- Project folder + git repo created.
- AIMM Portugal declined to share data (dataset smaller than expected). Plan: Watkins = baseline, DCLDE = generalization, NOAA later.

## 2026-11-13: Perch 2.0 marine paper read
- Notes in Notion (Phase 1, step 3). Key gap: no cross-site / cross-hydrophone tests; Watkins described as possibly not representative of hydrophone PAM data.

## 2026-11-17: Watkins data recovered and audited
- **Environment:** native Windows Python was blocked by Smart App Control ("DLL load failed ... Application Control policy"). Turned off → numpy / scikit-learn / PyTorch work natively.
- **Old 2025 setup found** (Bens-Internship-2025): split = random stratified 70/15/15 by species, seed 42 (train 10,784 / val 2,311 / test 2,312). Old Perch setup (Bens-Internship-2026): TensorFlow + tensorflow_hub, Perch v2 CPU model from Kaggle, 32 kHz, one 5 s window per file.
- **Audio source:** official WHOI site has been down for ~1 year (redirects to maintenance page) → used the unofficial archive.org copy `watkins-marine-mammal-sound-database-full-cuts` (6.7 GB zip; 15,248 WAVs; 54 species folders; `<species>/<year>/<file>.wav`). WHOI terms: free for personal/academic use, no reposting on third-party sites, so this mirror likely breaks their terms. Credit: "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution."
- USB drive disconnected once during download (NTFS "Delayed Write Failed") → downloaded to C: first, then unpacked to D:.
- **54 vs 44 species:** 4 are spelling variants (`Dall'sPorpoise` = `Dall_sPorpoise`, Fraser's, Risso's, Heaviside's); 10 rare species (1–7 clips each) were not in the 2025 dataset.
- **Problems in the 2025 dataset** (`old/index.csv`, one row per file):
  - 15,407 rows but only 15,107 files: **299 clips stored under 2 species** (multi-species recordings, e.g. pilot whale + sperm whale).
  - **Leakage:** 64 identical files in both train and test (different labels); 317 / 321 test recordings also in train.
  - **519 clips** labelled White-beaked dolphin in 2025 are White-sided dolphin in the download. Can't verify (WHOI offline). The 2025 report's main errors were White-sided dolphin → possibly label noise.
  - → the 2025 result (94.7 % acc) is optimistic.

## 2026-11-17: Rebuild from scratch (REPSOL structure)
- Decision: retrain everything on cleaned data, project restructured to the REPSOL layout (see README).
- **Cleaning rules:** keep the 44 species with one spelling each; drop multi-species clips, label conflicts, unreadable files, clips < 0.2 s, byte-identical duplicates; flag (keep) sample rates < 4 kHz / > 96 kHz. Every dropped clip + reason in `Annotations/excluded.csv`.
- **Two splits:**
  - main = 44 species, clip-level, stratified 70/15/15, seed 42 (comparable to 2025);
  - grouped = 34 species, by recording (no recording in two sets). 10 species have < 3 recordings and can't be in train, val and test without sharing a recording, so they are left out of this split only.
- **Recordings per species** (after cleaning, recording = first 5 chars of filename): 4 species have 1 recording (Tucuxi, Heaviside's, Dusky, Dall's), 6 have 2, 8 have 3; Killer whale has 65.
- **Spectrograms:** same as 2025 (44 kHz, 128 mels, first 400 frames ≈ 4.65 s) but fixed n_fft 1024 / hop 512 (2025 varied n_fft with clip length); only first ~5.2 s loaded.
- **EfficientNet:** 2025 recipe unchanged (so only the data differs). EfficientNet_01 = main split, EfficientNet_02 = grouped split.
- **Environment:** PyTorch 2.11 + CUDA 12.8 runs on the RTX 5050 natively on Windows (WSL2 only needed later for Perch / TensorFlow).

## 2026-11-17: Results of the rebuild
- Pipeline ran end to end (~35 min): 14,098 clean clips / 44 species / 378 recordings; 1,151 excluded (label_conflict 535, multi_species 299, too_short 272, species_not_in_44 31, duplicate_audio 13, no_audio 1).
- Bug found and fixed: the first grouped split kept recordings together *per species*, but some tapes hold several species → 6 recordings were shared between train and test. Now a recording is assigned once across all species (0 shared). EfficientNet_02 retrained on the corrected split.
- **EfficientNet_01** (main, 44 species): acc **0.971**, macro F1 0.944 (2025: 0.947 / 0.91).
- **EfficientNet_02** (grouped, 34 species, unseen recordings): acc **0.606**, macro F1 0.436. Train acc 96 %, val loss rises from epoch 1 → memorises recordings. 7 species F1 = 0.
- Mean F1 on the 34 shared species: 0.95 (seen recordings) → 0.44 (unseen recordings).
- **Shortcut suspect:** low-sample-rate clips show an empty band at the top of the 44 kHz spectrogram; its height = original sample rate = recording equipment. Idea: limit all clips to a common band and retrain.
- Grouped val/test are small per species (often one recording) → noisy per-species numbers.
- Speed: ~36 s/epoch on RTX 5050 vs ~17 min/epoch in 2025 (GTX 1050).

## 2026-11-17: Improving EfficientNet on unseen recordings (ideas from DL Labs 3 + 5)
- Protocol: one change at a time, seeds 42/43/44, choose on mean val macro-F1, report test only (no tuning on test).
- 02b (2025 recipe, 3 seeds): val F1 0.251, test F1 0.447 ± 0.04, acc 0.626.
- 03 (lr 1e-4, wd 1e-2): val F1 0.296, test F1 0.423 (within noise).
- 04 (03 + random low-pass, SpecAugment, noise, random offset): val F1 0.162, test F1 0.259 → looked terrible.
- **Self-evaluation:** best epoch was ~1 in every run. On unseen recordings val loss rises from epoch 1 (over-confident mistakes) while val accuracy keeps rising (04: 24 → 37 % in 5 epochs), so early stopping on loss kept an under-trained model.
- **Fix:** select checkpoint / early stop / LR schedule on val macro-F1 (REPSOL convention).
  - 05 (03 + F1 selection): **val F1 0.364, test F1 0.484 ± 0.02, acc 0.658 ± 0.01**, best epoch ~10 → best EfficientNet.
  - 06 (04 + F1 selection): val F1 0.344, test F1 0.451. Augmentation doesn't help.
- Stopped tuning (small val set → winner's curse).

## 2026-11-17: Perch 2.0
- No WSL2 needed: perch-hoplite 1.0.2 + TensorFlow 2.21 (CPU) run on native Windows in a separate venv `.venv-perch`. Official Perch v2 CPU model (Kaggle) via preset `perch_v2_cpu`. (Hoplite also has an ONNX version that could use the GPU, but its weights are an unofficial community conversion.)
- Embedded 14,098 clips (21,000 five-second windows) in 54 min, CPU, in parallel with GPU training.
- Linear probe (logistic regression, C chosen on val): main acc 0.941 / F1 0.924; **grouped acc 0.722 / F1 0.543 (first window), 0.724 / 0.568 (mean of windows)** → beats best EfficientNet (0.658 / 0.484) on unseen recordings; EfficientNet slightly better on seen recordings (0.971 / 0.944).
- Few-shot on grouped: k=32 per species → F1 0.51, already > EfficientNet trained on everything.
- Per species mixed: Perch better on 18/34; the two fail on different species.
- UMAP: several species form clean islands; layout also follows sample rate. HDBSCAN clusters agree with recordings about as much as with species.

## Next
- Ideas: fine-tune Perch / MLP / ensemble with EfficientNet; recording-balanced sampling; DCLDE cross-dataset test; other Hoplite models (BirdNET, SurfPerch, AVES).
- Send the update to Mohammed (overdue).

## Environment

| Item | Version / value |
| --- | --- |
| OS | Windows 11 Pro (10.0.26300), Smart App Control off |
| CPU / RAM | Intel i7-14650HX (16 cores) / 16 GB |
| GPU | NVIDIA GeForce RTX 5050 Laptop, 8 GB, compute 12.0, driver 617.14 |
| Python | 3.12.10, venv in `.venv/` |
| PyTorch | 2.11.0+cu128 (torchvision, torchaudio) |
| Other | numpy 2.5, pandas 3.1, scikit-learn 1.9, librosa 1.0, soundfile, matplotlib 3.11, seaborn 0.13 |
| Perch env | `.venv-perch`: perch-hoplite 1.0.2, TensorFlow 2.21 (CPU), umap-learn 0.6 |
| WSL2 | not installed (not needed so far) |
| Data | external USB drive `D:\UNI&WORK\Watkins` (ADATA HV620S, 1 TB) |
