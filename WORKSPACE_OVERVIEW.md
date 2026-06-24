# Workspace Overview

**Location reviewed:** `C:\git hub`  
**Reviewed:** June 24, 2026

## Executive Summary

This is a research-oriented autism screening and intervention-analysis workspace containing one active MAVEN video application, a skeleton action model, an audio screening codebase, three datasets, and three large Python environments.

The MAVEN backend is operational: it loads its checkpoint on CUDA and completed an end-to-end prediction. However, the current checkpoint is a **debug/demo artifact**, not a validated model. Its metadata records `limit=5`, meaning five videos per class were used. Under the current loader this is only 8 training clips, 2 validation clips, and 10 test clips. The displayed 100% validation accuracy and AUC therefore are not reliable scientific evidence.

The general MAVEN validation design also splits video clips rather than subjects. On the full training dataset, all 80 subject/class identities occur in both training and validation, creating leakage. The separate testing folders do use subjects distinct from training.

## Workspace Inventory

| Folder | Purpose | Files | Approx. size |
|---|---|---:|---:|
| `-MAVEN-Motion-Aware-Visual-Evaluation-Network` | Active Flask/PyTorch app and React prototype | 42,212 | 5.362 GB |
| `.venv` | Shared Python 3.14 environment | 37,974 | 5.040 GB |
| `ASD-Skeleton-Model` | ST-GCN skeleton activity model and copied environment | 63,956 | 9.491 GB |
| `Audio-Neural-Network-ASD-screening` | Original audio code plus local PyTorch experiment | 49 | 0.003 GB |
| `Audio_dataset` | Balanced WAV dataset | 80 | 0.169 GB |
| `autism_data_anonymized` | Balanced MP4 train/test dataset | 19,360 | 2.276 GB |
| `MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis` | MMASD upstream repository and samples | 68 | 0.327 GB |
| `MMASD_DATASET` | Full skeleton, optical-flow, and rating data | 1,667,771 | 8.787 GB |
| **Total** |  | **1,831,470** | **31.455 GB** |

Roughly 19.9 GB in the environment-bearing folders is mostly generated dependencies rather than project source.

## MAVEN Video Application

### Deployed architecture

- Flask, PyTorch, OpenCV backend
- EfficientNetV2-S frame encoder in the active checkpoint
- Four-layer temporal Transformer
- Attention pooling and binary ASD/TD classifier
- Input: 30 frames at 112×112
- Parameters: 24,038,995
- CUDA available

### Verification

| Check | Result |
|---|---|
| Backend dependency imports | Pass |
| Python compilation | Pass |
| `/api/model_info` | HTTP 200 |
| Checkpoint load | Pass on CUDA |
| End-to-end prediction | HTTP 200 |
| Smoke-test total time | About 2.08 seconds |
| Model inference portion | About 1.16 seconds |

The backend deletes its temporary uploaded file after processing.

### Checkpoint reality

`video_model_best.pth` reports:

- Epoch 5
- EfficientNetV2-S projection
- 30 frames, 112×112
- `limit=5` per class
- Last invocation used one epoch with resume enabled
- Validation set under this configuration: 2 clips
- Test set under this configuration: 10 clips

Recorded test results are 70% accuracy, 1.0 AUC, 100% sensitivity, 40% specificity, and 0.769 F1. The sample is far too small for these values to support a performance claim.

### Data splitting

| Split | Class | Videos | Subjects |
|---|---|---:|---:|
| Training | ASD | 4,840 | 40 |
| Training | TD | 4,840 | 40 |
| Testing | ASD | 4,840 | 40 |
| Testing | TD | 4,840 | 40 |

Training and testing have no overlapping subject IDs or filenames.

The validation split is unsafe: the full-data loader produces 7,744 train clips and 1,936 validation clips, but all 80 validation subject/class identities also occur in training. Validation must be grouped by subject.

### Application and documentation issues

- README and architecture docs describe MobileNetV3 and about 4.6M parameters; the active model is EfficientNetV2-S with about 24.0M.
- README performance claims are not supported by the local history.
- `ARCHITECTURE.md` begins with accidental text `gge#`.
- Landing links use `/model`, but Flask defines `/api/model`, causing a 404.
- The React frontend does not build.
- Most React UI files are under `frontend/@/components/ui` instead of `frontend/src/components/ui`.
- TypeScript reports missing modules and import/type errors.
- ESLint reports seven errors.
- React expects `auc`; the backend returns `val_auc`.
- Obsolete `lstm_attn_ms` naming remains.
- Templates depend on external CDN assets.
- `requirements.txt` omits at least `flask-cors` and `Pillow`.
- The MAVEN Git working tree contains extensive modified, deleted, and untracked work.

## ASD Skeleton Model

- ST-GCN over MMASD 2D OpenPose or 3D ROMP skeletons
- Default: 2D, 25 joints, 150 frames, 11 activity classes
- Parameters: 1,726,127
- Loader successfully finds 1,693 2D samples
- Saved checkpoint: epoch 16, 57.71% validation accuracy, 1.1894 validation loss

The class distribution is highly imbalanced: drumming has 545 samples while the smallest classes have about 102–105.

Main risks:

- No README or requirements file
- Hard-coded local dataset path
- Random sample-level splitting may leak child/session identity
- No class-balanced loss or sampler
- `train.py` has an error-path reference to an unimported `DATASET_ROOT`
- Bundled environment points to Linux `/usr/bin/python3.12` and does not run on Windows
- Both `Lib` and `lib` trees are present, contributing to the 9.49 GB size
- Folder is not under Git

## Audio ASD Screening

The repository contains the original Python 3.6 / TensorFlow 1.15 pipeline plus an untracked PyTorch experiment.

Dataset:

- 40 AUTISM WAV files
- 40 NORMAL WAV files
- About 172.6 MB
- Deduplication reduced 80 records to 66
- Fourteen same-class duplicates were removed

The one-epoch PyTorch result has validation AUC 0.429, test AUC 0.286, 63.6% test accuracy, 100% sensitivity, and 0% specificity. It is effectively predicting the positive class and is not usable yet.

Other issues:

- Current shared environment lacks `librosa`, so the script does not run
- Linux paths are hard-coded as defaults
- New script and checkpoint folder are untracked
- No `.gitignore`
- Git flags the repository as owned by `TrustedInstaller`
- README badge says CC BY-NC-ND, while `LICENSE.md` says CC BY-NC-SA

## MMASD Dataset

| Modality | Files | Approx. size |
|---|---:|---:|
| 2D skeleton | 565,754 | 0.864 GB |
| 3D skeleton | 622,161 | 1.902 GB |
| Optical flow JPG | 479,854 | 6.022 GB |

The dataset contains 254,674 OpenPose JSON files, 933,240 NPZ files, 479,854 optical-flow images, a 758.9 MB 3D-coordinate ZIP beside extracted data, and an `ADOS_rating.xlsx` clinical/demographic spreadsheet.

The upstream README describes 1,315 segmented samples, while the local skeleton loader finds 1,693 sample folders. Inclusion rules and expected counts need documentation.

The ADOS spreadsheet and subject-derived data should be treated as sensitive and kept out of public Git history.

The upstream MMASD repository is Apache-2.0 licensed, has a deleted tracked GIF and a differently named untracked replacement, and is also flagged by Git for `TrustedInstaller` ownership. Dataset redistribution terms should be checked separately from the code license.

## Storage and Environment Health

- Shared `.venv`: Python 3.14.3, 5.04 GB
- MAVEN `venv_cuda`: Python 3.13.12, functional
- Skeleton `venv`: copied Linux Python 3.12.3 environment, broken on Windows

The workspace is dominated by duplicated PyTorch/CUDA environments. The broken skeleton environment is the clearest cleanup candidate, after confirming no unique artifacts depend on it.

## Readiness

| Area | Status |
|---|---|
| MAVEN backend demo | Working |
| MAVEN scientific validation | Not established |
| Flask navigation | Broken route |
| React frontend | Does not build |
| Video train/test separation | Good |
| Video train/validation separation | Fails |
| Skeleton artifact | Present |
| Skeleton environment | Broken |
| Audio experiment | Poor metrics and missing dependency |
| Reproducible setup | Incomplete |
| Repository cleanliness | Poor |
| Clinical/production readiness | Not ready |

## Recommended Action Plan

### Priority 0

1. Split all modalities by subject/session group.
2. Retrain MAVEN without `--limit`.
3. Keep the testing subjects locked until threshold selection and tuning are complete.
4. Report exact subjects/clips, confusion matrices, ROC-AUC, PR-AUC, sensitivity, specificity, F1, and confidence intervals.
5. Choose one frontend, fix `/model`, and repair its build/API contract.

### Priority 1

1. Add tested Python constraints, missing dependencies, and lock files.
2. Replace hard-coded paths with CLI arguments or environment variables.
3. Add skeleton documentation and audio `.gitignore`.
4. Record split manifests, dataset versions, seeds, and full training configuration in every checkpoint.
5. Review and commit or intentionally discard the large MAVEN working-tree change set.
6. Resolve Windows repository ownership.

### Priority 2

1. Rebuild, verify, and remove the broken skeleton environment.
2. Consolidate CUDA environments where practical.
3. Remove the 3D ZIP only after verifying extracted data and backups.
4. Move datasets/checkpoints to DVC, Git LFS, or a documented external registry.
5. Restrict CORS and validate media content before any non-local deployment.

## Bottom Line

The workspace contains a promising operational prototype, but the next milestone should be a subject-grouped, fully reproducible training and evaluation run. The current strongest metrics come from a tiny debug checkpoint and a leakage-prone validation design. Once evaluation is trustworthy, the backend and frontend can be consolidated into a credible demonstration system.
