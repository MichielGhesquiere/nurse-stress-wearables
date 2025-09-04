# Nurse Stress Prediction (Wearable Sensors)

End-to-end pipeline for stress detection on the Nurse Wearables dataset. Includes scalable data loading, feature engineering and caching, Random Forest (RF) training on engineered features, a 1D CNN baseline on raw time windows (DL), evaluation visualizations, and CI.

Dataset: https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors/data

---

## What’s in this repo

- Robust CSV ➜ Parquet converter for faster IO.
- Memory-safe loaders and per-subject processing for the 11.5M-row `merged_data.csv`.
- Feature engineering with a partitioned feature store under `data/features/`.
- RF pipeline on engineered features; DL pipeline on raw windows.
- Saved artifacts: reports, curves (ROC/PR), confusion matrices, feature importances, per-subject metrics, and models.
- CI + basic tests; code style with Black/ruff.

---

## Project structure

```
├── config/
│   └── config.yaml                 # Main configuration (paths, RF/DL params, importance flags)
├── data/
│   ├── merged_data.csv             # Source CSV (not tracked)
│   ├── features/                   # Feature store (parquet)
│   │   └── feature_set=v1/id=SUBJ/d=DATE/part.parquet
│   └── model_rf.joblib             # Saved RF model
├── results/
│   ├── curves/                     # ROC and PR curves (rf_, dl_, and combined)
│   ├── rf_classification_report.txt
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importance.csv / .png
│   ├── rf_permutation_importance.csv / .png (optional)
│   ├── rf_per_subject_metrics.csv
│   └── model_dl.pt                 # Saved DL model weights
├── scripts/
│   ├── train_model.py              # Main entrypoint (RF, DL, or both)
│   ├── convert_to_parquet.py       # CSV ➜ Parquet with DuckDB/Arrow fallback
│   └── create_subset_data.py       # Utility for smaller smoke tests
├── src/
│   ├── data/data_loader.py         # Chunked loading, subject list, subject filter
│   ├── features/
│   │   ├── signal_features.py      # Feature engineering (rolling, movement, physiological)
│   │   └── feature_store.py        # Write per-subject-day partitions
│   ├── training/
│   │   ├── trainer.py              # RF training (by-subject or from cached features)
│   │   └── dl_trainer.py           # 1D CNN baseline and dataset windowing
│   └── utils/plotting.py           # Plots, curves, combined ROC, reports
├── tests/                          # Minimal tests for loader and features
├── env.yml                         # Conda environment (Python + DS/ML stack)
└── README.md
```

---

## Setup

1) Create environment

```powershell
conda env create -f env.yml; conda activate nurse-stress
```

2) Place data

- Put `data/merged_data.csv` in the `data/` folder.
- Optional but recommended: convert to Parquet for speed.

```powershell
python -m scripts.convert_to_parquet --csv "data/merged_data.csv" --out "data/merged_data.parquet" --engine duckdb
```

---

## Configuration

Edit `config/config.yaml`.

- results_dir: where artifacts are saved
- feature_store_root: where engineered features are cached (`data/features`)
- model_type: rf | dl | both
- use_feature_store: true (use cached features for RF if present)
- feature_set: v1 (partition name used in the store)
- dl.*: window_size, stride, batch_size, epochs, lr, max_windows_per_subject
- feature importance flags:
  - feature_importance_top_n
  - compute_permutation_importance
  - feature_importance_sample

---

## How to run

Train RF and/or DL and save artifacts.

```powershell
python -m scripts.train_model --config "config/config.yaml" --data-path "data/merged_data.csv"
```

Notes:
- RF prefers cached features under `data/features/feature_set=v1/...`. If not found, it falls back to on-the-fly feature extraction per subject.
- DL builds a raw-window dataset from the CSV; increase `dl.epochs` (e.g., 20–50) for stronger training.

---

## RF vs DL

- RF: Trains on engineered features (movement magnitude, rolling stats, physiological aggregates). Fast to train, interpretable via feature importances and permutation importance.
- DL (1D CNN): Trains directly on raw windows (e.g., 60-second windows with stride). Captures temporal patterns without manual features but needs more epochs and GPU for best results.

Outputs for comparison:
- Individual ROC/PR curves for each model under `results/curves/` (rf_* and dl_*), plus a combined ROC: `results/curves/roc_rf_vs_dl.png`.
- `results/comparison_summary.txt` includes classification reports.

Tips:
- If DL shows poor specificity (low true negatives), try more epochs, adjust window/stride, or rebalance windows.
- For RF interpretability, inspect `rf_feature_importance.csv/.png` and enable permutation importance for robustness.

---

## Data details (summary)

`merged_data.csv` ≈ 11.5M rows with columns: `X, Y, Z, EDA, HR, TEMP, id, datetime, label`.
- Orientation: X, Y, Z
- Physiological: EDA, HR, TEMP
- Labels: stress vs non-stress (converted to binary)
- `id` is normalized to string type across the pipeline.

---

## Troubleshooting

- Parquet write error on id dtype: fixed by coercing `id` to string in the feature store.
- Large CSV memory: processed per-subject and with chunked loading; consider Parquet for faster IO.
- Windows path/quoting: scripts handle Windows paths; ensure you use double quotes around paths in PowerShell.
- DL not using GPU: ensure PyTorch with CUDA is installed; logs show `device=cuda` when active.

---

## License

This repository provides code and configuration around the public dataset. See dataset terms on Kaggle/Dryad for data usage.