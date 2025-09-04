import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
import joblib
from typing import Dict, Any, List
import logging


class StressDetectionTrainer:
    """Train stress detection models with memory-efficient processing"""

    def __init__(self, model, feature_extractor, config: Dict[str, Any]):
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train_by_subjects(self, data_loader, subject_ids: List[str]) -> Dict:
        """Train model processing one subject at a time"""
        all_features = []
        all_labels = []

        results_dir = self.config.get("results_dir")
        feature_store_root = self.config.get("feature_store_root", None)
        save_features = bool(self.config.get("save_features", False))

        if results_dir:
            from pathlib import Path

            Path(results_dir).mkdir(parents=True, exist_ok=True)
            plots_root = Path(results_dir) / "plots"
            plots_root.mkdir(parents=True, exist_ok=True)
            # Lazy import to avoid heavy deps at module import
            from src.utils.plotting import plot_signals_with_labels

        # Feature store lives under data/ by default
        from pathlib import Path

        if save_features:
            feats_root = Path(feature_store_root) if feature_store_root else Path("data/features")
            feats_root.mkdir(parents=True, exist_ok=True)

        for subject_id in subject_ids:
            self.logger.info(f"Processing subject {subject_id}")

            # Load subject data
            subject_data = data_loader.load_subject_data(subject_id)

            if subject_data.empty:
                continue

            # Extract features
            features = self.feature_extractor.extract_all_features(subject_data)

            # Optional: save features per subject (and per day)
            if save_features:
                try:
                    from src.features.feature_store import write_features_per_day

                    written = write_features_per_day(features, str(feats_root), feature_set="v1", force=False)
                    self.logger.info(f"Wrote {len(written)} feature partitions for subject {subject_id}")
                except Exception as e:
                    self.logger.warning(f"Feature store write failed for subject {subject_id}: {e}")
                # Also a quick plot for a short window (first ~20k rows)
                try:
                    sample = features[
                        ["id", "datetime", "label"]
                        + [c for c in ["EDA", "HR", "TEMP", "movement_magnitude"] if c in features.columns]
                    ]
                    if results_dir:
                        plot_signals_with_labels(
                            sample.head(20000),
                            out_path=str((plots_root / f"signals_subject={subject_id}.png")),
                            title=f"Subject {subject_id} signals + labels",
                        )
                except Exception as e:
                    self.logger.warning(f"Plotting failed for subject {subject_id}: {e}")

            # Prepare for training
            feature_cols = [col for col in features.columns if col not in ["id", "datetime", "label"]]

            X_subject = features[feature_cols].fillna(0)
            y_subject = features["label"] != 0  # Binary classification

            all_features.append(X_subject)
            all_labels.append(y_subject)

        # Combine all subjects
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        # Train model
        self.model.fit(X, y)

        # Evaluate
        y_pred = self.model.predict(X)
        y_score = None
        # Try to get probabilities for ROC/PR
        if hasattr(self.model, "predict_proba"):
            try:
                y_score = self.model.predict_proba(X)[:, 1]
            except Exception:
                y_score = None

        report_text = classification_report(y, y_pred)
        metrics = {
            "classification_report": report_text,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }

        # Save evaluation artifacts
        if results_dir:
            from src.utils.plotting import save_classification_report, save_curves, save_per_subject_metrics
            from pathlib import Path

            out_txt = Path(results_dir) / "rf_classification_report.txt"
            out_png = Path(results_dir) / "rf_confusion_matrix.png"
            try:
                save_classification_report(y, y_pred, str(out_txt), str(out_png))
            except Exception as e:
                self.logger.warning(f"Saving reports failed: {e}")

            # Save ROC/PR curves if we have scores
            if y_score is not None:
                try:
                    curves_dir = Path(results_dir) / "curves"
                    curves_dir.mkdir(parents=True, exist_ok=True)
                    save_curves(y.astype(int), y_score, str(curves_dir), prefix="rf_")
                except Exception as e:
                    self.logger.warning(f"Saving ROC/PR failed: {e}")

            # Feature importance
            try:
                import matplotlib.pyplot as plt

                top_n = int(self.config.get("feature_importance_top_n", 30))
                fi = None
                if hasattr(self.model, "feature_importances_"):
                    fi = pd.Series(self.model.feature_importances_, index=X.columns)

                if fi is not None:
                    fi_sorted = fi.sort_values(ascending=False).head(top_n)
                    fi_df = fi_sorted.reset_index().rename(columns={"index": "feature", 0: "importance"})
                    fi_df.columns = ["feature", "importance"]
                    fi_df.to_csv(Path(results_dir) / "rf_feature_importance.csv", index=False)
                    plt.figure(figsize=(10, max(4, int(top_n * 0.4))))
                    fi_sorted[::-1].plot(kind="barh")
                    plt.tight_layout()
                    plt.savefig(Path(results_dir) / "rf_feature_importance.png", dpi=150)
                    plt.close()

                # Optional permutation importance (costly)
                if bool(self.config.get("compute_permutation_importance", False)):
                    n_sample = int(self.config.get("feature_importance_sample", 50000))
                    idx = slice(None) if len(y) <= n_sample else slice(0, n_sample)
                    y_slice = y.iloc[idx] if hasattr(y, "iloc") else (y[idx] if hasattr(y, "__getitem__") else y)
                    X_slice = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

                    r = permutation_importance(self.model, X_slice, y_slice, n_repeats=3, random_state=42)
                    pi = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
                    pi_df = pi.reset_index().rename(columns={"index": "feature", 0: "importance"})
                    pi_df.columns = ["feature", "importance"]
                    pi_df.to_csv(Path(results_dir) / "rf_permutation_importance.csv", index=False)

                    # Plot top-N permutation importance
                    pi_top = pi.head(top_n)
                    plt.figure(figsize=(10, max(4, int(top_n * 0.4))))
                    pi_top[::-1].plot(kind="barh")
                    plt.tight_layout()
                    plt.savefig(Path(results_dir) / "rf_permutation_importance.png", dpi=150)
                    plt.close()
            except Exception as e:
                self.logger.warning(f"Feature importance failed: {e}")

            # Per-subject breakdown after training
            rows = []
            for subject_id in subject_ids:
                try:
                    df = data_loader.load_subject_data(subject_id)
                    if df.empty:
                        continue
                    feats = self.feature_extractor.extract_all_features(df)
                    feat_cols = [c for c in feats.columns if c not in ["id", "datetime", "label"]]
                    Xs = feats[feat_cols].fillna(0)
                    ys = (feats["label"] != 0).astype(int)
                    yp = self.model.predict(Xs)
                    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

                    p, r, f1, _ = precision_recall_fscore_support(ys, yp, average="binary", zero_division=0)
                    acc = accuracy_score(ys, yp)
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "n": int(len(ys)),
                            "acc": acc,
                            "precision": p,
                            "recall": r,
                            "f1": f1,
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Per-subject metrics failed for {subject_id}: {e}")

            if rows:
                try:
                    save_per_subject_metrics(rows, str(Path(results_dir) / "rf_per_subject_metrics.csv"))
                except Exception as e:
                    self.logger.warning(f"Saving per-subject metrics failed: {e}")

        return metrics

    def save_model(self, filepath: str):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def train_from_feature_store(self, features_root: str, feature_set: str = "v1") -> Dict:
        """Train RF directly from cached parquet feature partitions.

        Expects layout: {features_root}/feature_set={feature_set}/id=*/d=*/*.parquet
        Returns metrics including y_true/y_pred/y_score for plotting.
        """
        from pathlib import Path
        root = Path(features_root) / f"feature_set={feature_set}"
        files = list(root.glob("id=*/d=*/*.parquet"))
        if not files:
            raise RuntimeError(f"No feature files found under {root}")

        self.logger.info(f"Loading cached features: {len(files)} partitions from {root}")
        X_parts, y_parts = [], []
        for p in files:
            try:
                df = pd.read_parquet(p)
                if 'label' not in df.columns:
                    continue
                feat_cols = [c for c in df.columns if c not in ['id','datetime','label','date']]
                if not feat_cols:
                    continue
                X_parts.append(df[feat_cols].fillna(0))
                y_parts.append((df['label'] != 0).astype(int))
            except Exception as e:
                self.logger.warning(f"Failed reading {p}: {e}")

        if not X_parts:
            raise RuntimeError("No usable feature partitions loaded.")

        X = pd.concat(X_parts, ignore_index=True)
        y = pd.concat(y_parts, ignore_index=True)
        self.logger.info(f"Training RF on cached features: X={X.shape}, pos={int(y.sum())}/{len(y)}")

        # Train
        self.model.fit(X, y)

        # Evaluate
        y_pred = self.model.predict(X)
        y_score = None
        if hasattr(self.model, 'predict_proba'):
            try:
                y_score = self.model.predict_proba(X)[:,1]
            except Exception:
                y_score = None

        metrics = {
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'y_true': y.astype(int).to_numpy(),
            'y_pred': y_pred.astype(int),
            'y_score': None if y_score is None else np.asarray(y_score, dtype=float),
            'feature_names': list(X.columns),
        }

        # Save artifacts
        results_dir = self.config.get('results_dir')
        if results_dir:
            try:
                from src.utils.plotting import save_classification_report, save_curves
                out_txt = Path(results_dir) / 'rf_classification_report.txt'
                out_png = Path(results_dir) / 'rf_confusion_matrix.png'
                save_classification_report(y, y_pred, str(out_txt), str(out_png))
                if y_score is not None:
                    curves_dir = Path(results_dir) / 'curves'
                    curves_dir.mkdir(parents=True, exist_ok=True)
                    save_curves(y.astype(int), y_score, str(curves_dir), prefix='rf_')
            except Exception as e:
                self.logger.warning(f"Saving RF artifacts failed: {e}")

            # Feature importance (same as in train_by_subjects)
            try:
                import matplotlib.pyplot as plt
                top_n = int(self.config.get('feature_importance_top_n', 30))
                fi = None
                if hasattr(self.model, 'feature_importances_'):
                    fi = pd.Series(self.model.feature_importances_, index=X.columns)
                if fi is not None:
                    fi_sorted = fi.sort_values(ascending=False).head(top_n)
                    fi_df = fi_sorted.reset_index().rename(columns={'index': 'feature', 0: 'importance'})
                    fi_df.columns = ['feature','importance']
                    fi_df.to_csv(Path(results_dir)/'rf_feature_importance.csv', index=False)
                    plt.figure(figsize=(10, max(4, int(top_n*0.4))))
                    fi_sorted[::-1].plot(kind='barh')
                    plt.tight_layout()
                    plt.savefig(Path(results_dir)/'rf_feature_importance.png', dpi=150)
                    plt.close()
            except Exception as e:
                self.logger.warning(f"RF feature importance failed: {e}")

        return metrics