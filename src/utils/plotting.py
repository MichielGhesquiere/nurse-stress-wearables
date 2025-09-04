import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_signals_with_labels(df: pd.DataFrame, out_path: str, title: str = "Signals with stress labels",
                             cols: Optional[List[str]] = None, max_points: int = 20000):
    """Plot raw signals (and selected feature columns) with label overlay.

    - df must have 'datetime' and 'label'.
    - cols: columns to plot; defaults to ['EDA','HR','TEMP','movement_magnitude'] if present.
    - Downsamples if more than max_points to keep plots responsive.
    """
    ensure_dir(os.path.dirname(out_path))
    if cols is None:
        cols = [c for c in ['EDA','HR','TEMP','movement_magnitude'] if c in df.columns]
    if not cols:
        return

    data = df.sort_values('datetime')
    if len(data) > max_points:
        data = data.iloc[:: max(1, len(data)//max_points), :]

    fig, axes = plt.subplots(nrows=len(cols)+1, ncols=1, figsize=(14, 3*(len(cols)+1)), sharex=True)
    # Normalize axes to a flat list
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for i, c in enumerate(cols):
        axes[i].plot(data['datetime'], data[c], lw=0.8)
        axes[i].set_ylabel(c)
        axes[i].grid(True, alpha=0.3)

    # Label track
    ax = axes[-1]
    ax.step(data['datetime'], (data['label'] != 0).astype(int), where='post', color='crimson')
    ax.set_ylabel('stress')
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_classification_report(y_true, y_pred, out_txt: str, out_png: Optional[str] = None):
    ensure_dir(os.path.dirname(out_txt))
    report = classification_report(y_true, y_pred)
    with open(out_txt, 'w') as f:
        f.write(report)

    if out_png is not None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(6,5))
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        plt.tight_layout()
        ensure_dir(os.path.dirname(out_png))
        plt.savefig(out_png, dpi=150)
        plt.close(fig)


def save_curves(y_true, y_score, out_dir: str, prefix: str = ""):
    ensure_dir(out_dir)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}roc.png"), dpi=150)
    plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}pr.png"), dpi=150)
    plt.close()


def save_per_subject_metrics(rows: List[dict], out_csv: str):
    import pandas as pd
    ensure_dir(os.path.dirname(out_csv))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
