import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    torch = None


def _check_torch():
    if torch is None:
        raise RuntimeError("PyTorch not installed. Please install torch to use DL trainer.")


class WindowedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        z = self.net(x)
        return self.head(z)


class DLTrainer:
    def __init__(self, config: Dict[str, Any]):
        _check_torch()
        self.cfg = config.get('dl', {})
        self.logger = logging.getLogger(__name__)

    def _window_subject(self, df: pd.DataFrame, window: int, stride: int,
                        signals: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        df = df.sort_values('datetime')
        Xsig = df[signals].values
        y = (df['label'].values != 0).astype(np.int64)
        n = len(df)
        if n < window:
            return np.empty((0, window, len(signals))), np.empty((0,), dtype=np.int64)
        starts = np.arange(0, n - window + 1, stride)
        Xw = np.stack([Xsig[s:s+window] for s in starts])
        # Window label as majority in window
        yw = np.array([np.round(y[s:s+window].mean()) for s in starts], dtype=np.int64)
        return Xw, yw

    def build_dataset(self, data_loader, subject_ids: List[str]) -> Tuple[Dataset, List[str]]:
        window = int(self.cfg.get('window_size', 60))
        stride = int(self.cfg.get('stride', 30))
        max_per_subj = int(self.cfg.get('max_windows_per_subject', 2000))
        signals = [c for c in ['EDA','HR','TEMP','X','Y','Z']]
        Xs, ys = [], []
        for sid in subject_ids:
            self.logger.info(f"DL: loading subject {sid} for windowing")
            df = data_loader.load_subject_data(sid)
            if df.empty:
                continue
            # basic cleanup
            df = df.dropna(subset=['datetime'])
            for c in signals:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                else:
                    df[c] = 0.0
            df = df.dropna(subset=signals)
            Xw, yw = self._window_subject(df, window, stride, signals)
            if len(Xw) == 0:
                continue
            if len(Xw) > max_per_subj:
                idx = np.linspace(0, len(Xw)-1, max_per_subj, dtype=int)
                Xw, yw = Xw[idx], yw[idx]
            Xs.append(Xw)
            ys.append(yw)
        if not Xs:
            raise RuntimeError("No DL windows constructed")
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        self.logger.info(f"DL dataset: X={X.shape}, y_pos={y.sum()} / {len(y)}")
        return WindowedDataset(X, y), signals

    def train(self, dataset: Dataset) -> Dict[str, Any]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bs = int(self.cfg.get('batch_size', 256))
        epochs = int(self.cfg.get('epochs', 3))
        lr = float(self.cfg.get('lr', 1e-3))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
        n_channels = dataset.X.shape[-1]
        model = CNN1D(n_channels).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        self.logger.info(f"DL training: device={device}, epochs={epochs}, bs={bs}")
        model.train()
        for ep in range(epochs):
            total, correct, loss_sum = 0, 0, 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss.item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(len(yb))
            acc = correct / max(1, total)
            self.logger.info(f"epoch {ep+1}/{epochs} - loss={loss_sum/max(1,total):.4f} acc={acc:.3f}")
        return {"model": model, "device": device}

    def evaluate(self, model_obj, dataset: Dataset) -> Dict[str, Any]:
        device = model_obj["device"]
        model = model_obj["model"]
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_pred.append(pred)
                y_true.append(yb.numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
        try:
            # Get probabilities via softmax
            import torch.nn.functional as F
            loader = DataLoader(dataset, batch_size=512, shuffle=False)
            scores = []
            for xb, _ in loader:
                with torch.no_grad():
                    logits = model(xb.to(device))
                    prob = F.softmax(logits, dim=1)[:,1].cpu().numpy()
                    scores.append(prob)
            y_score = np.concatenate(scores)
        except Exception:
            y_score = None
        return {
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_score': None if y_score is None else y_score.tolist(),
        }

    def save_model(self, model_obj, path: str):
        _check_torch()
        from pathlib import Path
        Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
        torch.save(model_obj['model'].state_dict(), path)
