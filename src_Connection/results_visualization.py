# results_visualization.py

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RESULTS_DIR, SUPERCLASSES
from src.utils import ensure_dir

# -------------------------------------------------------------------------
# I/O - loading results
# -------------------------------------------------------------------------
FILES = {
    "non_frozen": RESULTS_DIR / "non_frozen_run.csv",
    "frozen":     RESULTS_DIR / "frozen_run.csv",
}
PERCLASS = {
    "non_frozen": RESULTS_DIR / "non_frozen_run_perclass.csv",
    "frozen":     RESULTS_DIR / "frozen_run_perclass.csv",
}
CONFUSION = {
    "non_frozen": RESULTS_DIR / "non_frozen_run_cm.csv",
    "frozen":     RESULTS_DIR / "frozen_run_cm.csv",
}
OUT = RESULTS_DIR / "viz"
ensure_dir(OUT)

PHASE_ENABLED  = "post_cv"
PHASE_DISABLED = "no_cv"


def _print_missing(path: Path):
    print(f"Missing: {path}")


# --- Load CSVs with numeric columns coerced to numbers (NaN if invalid) ---
def load_numeric_csv(path: Path):
    if not path.exists():
        _print_missing(path)
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None
    if df is None or df.empty:
        return df

    # Normalize common columns if present
    for col in ["round", "accuracy", "loss", "wall_time_sec", "trainable_params", "true", "pred", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize client_id to string (GLOBAL vs numeric client ids)
    if "client_id" in df.columns:
        df["client_id"] = df["client_id"].astype(str)

    return df


# --- Load all dataframes (calling load_numeric_csv) ---------------------
dfs       = {name: load_numeric_csv(p) for name, p in FILES.items()}
percls    = {name: load_numeric_csv(p) for name, p in PERCLASS.items()}
conf_long = {name: load_numeric_csv(p) for name, p in CONFUSION.items()}

# Keep only non-empty frames
dfs       = {k: v for k, v in dfs.items() if v is not None and not v.empty}
percls    = {k: v for k, v in percls.items() if v is not None and not v.empty}
conf_long = {k: v for k, v in conf_long.items() if v is not None and not v.empty}


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def perclass_columns(df: pd.DataFrame):
    """Return mapping label->Series, accepting acc_<name> or acc_<index>."""
    cols = [c for c in df.columns if str(c).startswith("acc_")]
    mapping = {}
    for c in cols:
        key = str(c)[4:]
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(SUPERCLASSES):
                mapping[SUPERCLASSES[idx]] = df[c]
        else:
            mapping[key] = df[c]
    return mapping


def _safe_save(fig, name: str):
    out = OUT / name
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
    finally:
        plt.close(fig)


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

# -- Global accuracy vs round --------------------------------------------
if dfs:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, df in dfs.items():
        if df is None or df.empty or "client_id" not in df.columns:
            continue
        g = df.loc[df["client_id"].astype(str) == "GLOBAL"].copy()
        if g.empty or "round" not in g.columns or "accuracy" not in g.columns:
            continue
        g = g.sort_values("round")
        ax.plot(g["round"], g["accuracy"], label=f"Global ({name})")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy"); ax.set_title("Global Accuracy per Round")
    ax.legend(); ax.grid(True); fig.tight_layout()
    _safe_save(fig, "global_accuracy.png")


# -- Client accuracies vs round ------------------------------------------
for name, df in dfs.items():
    if df is None or df.empty or "client_id" not in df.columns:
        continue
    sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
    if sub.empty or "round" not in sub.columns or "accuracy" not in sub.columns:
        continue
    sub = sub.sort_values(["client_id", "round"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cid, grp in sub.groupby("client_id"):
        if grp.empty:
            continue
        ax.plot(grp["round"], grp["accuracy"], alpha=0.6, label=f"Client {cid}")
    # overlay global if present
    g = df[df["client_id"].astype(str) == "GLOBAL"].copy()
    if not g.empty and "round" in g.columns and "accuracy" in g.columns:
        g = g.sort_values("round")
        ax.plot(g["round"], g["accuracy"], linewidth=3, linestyle="--", label="GLOBAL")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Client vs Global Accuracy — {name}")
    ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
    _safe_save(fig, f"clients_vs_global_{name}.png")


# -- Total wall-time per round (sum across clients) ----------------------
for name, df in dfs.items():
    if df is None or df.empty or "wall_time_sec" not in df.columns or "round" not in df.columns:
        continue
    sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
    if sub.empty:
        continue
    agg = (sub.groupby("round", as_index=False)
              .agg(wall_time_sec=("wall_time_sec", "sum"))
              .sort_values(by="round"))
    if agg.empty:
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(agg["round"], agg["wall_time_sec"])
    ax.set_xlabel("Round"); ax.set_ylabel("Total client wall-time (s)")
    ax.set_title(f"Total Wall-Time per Round — {name}")
    ax.grid(True); fig.tight_layout()
    _safe_save(fig, f"walltime_{name}.png")


# -- Per-client trajectories (frozen vs non-frozen, one figure) ----------
if all(k in dfs for k in ("frozen", "non_frozen")):
    nf = dfs.get("non_frozen"); fr = dfs.get("frozen")
    if nf is not None and fr is not None and not nf.empty and not fr.empty:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cids = set(nf.query("client_id!='GLOBAL'")["client_id"]).union(
               set(fr.query("client_id!='GLOBAL'")["client_id"]))
        for cid in sorted(cids):
            for run_name, dfi, style in [("non_frozen", nf, "-"), ("frozen", fr, "--")]:
                sub_df = dfi.loc[dfi["client_id"] == cid].copy()
                if sub_df.empty or "round" not in sub_df.columns or "accuracy" not in sub_df.columns:
                    continue
                sub_df.sort_values(by="round", inplace=True)
                lbl = f"Client {cid} ({'unfrozen' if run_name=='non_frozen' else 'frozen'})"
                ax.plot(sub_df["round"], sub_df["accuracy"], linestyle=style, label=lbl, alpha=0.8)
        ax.set_xlabel("Round"); ax.set_ylabel("Local Accuracy")
        ax.set_title("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)")
        ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
        _safe_save(fig, "per_client_trajectories.png")


# -- Per-class accuracy over rounds (line plot) --------------------------
if percls:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if "non_frozen" in percls and not percls["non_frozen"].empty:
        nf = percls["non_frozen"].sort_values(by="round")
        for label, series in perclass_columns(nf).items():
            try:
                ax.plot(nf["round"], series, label=f"{label} (unfrozen)")
            except Exception:
                pass
    if "frozen" in percls and not percls["frozen"].empty:
        fr = percls["frozen"].sort_values(by="round")
        for label, series in perclass_columns(fr).items():
            try:
                ax.plot(fr["round"], series, linestyle="--", label=f"{label} (frozen)")
            except Exception:
                pass
    ax.set_xlabel("Round"); ax.set_ylabel("Per-class Accuracy")
    ax.set_title("Per-class Accuracy per Round")
    ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
    _safe_save(fig, "perclass_over_rounds.png")


# -- Final-round per-class accuracy (bar) --------------------------------
if percls:
    labels = SUPERCLASSES
    x = np.arange(len(labels)); width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)
    drew_any = False
    if "non_frozen" in percls and not percls["non_frozen"].empty:
        nf_last = percls["non_frozen"].sort_values(by="round").tail(1)
        nf_map  = perclass_columns(nf_last)
        y_nf    = [float(nf_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
        ax.bar(x - width/2, y_nf, width, label="Unfrozen"); drew_any = True
    if "frozen" in percls and not percls["frozen"].empty:
        fr_last = percls["frozen"].sort_values(by="round").tail(1)
        fr_map  = perclass_columns(fr_last)
        y_fr    = [float(fr_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
        ax.bar(x + width/2, y_fr, width, label="Frozen"); drew_any = True
    if drew_any:
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy"); ax.set_title("Final-Round Per-class Accuracy")
        ax.legend(); fig.tight_layout()
        _safe_save(fig, "perclass_final_bar.png")
    else:
        plt.close(fig)


# -- Confusion heatmaps (last round of each run) -------------------------
for name, dfc in conf_long.items():
    if dfc is None or dfc.empty:
        continue
    if "round" not in dfc.columns or "true" not in dfc.columns or "pred" not in dfc.columns or "count" not in dfc.columns:
        continue
    r = int(dfc["round"].max())
    sub = dfc[dfc["round"] == r].copy()
    if sub.empty:
        continue
    n = len(SUPERCLASSES)
    # pivot and coerce to square matrix (0..n-1)
    try:
        piv = sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
        # force integer-like indices/cols to 0..n-1
        piv.index = pd.to_numeric(piv.index, errors="coerce")
        piv.columns = pd.to_numeric(piv.columns, errors="coerce")
        cm = (piv.reindex(index=range(n), columns=range(n), fill_value=0)
                 .to_numpy(dtype=float))
    except Exception as e:
        print(f"Confusion pivot failed for {name}: {e}")
        continue

    with np.errstate(invalid="ignore", divide="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cm / np.maximum(row_sum, 1), nan=0.0, posinf=0.0, neginf=0.0)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cmn, aspect="auto", vmin=0, vmax=1.0)
    ax.set_title(f"Confusion Matrix — {name} (round {r})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n)); ax.set_xticklabels(SUPERCLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(SUPERCLASSES)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Row-normalized")
    # annotate
    for i in range(n):
        for j in range(n):
            val = cmn[i, j] * 100.0
            ax.text(j, i, f"{val:0.2f}", ha="center", va="center",
                    color=("white" if val >= 50 else "black"), fontsize=9)
    fig.tight_layout()
    _safe_save(fig, f"confusion_{name}_r{r:02d}.png")


# -- Global accuracy by phase (within each run) --------------------------
for name, df in dfs.items():
    if df is None or df.empty or "phase" not in df.columns:
        continue
    g = df[df["client_id"].astype(str) == "GLOBAL"].copy()
    if g.empty or "round" not in g.columns or "accuracy" not in g.columns:
        continue
    pre  = g[g["phase"].astype(str).eq(PHASE_DISABLED)].sort_values(by="round")
    post = g[g["phase"].astype(str).eq(PHASE_ENABLED)].sort_values(by="round")
    if pre.empty and post.empty:
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not pre.empty:  ax.plot(pre["round"], pre["accuracy"], label=f"{name} (no-CV)")
    if not post.empty: ax.plot(post["round"], post["accuracy"], label=f"{name} (post-CV)")
    ax.set_xlabel("Round"); ax.set_ylabel("Global Accuracy")
    ax.set_title(f"Global Accuracy by Phase — {name}")
    ax.legend(); ax.grid(True); fig.tight_layout()
    _safe_save(fig, f"global_by_phase_{name}.png")


# -- Per-client accuracy by phase (overlay pre/post) ---------------------
for name, df in dfs.items():
    if df is None or df.empty or "phase" not in df.columns or "client_id" not in df.columns:
        continue
    sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
    if sub.empty or "round" not in sub.columns or "accuracy" not in sub.columns:
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111)
    drew_any = False
    for phase, style in [(PHASE_DISABLED, "-"), (PHASE_ENABLED, "--")]:
        grp = sub[sub["phase"].astype(str).eq(phase)]
        for cid, g in grp.groupby("client_id"):
            g = g.sort_values(by="round")
            if g.empty: 
                continue
            ax.plot(g["round"], g["accuracy"], linestyle=style, alpha=0.7,
                    label=f"Client {cid} ({name}, {phase})")
            drew_any = True
    if drew_any:
        ax.set_xlabel("Round"); ax.set_ylabel("Local Accuracy")
        ax.set_title(f"Per-Client Accuracy by Phase — {name}")
        ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
        _safe_save(fig, f"clients_by_phase_{name}.png")
    else:
        plt.close(fig)

print(f"Saved plots (where possible) to: {OUT.resolve()}")


# =============================================================================
# The remainder provides unified evaluation utilities used by Centralized.py.
# (Kept as in your original file; lightly hardened for imports.)
# =============================================================================

# ---------
# ## 15) Unified Evaluation + ROC-AUC (from notebook)
# ---------
from __future__ import annotations as _ann_eval  # avoid shadowing

import time as _time, re as _re
import torch as _torch
from sklearn.metrics import (classification_report as _clsrep, accuracy_score as _acc,
                             f1_score as _f1, precision_score as _prec, recall_score as _rec,
                             roc_auc_score as _rocauc, confusion_matrix as _sk_cm)

# Avoid circular imports at module load time
try:
    from . import config as CFG
    from .utils import torch_loader_kwargs as _torch_loader_kwargs
except Exception:
    # Fallback if imported as a standalone script (optional)
    from src import config as CFG  # type: ignore
    from src.utils import torch_loader_kwargs as _torch_loader_kwargs  # type: ignore


class TorchAdapter:
    def __init__(self, model, label_encoder, is_binary: bool, batch=32, device=None):
        self.model = model.to(device or _torch.device("cpu")).eval()
        self.le = label_encoder; self.is_binary = is_binary
        self.batch = int(batch); self.device = device or _torch.device("cpu")
        self.classes_ = getattr(self.le, "classes_", None)

    @_torch.no_grad()
    def _loader(self, paths):
        from torch.utils.data import Dataset, DataLoader
        from .data_loader import load_waveform_np
        class _PredictOnly(Dataset):
            def __init__(self, p):
                self.paths=list(p)
                self.T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                return _torch.from_numpy(load_waveform_np(self.paths[i], T=self.T, factor=CFG.DOWNSAMPLE_FACTOR))
        return DataLoader(_PredictOnly(paths), **_torch_loader_kwargs(False, self.batch, self.device.type))

    @_torch.no_grad()
    def predict(self, paths):
        preds=[]
        for xb in self._loader(paths):
            xb=xb.to(self.device); logits=self.model(xb)
            y_idx=((_torch.sigmoid(logits.view(-1))>=0.5).long().cpu().numpy()
                   if self.is_binary else logits.argmax(1).cpu().numpy())
            preds.extend(self.le.inverse_transform(y_idx))
        return np.asarray(preds)

    @_torch.no_grad()
    def predict_proba(self, paths):
        out=[]
        for xb in self._loader(paths):
            xb=xb.to(self.device); logits=self.model(xb)
            if self.is_binary:
                p1=_torch.sigmoid(logits.view(-1)).unsqueeze(1); p0=1.0-p1; p=_torch.cat([p0,p1], dim=1)
            else:
                p=_torch.softmax(logits, dim=1)
            out.append(p.cpu().numpy().astype("float32"))
        return np.vstack(out)


def evaluate_models(models_dict, paths, y_true, label_encoder, is_binary, device):
    rows=[]
    for name, adapter in models_dict.items():
        y_pred = adapter.predict(paths)
        try:
            print(f"\n=== {name} ===")
            print(_clsrep(y_true, y_pred, digits=3, zero_division=0))
        except Exception:
            pass
        row = {
            "model": name,
            "accuracy": _acc(y_true, y_pred),
            "precision_macro": _prec(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": _rec(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": _f1(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": _f1(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc_ovr_macro": np.nan
        }
        try:
            classes = np.unique(y_true)
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=classes)
            scores = adapter.predict_proba(paths)
            est = np.asarray(adapter.classes_)
            pos = {c:i for i,c in enumerate(est)}
            idx = [pos[c] for c in classes]
            scores = scores[:, idx]
            row["roc_auc_ovr_macro"] = float(_rocauc(y_bin, scores, average="macro", multi_class="ovr"))
        except Exception:
            pass
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    return df


# ---------
# ## 16) Confusion Matrix (from notebook)
# ---------
def _slug(txt): return _re.sub(r"[^a-z0-9]+", "_", str(txt).lower()).strip("_")

def plot_confusion(y_true, y_pred, title="Confusion Matrix", classes=None, save_path=None, collapse_to_3=False):
    if collapse_to_3:
        def _map3(a):
            a=str(a)
            if a=="MI": return "MI"
            if a=="NORM": return "NORM"
            return "OTHER"
        y_true = np.vectorize(_map3)(np.array(y_true)); y_pred = np.vectorize(_map3)(np.array(y_pred))
        classes = ["MI","NORM","OTHER"]
    else:
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        classes = classes or sorted(list(set(y_true)|set(y_pred)))
    cm = _sk_cm(y_true, y_pred, labels=classes)
    cm_pct = cm.astype(float); row_sum = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_pct, np.maximum(row_sum, 1), where=row_sum>0) * 100.0
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    fig.colorbar(im, ax=ax).set_label("%")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            ax.text(j, i, f"{val:0.2f}", ha="center", va="center",
                    color=("white" if val >= 50 else "black"), fontsize=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------
# ## 17) Learning Curves (from notebook)
# ---------
def plot_learning_curves(histories: dict[str, dict], out_dir: str | Path | None = None):
    if not histories:
        print("No histories to plot.")
        return
    out_dir = Path(out_dir or (Path(CFG.ART_DIR) / "figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, hist in histories.items():
        ep = range(1, len(hist.get('loss', [])) + 1)
        plt.figure(figsize=(7,4))
        plt.plot(ep, hist.get('loss', []), marker='o', linewidth=1, label='Training Loss')
        plt.plot(ep, hist.get('val_loss', []), marker='o', linewidth=1, label='Validation Loss')
        plt.title(f'Loss — {name}'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
        fp1 = out_dir / f"loss_curve_{_slug(name)}.png"
        plt.tight_layout(); plt.savefig(fp1, dpi=300, bbox_inches='tight'); print('Saved:', fp1); plt.close()

        acc = hist.get('accuracy', []); val_acc = hist.get('val_accuracy', [])
        if len(acc):
            plt.figure(figsize=(7,4))
            plt.plot(ep, acc, marker='o', linewidth=1, label='Training Acc')
            plt.plot(ep, val_acc, marker='o', linewidth=1, label='Validation Acc')
            plt.title(f'Accuracy — {name}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(alpha=0.3)
            fp2 = out_dir / f"accuracy_curve_{_slug(name)}.png"
            plt.tight_layout(); plt.savefig(fp2, dpi=300, bbox_inches='tight'); print('Saved:', fp2); plt.close()


# ---------
# ## 17c) Overfitting quick check (from notebook)
# ---------
def overfitting_check(histories: dict[str, dict], cv_all_df: pd.DataFrame | None = None,
                      min_train=0.80, max_val=0.70, min_gap=0.12):
    rows = []
    for name, hist in histories.items():
        tr = np.array(hist.get("accuracy", []), float)
        va = np.array(hist.get("val_accuracy", []), float)
        if tr.size == 0 or va.size == 0:
            rows.append({"model": name, "OVERFITTING": "no history"})
            continue
        gap = float(tr[-1] - va[-1])
        flag = (tr[-1] >= min_train) and (va[-1] <= max_val) and (gap >= min_gap)
        rows.append({"model": name, "train_acc_last": float(tr[-1]), "val_acc_last": float(va[-1]),
                     "gap": gap, "OVERFITTING": "YES" if flag else "NO/unclear"})
    df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    if cv_all_df is not None and not cv_all_df.empty:
        g = cv_all_df.groupby("model")["accuracy"].agg(cv_mean="mean", cv_std="std")
        last_tr = {k: (np.array(v.get("accuracy", []), float)[-1] if v.get("accuracy") else np.nan)
                   for k, v in histories.items()}
        g["train_acc_last"] = [last_tr.get(m if "(PyTorch)" in m else f"{m} (PyTorch)", np.nan) for m in g.index]
        g["train_minus_cv"] = g["train_acc_last"] - g["cv_mean"]
        return df, g
    return df, None


# ---------
# ## 18a) Fairness / Group Metrics (from notebook)
# ---------
from sklearn.metrics import f1_score as _f1_score

def fairness_macro_f1_by_groups(best_adapter, y_true, test_index, meta_df: pd.DataFrame):
    paths = meta_df.loc[test_index, "record_path"].astype(str).values
    y_pred = np.asarray(best_adapter.predict(paths))
    y_true = np.asarray(list(y_true))
    meta_test = meta_df.loc[test_index].copy()

    if "sex_norm" not in meta_test.columns and "sex" in meta_test.columns:
        def _sex_norm(series: pd.Series) -> pd.Series:
            out = pd.Series("unknown", index=series.index, dtype="string")
            s_num = pd.to_numeric(series, errors="coerce")
            out.loc[s_num == 1] = "male"; out.loc[s_num == 0] = "female"
            s = series.astype("string").str.strip().str.lower()
            out.loc[s.str.startswith("m", na=False)] = "male"
            out.loc[s.str.startswith("f", na=False)] = "female"
            out.loc[s.isin({"male","man"})] = "male"
            out.loc[s.isin({"female","woman"})] = "female"
            return out
        meta_test["sex_norm"] = _sex_norm(meta_test["sex"])

    meta_test["age_num"] = pd.to_numeric(meta_test.get("age", np.nan), errors="coerce")
    meta_test["age_bin"] = pd.cut(meta_test["age_num"], bins=[0,40,60,200], labels=["<=40","41-60","61+"])

    rows = []
    def _macro_f1_mask(mask):
        a, b = y_true[mask], y_pred[mask]
        if len(np.unique(a)) < 2:
            return np.nan
        return _f1_score(a, b, average="macro", zero_division=0)

    for grp_name, series in [("sex_norm", meta_test["sex_norm"]), ("age_bin", meta_test["age_bin"])]:
        for g, idxs in series.groupby(series).groups.items():
            mask = meta_test.index.isin(idxs)
            rows.append({"group": grp_name, "value": str(g), "n": int(mask.sum()), "macro_F1": _macro_f1_mask(mask)})
    fair_df = pd.DataFrame(rows).sort_values(["group","value"])
    return fair_df


# ---------
# ## 18b) Interpretation — Gradient×Input saliency (from notebook)
# ---------
def saliency_grad_times_input(model: _torch.nn.Module, path: str, device=None):
    device = device or _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    from .data_loader import load_waveform_np
    T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
    x = load_waveform_np(path, T=T, factor=CFG.DOWNSAMPLE_FACTOR)
    xt = _torch.from_numpy(x).unsqueeze(0).to(device)
    xt.requires_grad_(True)

    with _torch.enable_grad():
        logits = model(xt)
        if logits.ndim == 2 and logits.size(1) > 1:
            target = logits[0, logits.argmax(1).item()]
        else:
            target = logits.view(-1)[0]
        model.zero_grad(set_to_none=True)
        if xt.grad is not None: xt.grad.zero_()
        target.backward()
        grad = xt.grad.detach(); xt_det = xt.detach()
        sal = (grad * xt_det).abs().squeeze(0).cpu().numpy()
        sal_mean = sal.mean(axis=1)
    return x, sal_mean

def plot_saliency_overlay(model, sample_paths: list[str], out_dir: str | Path | None = None, device=None):
    out_dir = Path(out_dir or (Path(CFG.ART_DIR) / "figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sample_paths:
        x, s = saliency_grad_times_input(model, p, device=device)
        fig, ax1 = plt.subplots(figsize=(10, 3.0))
        ax1.plot(x[:, 0], linewidth=0.8); ax1.set_title("ECG (Lead I) with saliency overlay"); ax1.set_ylabel("mV"); ax1.grid(alpha=0.3)
        ax2 = ax1.twinx(); ax2.plot(s, linewidth=0.8, alpha=0.85); ax2.set_ylabel("saliency")
        fp = out_dir / f"saliency_{Path(p).name.replace('/', '_')}.png"
        plt.tight_layout(); plt.savefig(fp, dpi=220, bbox_inches="tight"); print("Saved:", fp); plt.close()

# ---------
# ## 18c) Efficiency — params & inference time (from notebook)
# ---------
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def benchmark_inference(adapter, paths, device=None, warmup=1):
    device = device or _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = adapter.model.to(device).eval()
    if warmup:
        _ = adapter.predict(paths[:min(8, len(paths))])
    t0 = _time.time()
    _ = adapter.predict(paths)
    return _time.time() - t0

def summarize_efficiency(adapters: dict[str, object], test_paths: list[str]):
    rows = []
    for name, adapter in adapters.items():
        try:
            secs = benchmark_inference(adapter, test_paths)
            params = count_params(adapter.model)
            rows.append({"model": name, "params": int(params), "test_infer_sec": float(secs),
                         "samples": int(len(test_paths)),
                         "samples_per_sec": float(len(test_paths))/max(secs, 1e-6)})
        except Exception:
            rows.append({"model": name, "params": np.nan, "test_infer_sec": np.nan,
                         "samples": len(test_paths), "samples_per_sec": np.nan})
    return pd.DataFrame(rows).sort_values("samples_per_sec", ascending=False)
