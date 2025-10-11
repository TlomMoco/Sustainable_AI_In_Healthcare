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


# --- Load CSVs with numeric columns coerced to numbers (NaN if invalid) ---
def load_numeric_csv(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    for col in ["round","accuracy","loss","wall_time_sec","trainable_params","true","pred","count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# --- Load all dataframes (calling load_numeric_csv) ---------------------
dfs      = {name: load_numeric_csv(p) for name, p in FILES.items()}
percls   = {name: load_numeric_csv(p) for name, p in PERCLASS.items()}
conf_long= {name: load_numeric_csv(p) for name, p in CONFUSION.items()}

dfs      = {k:v for k,v in dfs.items() if v is not None}
percls   = {k:v for k,v in percls.items() if v is not None}
conf_long= {k:v for k,v in conf_long.items() if v is not None}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def perclass_columns(df):
    """Return mapping label-> Series, accepting acc_<name> or acc_<index>."""
    cols = [c for c in df.columns if c.startswith("acc_")]
    mapping = {}
    for c in cols:
        key = c[4:]
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(SUPERCLASSES):
                mapping[SUPERCLASSES[idx]] = df[c]
        else:
            mapping[key] = df[c]
    return mapping


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

# -- Global accuracy vs round ----------------------------------
plt.figure()
for name, df in dfs.items():
    g = df[df["client_id"] == "GLOBAL"].sort_values("round")
    if len(g):
        plt.plot(g["round"], g["accuracy"], label=f"Global ({name})")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("Global Accuracy per Round")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(OUT / "global_accuracy.png", dpi=150); plt.close()



# -- Client accuracies vs round ---------------------------------
for name, df in dfs.items():
    plt.figure()
    sub = df[df["client_id"] != "GLOBAL"].sort_values(["client_id", "round"])
    for cid, grp in sub.groupby("client_id"):
        plt.plot(grp["round"], grp["accuracy"], alpha=0.6, label=f"Client {cid}")
    g = df[df["client_id"] == "GLOBAL"].sort_values("round")
    if len(g):
        plt.plot(g["round"], g["accuracy"], linewidth=3, linestyle="--", label="GLOBAL")
    plt.xlabel("Round"); plt.ylabel("Accuracy")
    plt.title(f"Client vs Global Accuracy — {name}")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"clients_vs_global_{name}.png", dpi=150); plt.close()



# -- Wall-time per round (total across clients) -------------------
for name, df in dfs.items():
    if "wall_time_sec" not in df.columns:
        continue
    sub = df[df["client_id"] != "GLOBAL"]
    agg = (
        sub.groupby("round", as_index=False)
           .agg(wall_time_sec=("wall_time_sec", "sum"))
           .sort_values(by="round")
    )
    plt.figure()
    plt.plot(agg["round"], agg["wall_time_sec"])
    plt.xlabel("Round"); plt.ylabel("Total client wall-time (s)")
    plt.title(f"Total Wall-Time per Round — {name}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"walltime_{name}.png", dpi=150); plt.close()



# -- Per-client trajectories (frozen vs non-frozen, one figure) ---
if all(k in dfs for k in ("frozen", "non_frozen")):
    plt.figure()
    cids = set(dfs["non_frozen"].query("client_id!='GLOBAL'")["client_id"]).union(
           set(dfs["frozen"].query("client_id!='GLOBAL'")["client_id"]))
    for cid in sorted(cids):
        for name, style in [("non_frozen","-"), ("frozen","--")]:
            sub_df = dfs[name].loc[dfs[name]["client_id"] == cid].copy()  # DataFrame
            sub_df.sort_values(by="round", inplace=True)                  # sort DataFrame
            if not sub_df.empty:
                lbl = f"Client {cid} ({'unfrozen' if name=='non_frozen' else 'frozen'})"
                plt.plot(sub_df["round"], sub_df["accuracy"], linestyle=style, label=lbl, alpha=0.8)
    plt.xlabel("Round"); plt.ylabel("Local Accuracy")
    plt.title("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / "per_client_trajectories.png", dpi=150); plt.close()



# -- Per-class accuracy over rounds (line plot) --
if percls:
    plt.figure()
    if "non_frozen" in percls:
        nf = percls["non_frozen"].sort_values(by="round")
        for label, series in perclass_columns(nf).items():
            plt.plot(nf["round"], series, label=f"{label} (unfrozen)")
    if "frozen" in percls:
        fr = percls["frozen"].sort_values(by="round")
        for label, series in perclass_columns(fr).items():
            plt.plot(fr["round"], series, linestyle="--", label=f"{label} (frozen)")
    plt.xlabel("Round"); plt.ylabel("Per-class Accuracy")
    plt.title("Per-class Accuracy per Round")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / "perclass_over_rounds.png", dpi=150); plt.close()

# -- Final-round per-class accuracy (bar) --
if percls:
    labels = SUPERCLASSES
    x = np.arange(len(labels)); width = 0.35
    plt.figure()
    if "non_frozen" in percls:
        nf_last = percls["non_frozen"].sort_values(by="round").tail(1)
        nf_map  = perclass_columns(nf_last)
        y_nf    = [float(nf_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
        plt.bar(x - width/2, y_nf, width, label="Unfrozen")
    if "frozen" in percls:
        fr_last = percls["frozen"].sort_values(by="round").tail(1)
        fr_map  = perclass_columns(fr_last)
        y_fr    = [float(fr_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
        plt.bar(x + width/2, y_fr, width, label="Frozen")
    plt.xticks(x, labels); plt.ylabel("Accuracy")
    plt.title("Final-Round Per-class Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "perclass_final_bar.png", dpi=150); plt.close()



# -- Confusion heatmaps (last round of each run) ------------------
for name, dfc in conf_long.items():
    if dfc is None or dfc.empty:
        continue
    r = int(dfc["round"].max())
    sub = dfc[dfc["round"] == r]
    n = len(SUPERCLASSES)
    cm = (sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
              .reindex(index=range(n), columns=range(n), fill_value=0)
              .to_numpy(dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    plt.figure(figsize=(6,5))
    plt.imshow(cmn, aspect="auto")
    plt.title(f"Confusion Matrix — {name} (round {r})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(n), SUPERCLASSES, rotation=45, ha="right")
    plt.yticks(range(n), SUPERCLASSES)
    plt.colorbar(label="Row-normalized")
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_{name}_r{r:02d}.png", dpi=150); plt.close()



# -- Global accuracy by phase (within each run) ---------------------------
for name, df in dfs.items():
    if "phase" not in df.columns:
        continue
    g = df[df["client_id"] == "GLOBAL"].copy()
    pre  = g[g["phase"].eq(PHASE_DISABLED)].sort_values(by="round")
    post = g[g["phase"].eq(PHASE_ENABLED)].sort_values(by="round")
    if pre.empty and post.empty:
        continue
    plt.figure()
    if not pre.empty:  plt.plot(pre["round"], pre["accuracy"], label=f"{name} (no-CV)")
    if not post.empty: plt.plot(post["round"], post["accuracy"], label=f"{name} (post-CV)")
    plt.xlabel("Round"); plt.ylabel("Global Accuracy")
    plt.title(f"Global Accuracy by Phase — {name}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"global_by_phase_{name}.png", dpi=150); plt.close()



# -- Per-client accuracy by phase (overlay pre/post) ---------------------
for name, df in dfs.items():
    if "phase" not in df.columns:
        continue
    plt.figure()
    sub = df[df["client_id"] != "GLOBAL"].copy()
    for phase, style in [(PHASE_DISABLED,"-"), (PHASE_ENABLED,"--")]:
        grp = sub[sub["phase"].eq(phase)]
        for cid, g in grp.groupby("client_id"):
            g = g.sort_values(by="round")
            plt.plot(g["round"], g["accuracy"], linestyle=style, alpha=0.7,
                     label=f"Client {cid} ({name}, {phase})")
    plt.xlabel("Round"); plt.ylabel("Local Accuracy")
    plt.title(f"Per-Client Accuracy by Phase — {name}")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"clients_by_phase_{name}.png", dpi=150); plt.close()


print(f"Saved plots to: {OUT.resolve()}")









# ---------
# ## 15) Unified Evaluation + ROC-AUC (from notebook)
# ---------
from __future__ import annotations
import time, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from . import config as CFG
from .utils import torch_loader_kwargs

class TorchAdapter:
    def __init__(self, model, label_encoder, is_binary: bool, batch=32, device=None):
        self.model = model.to(device or torch.device("cpu")).eval()
        self.le = label_encoder; self.is_binary = is_binary
        self.batch = int(batch); self.device = device or torch.device("cpu")
        self.classes_ = getattr(self.le, "classes_", None)
    @torch.no_grad()
    def _loader(self, paths):
        from torch.utils.data import Dataset, DataLoader
        from .data_loader import load_waveform_np
        class _PredictOnly(Dataset):
            def __init__(self, p):
                self.paths=list(p)
                self.T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                return torch.from_numpy(load_waveform_np(self.paths[i], T=self.T, factor=CFG.DOWNSAMPLE_FACTOR))
        return DataLoader(_PredictOnly(paths), **torch_loader_kwargs(False, self.batch, self.device.type))
    @torch.no_grad()
    def predict(self, paths):
        preds=[]
        for xb in self._loader(paths):
            xb=xb.to(self.device); logits=self.model(xb)
            y_idx=((torch.sigmoid(logits.view(-1))>=0.5).long().cpu().numpy()
                   if self.is_binary else logits.argmax(1).cpu().numpy())
            preds.extend(self.le.inverse_transform(y_idx))
        return np.asarray(preds)
    @torch.no_grad()
    def predict_proba(self, paths):
        out=[]
        for xb in self._loader(paths):
            xb=xb.to(self.device); logits=self.model(xb)
            if self.is_binary:
                p1=torch.sigmoid(logits.view(-1)).unsqueeze(1); p0=1.0-p1; p=torch.cat([p0,p1], dim=1)
            else:
                p=torch.softmax(logits, dim=1)
            out.append(p.cpu().numpy().astype("float32"))
        return np.vstack(out)

def evaluate_models(models_dict, paths, y_true, label_encoder, is_binary, device):
    rows=[]
    for name, adapter in models_dict.items():
        y_pred = adapter.predict(paths)
        try:
            print(f"\n=== {name} ===")
            print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        except Exception:
            pass
        row = {
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
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
            row["roc_auc_ovr_macro"] = float(roc_auc_score(y_bin, scores, average="macro", multi_class="ovr"))
        except Exception:
            pass
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    return df

# ---------
# ## 16) Confusion Matrix (from notebook)
# ---------
def _slug(txt): return re.sub(r"[^a-z0-9]+", "_", str(txt).lower()).strip("_")

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
    cm = confusion_matrix(y_true, y_pred, labels=classes)
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
        print("No histories to plot."); 
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
            rows.append({"model": name, "OVERFITTING": "no history"}); 
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
from sklearn.metrics import f1_score
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
        return f1_score(a, b, average="macro", zero_division=0)

    for grp_name, series in [("sex_norm", meta_test["sex_norm"]), ("age_bin", meta_test["age_bin"])]:
        for g, idxs in series.groupby(series).groups.items():
            mask = meta_test.index.isin(idxs)
            rows.append({"group": grp_name, "value": str(g), "n": int(mask.sum()), "macro_F1": _macro_f1_mask(mask)})
    fair_df = pd.DataFrame(rows).sort_values(["group","value"])
    return fair_df

# ---------
# ## 18b) Interpretation — Gradient×Input saliency (from notebook)
# ---------
def saliency_grad_times_input(model: torch.nn.Module, path: str, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    from .data_loader import load_waveform_np
    T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
    x = load_waveform_np(path, T=T, factor=CFG.DOWNSAMPLE_FACTOR)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    xt.requires_grad_(True)

    with torch.enable_grad():
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = adapter.model.to(device).eval()
    if warmup:
        _ = adapter.predict(paths[:min(8, len(paths))])
    t0 = time.time()
    _ = adapter.predict(paths)
    return time.time() - t0

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
