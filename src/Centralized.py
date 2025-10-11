from __future__ import annotations
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_loader import load_metadata, map_superclasses, filter_single_label, stratified_patient_split, make_feature_table
from src.models import create_logistic_baseline
from src.config import SEED


if __name__ == "__main__":
    ptb = load_metadata()
    df = filter_single_label(map_superclasses(ptb))
    train_df, test_df = stratified_patient_split(df, test_size=0.2, seed=SEED)


    # ===== Baseline 1: ML on simple features =====
    Xtr, ytr, classes = make_feature_table(train_df)
    Xte, yte, _ = make_feature_table(test_df)

    logreg = create_logistic_baseline().fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)

    for name, clf in {"LogReg": logreg, "RF": rf}.items():
        ypred = clf.predict(Xte)
        print(f"\n== {name} ==")
        print(classification_report(yte, ypred, target_names=classes, digits=3))


















# ---------
# ## 12b) Baseline & 13) Train Selected Models (centralized) (from notebook)
# ---------
from __future__ import annotations
import numpy as np, pandas as pd, torch, math, time
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from . import config as CFG
from .utils import set_seed, pick_device, torch_loader_kwargs, log, sanitize_mps_env
from .data_loader import make_feature_table, train_test_split
from .data_preprocessing import ECGDataset, make_label_encoder, make_train_val_loaders
from .models import create_model
from .results_visualization import TorchAdapter, evaluate_models, plot_confusion, plot_learning_curves

def main():
    set_seed(CFG.SEED); sanitize_mps_env()
    # Build features + minimal meta table
    feature_df, features_df = make_feature_table(save_csv=True)
    train_mask, test_mask = train_test_split(features_df)

    # Dummy baseline on engineered features
    X_tr = feature_df.loc[feature_df.index.intersection(features_df.index[train_mask])].drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
    y_tr_b = feature_df.loc[X_tr.index, "label"].astype(str)
    X_te = feature_df.loc[feature_df.index.intersection(features_df.index[test_mask])].drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
    y_te_b = feature_df.loc[X_te.index, "label"].astype(str)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("clf", DummyClassifier(strategy="most_frequent"))])
    pipe.fit(X_tr, y_tr_b)
    y_pred_dummy = pipe.predict(X_te)
    print("\n[Baseline Dummy]"); print(classification_report(y_te_b, y_pred_dummy, digits=3, zero_division=0))

    # Deep dataset splits
    train_paths = features_df.loc[train_mask, "record_path"].astype(str).values
    test_paths  = features_df.loc[test_mask,  "record_path"].astype(str).values
    y_tr_series = features_df.loc[train_mask, "label"].astype(str)
    y_te_series = features_df.loc[test_mask,  "label"].astype(str)

    le = make_label_encoder(y_tr_series, y_te_series)
    y_tr = le.transform(y_tr_series.values)
    y_te = y_te_series.values
    n_classes = len(le.classes_)
    is_binary = (n_classes == 2)

    device = pick_device(); log(f"[Torch] device={device}")
    train_loader, val_loader, tr_pos, va_pos = make_train_val_loaders(train_paths, y_tr, device=device)

    # Train a couple selected models (respects CFG flags)
    histories = {}
    deep_models_raw = {}
    def _criterion():
        if is_binary: return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _eval_epoch(model, loader):
        model.eval(); crit = _criterion()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits.view(-1), yb.float()) if is_binary else crit(logits, yb)
                loss_sum += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long() if is_binary else logits.argmax(1)
                correct += (preds == yb).sum().item(); total += xb.size(0)
        return (loss_sum/total if total else math.nan), (correct/total if total else math.nan)

    def train_model_torch(name, epochs=CFG.EPOCHS):
        model = create_model(name, n_classes, binary=is_binary).to(device)
        base_lr = CFG.RECURRENT_LR if name.lower() in {"rnn","lstm","ann"} else CFG.BASE_LR
        opt = optim.Adam(model.parameters(), lr=base_lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)
        crit = _criterion()

        hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_state = None; best_val = float("inf")
        for ep in range(1, epochs+1):
            model.train(); ep_loss, ep_correct, ep_total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits.view(-1), yb.float()) if is_binary else crit(logits, yb)
                loss.backward()
                if CFG.GRAD_CLIP_NORM: nn.utils.clip_grad_norm_(model.parameters(), float(CFG.GRAD_CLIP_NORM))
                opt.step()
                ep_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long() if is_binary else logits.argmax(1)
                ep_correct += (preds == yb).sum().item(); ep_total += xb.size(0)
            tr_loss = ep_loss/max(1, ep_total); tr_acc = ep_correct/max(1, ep_total)
            va_loss, va_acc = _eval_epoch(model, val_loader)
            sch.step(va_loss)
            hist["loss"].append(tr_loss); hist["accuracy"].append(tr_acc)
            hist["val_loss"].append(va_loss); hist["val_accuracy"].append(va_acc)
            print(f"[{name}] epoch {ep:02d}/{epochs}  loss={tr_loss:.4f} acc={tr_acc:.4f}  val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        return model, hist

    for name, flag in [("cnn", CFG.RUN_TORCH_CNN), ("rnn", CFG.RUN_TORCH_RNN),
                       ("lstm", CFG.RUN_TORCH_LSTM), ("ann", CFG.RUN_TORCH_ANN)]:
        if not flag: 
            print(f"[Train] Skipped {name.upper()} (flag off)")
            continue
        print(f"\n[Train] === {name.upper()} ===")
        m, h = train_model_torch(name)
        deep_models_raw[f"{name.upper()} (PyTorch)"] = m
        histories[f"{name.upper()} (PyTorch)"] = h

    # Evaluate on TEST
    adapters = {name: TorchAdapter(m, le, is_binary, batch=CFG.BATCH_SIZE, device=device)
                for name, m in deep_models_raw.items()}
    unified_df = evaluate_models(adapters, test_paths, y_te, le, is_binary, device)
    print("\nUnified evaluation:"); print(unified_df)

    # Best model confusion matrix + learning curves
    if not unified_df.empty:
        best_name = unified_df.iloc[0]["model"]
        best_adapter = adapters[best_name]
        y_pred = best_adapter.predict(test_paths)
        plot_confusion(y_te, y_pred, title=f"Confusion — {best_name}",
                       save_path=str(Path(CFG.ART_DIR)/"figs"/f"confusion_{best_name}.png"),
                       collapse_to_3=CFG.CONFUSION_COLLAPSE_TO_3)
    plot_learning_curves(histories, out_dir=None)
    print("\n[Done] Centralized run complete.")

if __name__ == "__main__":
    main()
