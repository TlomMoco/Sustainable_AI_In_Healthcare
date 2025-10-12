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