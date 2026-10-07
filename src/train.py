"""Train, compare, tune and save the PHA classifier.

Run:  python -m src.train

Steps
 1. Stratified, asteroid-grouped train/test split (a given asteroid is never in both).
 2. Compare 3 models x 2 imbalance strategies (class weights vs SMOTE) with 5-fold CV.
    Preprocessing and SMOTE live INSIDE the Pipeline, so each CV fold fits them on its
    own training part only -> no leakage.
 3. Tune the best combination with RandomizedSearchCV (PR-AUC).
 4. Choose a decision threshold from out-of-fold predictions so recall >= TARGET_RECALL.
 5. Repeat the final fit for the ablation experiments and save everything.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold,
                                     cross_val_predict, cross_validate)
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src import clean as clean_mod
from src.config import (CLEAN_CSV, EXPERIMENTS, GROUP, MAIN_EXPERIMENT, METRICS_PATH,
                        MODEL_PATH, N_FOLDS, RAW_FEATURES, SEED, TARGET, TARGET_RECALL,
                        TEST_SIZE)
from src.features import add_features, select_columns

SCORING = {"recall": "recall", "precision": "precision", "f1": "f1",
           "roc_auc": "roc_auc", "pr_auc": "average_precision"}


# --------------------------------------------------------------------- data ----
def load_data() -> pd.DataFrame:
    """Load the cleaned data (cleaning the raw file first if needed)."""
    if not CLEAN_CSV.exists():
        df = clean_mod.clean(clean_mod.load_raw())
        CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_CSV, index=False)
    return pd.read_csv(CLEAN_CSV)


def split_data(df: pd.DataFrame):
    """Stratified + grouped split: returns (train_df, test_df).

    The same asteroid can have many close approaches, so rows are grouped by asteroid.
    Otherwise the model could memorise an asteroid in training and be "tested" on it.
    """
    sgkf = StratifiedGroupKFold(n_splits=int(1 / TEST_SIZE), shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET], df[GROUP]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cv_splitter() -> StratifiedGroupKFold:
    """Stratified 5-fold CV that keeps each asteroid inside one fold."""
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# ----------------------------------------------------------------- pipeline ----
def make_classifier(name: str, imbalance: str, pos_weight: float):
    """Build a classifier. 'class_weight' re-weights errors; 'smote' leaves weights alone."""
    weighted = imbalance == "class_weight"
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if weighted else None)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED,
            class_weight="balanced_subsample" if weighted else None)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
            tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=SEED,
            scale_pos_weight=pos_weight if weighted else 1.0)
    raise ValueError(name)


def build_pipeline(name: str, imbalance: str, feature_cols: list[str],
                   pos_weight: float = 1.0) -> Pipeline:
    """raw columns -> features -> impute (+missing flags) -> scale -> [SMOTE] -> model."""
    steps = [
        ("features", FunctionTransformer(add_features)),
        ("select", FunctionTransformer(select_columns, kw_args={"cols": feature_cols})),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))  # only applied while fitting
    steps.append(("clf", make_classifier(name, imbalance, pos_weight)))
    return Pipeline(steps)


PARAM_GRIDS = {
    "logreg": {"clf__C": uniform(0.01, 10)},
    "random_forest": {"clf__max_depth": [6, 10, 14, None],
                      "clf__min_samples_leaf": randint(1, 20),
                      "clf__max_features": ["sqrt", 0.5, 1.0]},
    "xgboost": {"clf__max_depth": randint(3, 9), "clf__learning_rate": uniform(0.02, 0.2),
                "clf__n_estimators": randint(150, 500), "clf__subsample": uniform(0.6, 0.4),
                "clf__min_child_weight": randint(1, 10)},
}


# ------------------------------------------------------------------ metrics ----
def pick_threshold(y_true, proba, target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold that still reaches target_recall (=> best precision at that recall)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def score_at(y_true, proba, threshold: float) -> dict:
    """Threshold-dependent and threshold-free metrics for one set of predictions."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {"threshold": round(threshold, 4),
            "recall": recall_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred), "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def fit_and_report(pipe: Pipeline, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """OOF-tune the threshold on train, fit on all train, report on the untouched test set."""
    Xtr, ytr, gtr = train[RAW_FEATURES], train[TARGET], train[GROUP]
    oof = cross_val_predict(pipe, Xtr, ytr, cv=cv_splitter(), groups=gtr,
                            method="predict_proba")[:, 1]
    threshold = pick_threshold(ytr, oof)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(test[RAW_FEATURES])[:, 1]
    return {"cv_oof_pr_auc": average_precision_score(ytr, oof), "threshold": threshold,
            "test_at_tuned_threshold": score_at(test[TARGET], proba, threshold),
            "test_at_0.5": score_at(test[TARGET], proba, 0.5)}, pipe


# --------------------------------------------------------------------- main ----
def main() -> None:
    t0 = time.time()
    df = load_data()
    train, test = split_data(df)
    assert not set(train[GROUP]) & set(test[GROUP]), "asteroid leaked across the split"
    pos_weight = float((train[TARGET] == 0).sum() / (train[TARGET] == 1).sum())
    print(f"train={len(train):,} test={len(test):,} | hazardous share "
          f"train={train[TARGET].mean():.3f} test={test[TARGET].mean():.3f}")

    main_cols = EXPERIMENTS[MAIN_EXPERIMENT]
    Xtr, ytr, gtr = train[RAW_FEATURES], train[TARGET], train[GROUP]

    # 1) model x imbalance comparison on the main experiment ------------------
    comparison = []
    for name in ["logreg", "random_forest", "xgboost"]:
        for imb in ["class_weight", "smote"]:
            res = cross_validate(build_pipeline(name, imb, main_cols, pos_weight), Xtr, ytr,
                                 cv=cv_splitter(), groups=gtr, scoring=SCORING)
            row = {"model": name, "imbalance": imb}
            row |= {m: float(res[f"test_{m}"].mean()) for m in SCORING}
            row |= {f"{m}_std": float(res[f"test_{m}"].std()) for m in SCORING}
            comparison.append(row)
            print(f"[{time.time() - t0:5.0f}s] {name:14s} {imb:12s} "
                  f"PR-AUC={row['pr_auc']:.3f} recall={row['recall']:.3f} precision={row['precision']:.3f}")
    best = max(comparison, key=lambda r: r["pr_auc"])
    print("Best by CV PR-AUC:", best["model"], best["imbalance"])

    # 2) tune the best combination ---------------------------------------------
    search = RandomizedSearchCV(
        build_pipeline(best["model"], best["imbalance"], main_cols, pos_weight),
        PARAM_GRIDS[best["model"]], n_iter=15, scoring="average_precision",
        cv=cv_splitter(), random_state=SEED, n_jobs=1, refit=False)
    search.fit(Xtr, ytr, groups=gtr)
    best_params = search.best_params_
    print(f"[{time.time() - t0:5.0f}s] tuned PR-AUC={search.best_score_:.3f}", best_params)

    # 3) final fit for every experiment (same model + tuned params) ------------
    experiments, final_pipe = {}, None
    for exp, cols in EXPERIMENTS.items():
        pipe = build_pipeline(best["model"], best["imbalance"], cols, pos_weight)
        pipe.set_params(**best_params)
        report, pipe = fit_and_report(pipe, train, test)
        report["features"] = cols
        experiments[exp] = report
        t = report["test_at_tuned_threshold"]
        print(f"[{time.time() - t0:5.0f}s] {exp:13s} test PR-AUC={t['pr_auc']:.3f} "
              f"ROC-AUC={t['roc_auc']:.3f} recall={t['recall']:.3f} precision={t['precision']:.3f}")
        if exp == MAIN_EXPERIMENT:
            final_pipe, final_threshold = pipe, report["threshold"]

    # 4) save -------------------------------------------------------------------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": final_pipe, "threshold": final_threshold,
                 "features": main_cols, "raw_features": RAW_FEATURES,
                 "model_name": best["model"], "imbalance": best["imbalance"]}, MODEL_PATH)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "split": {"train_rows": len(train), "test_rows": len(test),
                  "train_asteroids": int(train[GROUP].nunique()),
                  "test_asteroids": int(test[GROUP].nunique()),
                  "test_hazardous_share": float(test[TARGET].mean())},
        "cv_comparison": comparison, "best_combo": {"model": best["model"], "imbalance": best["imbalance"]},
        "tuned_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
        "tuned_cv_pr_auc": float(search.best_score_), "target_recall": TARGET_RECALL,
        "main_experiment": MAIN_EXPERIMENT, "experiments": experiments}, indent=2, default=str))
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
