"""Evaluate the saved main model on the held-out test set and write figures.

Run:  python -m src.evaluate
Outputs (reports/figures/): confusion matrix, ROC, PR curve, feature importance,
SHAP summary + force plots, error analysis plot; also reports/error_analysis.md.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             average_precision_score, roc_auc_score)

from src.config import FIG_DIR, METRICS_PATH, MODEL_PATH, RAW_FEATURES, REPORTS_DIR, TARGET
from src.features import add_features
from src.train import load_data, split_data


def save(fig, name: str) -> None:
    """Save a figure to reports/figures and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def transformed_matrix(pipe, X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Run every pipeline step except the classifier (samplers are skipped at predict time)."""
    Z = pipe[:-1].transform(X)
    return pd.DataFrame(Z, columns=names[:Z.shape[1]])


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    pipe, thr, names = bundle["pipeline"], bundle["threshold"], bundle["features"]
    _, test = split_data(load_data())
    y = test[TARGET].to_numpy()
    proba = pipe.predict_proba(test[RAW_FEATURES])[:, 1]
    pred = (proba >= thr).astype(int)
    print(f"Threshold={thr:.4f}  PR-AUC={average_precision_score(y, proba):.3f}  "
          f"ROC-AUC={roc_auc_score(y, proba):.3f}")

    # Confusion matrix at the tuned threshold ----------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Not PHA", "PHA"], cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Confusion matrix (threshold={thr:.3f})")
    save(fig, "confusion_matrix.png")

    # ROC and PR curves ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.legend()
    ax.set_title("ROC curve")
    save(fig, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax, name="Model")
    ax.axhline(y.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y.mean():.2f})")
    ax.legend()
    ax.set_title("Precision-recall curve")
    save(fig, "pr_curve.png")

    # Feature importance (model-native) -----------------------------------------
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=names[:len(clf.feature_importances_)])
        fig, ax = plt.subplots(figsize=(6, 4))
        imp.sort_values().plot.barh(ax=ax, color="#3b6fb6")
        ax.set_title("Feature importance (model-native)")
        save(fig, "feature_importance.png")

    # SHAP -------------------------------------------------------------------------
    sample = test.sample(min(2000, len(test)), random_state=0)
    Z = transformed_matrix(pipe, sample[RAW_FEATURES], names)
    explainer = shap.TreeExplainer(clf)
    sv = explainer(Z)
    values = sv.values[..., 1] if sv.values.ndim == 3 else sv.values  # RF returns 2 classes
    base = sv.base_values[..., 1] if sv.base_values.ndim == 2 else sv.base_values
    # SHAP values come from the scaled matrix; show the real (unscaled) values in plots
    shown = add_features(sample[RAW_FEATURES])[names].reset_index(drop=True)
    shap.summary_plot(values, shown, show=False)
    save(plt.gcf(), "shap_summary.png")

    # Force plots: the most confident true positive and a missed PHA -------------
    s_proba = pipe.predict_proba(sample[RAW_FEATURES])[:, 1]
    s_y = sample[TARGET].to_numpy()
    tp = np.where((s_y == 1))[0][np.argmax(s_proba[s_y == 1])]
    fn_candidates = np.where((s_y == 1) & (s_proba < thr))[0]
    for label, i in [("true_positive", tp)] + ([("false_negative", fn_candidates[0])] if len(fn_candidates) else []):
        shap.force_plot(float(np.ravel(base)[i]), values[i], shown.iloc[i].round(2),
                        matplotlib=True, show=False)
        plt.title(f"SHAP force plot - {label} (P(PHA)={s_proba[i]:.2f})", y=1.15)
        save(plt.gcf(), f"shap_force_{label}.png")

    # Error analysis --------------------------------------------------------------
    err = test.assign(proba=proba, pred=pred)
    err["kind"] = np.select([(y == 1) & (pred == 1), (y == 1) & (pred == 0),
                             (y == 0) & (pred == 1)], ["TP", "FN", "FP"], "TN")
    cols = ["absolute_magnitude", "relative_velocity", "miss_distance"]
    summary = err.groupby("kind")[cols].median().round(2)
    counts = err["kind"].value_counts()
    # Error rate by H bin: where does the model fail?
    err["H_bin"] = pd.cut(err["absolute_magnitude"], [0, 18, 20, 22, 24, 26, 40])
    by_h = err.groupby("H_bin", observed=True).apply(
        lambda g: pd.Series({"rows": len(g), "pha_share": g[TARGET].mean(),
                             "recall": (g.pred[g[TARGET] == 1] == 1).mean() if g[TARGET].sum() else np.nan,
                             "false_alarm_rate": (g.pred[g[TARGET] == 0] == 1).mean()}),
        include_groups=False).round(3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, color in [("TN", "#bbbbbb"), ("FP", "#e08a1e"), ("FN", "#c0392b"), ("TP", "#2e7d32")]:
        d = err[err.kind == kind]
        axes[0].scatter(d.absolute_magnitude, d.miss_distance / 384400, s=6, alpha=.4, label=f"{kind} ({len(d)})", c=color)
    axes[0].set_yscale("log"); axes[0].set_xlabel("Absolute magnitude H"); axes[0].set_ylabel("Miss distance (lunar distances)")
    axes[0].legend(markerscale=3); axes[0].set_title("Where the errors are")
    by_h[["recall", "false_alarm_rate"]].plot.bar(ax=axes[1], rot=30)
    axes[1].set_title("Recall and false-alarm rate by H bin"); axes[1].set_xlabel("H bin")
    save(fig, "error_analysis.png")

    REPORTS_DIR.mkdir(exist_ok=True)
    md = ["# Error analysis (test set, tuned threshold)\n",
          f"Counts: {counts.to_dict()}\n",
          "Median feature values by outcome:\n", summary.to_markdown(), "\n",
          "Performance by absolute-magnitude bin:\n", by_h.to_markdown(), "\n"]
    (REPORTS_DIR / "error_analysis.md").write_text("\n".join(md))
    print("\n".join(md))

    m = json.loads(METRICS_PATH.read_text())
    m["error_analysis"] = {"counts": {k: int(v) for k, v in counts.items()},
                           "by_h_bin": {str(k): v for k, v in by_h.to_dict("index").items()}}
    METRICS_PATH.write_text(json.dumps(m, indent=2, default=str))


if __name__ == "__main__":
    main()
