# evaluation.py  (fixed decision threshold)
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report,
)


def evaluate(name, pipeline, X_test, y_test, decision_threshold=0.5, out_dir="outputs"):
    """
    Evaluate at a probability DECISION threshold (default 0.5).
    Note: this is unrelated to the labeling percentile (e.g. 0.80) used to
    define future stars — don't reuse that here.
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= decision_threshold).astype(int)

    prec, rec, _ = precision_recall_curve(y_test, proba)
    metrics = {
        "model": name,
        "decision_threshold": decision_threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(auc(rec, prec)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, digits=3, zero_division=0),
    }
    Path(out_dir).mkdir(exist_ok=True)
    with open(Path(out_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
