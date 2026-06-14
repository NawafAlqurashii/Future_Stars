# training.py  (temporal model: features = season N, label = season N+1)
import argparse
import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from .preprocessing import build_temporal_training_table, FEATURE_COLUMNS
from .model import build_xgb, build_preprocessor, build_pipeline, save_model


def prepare_data(features_path, label_path, min_90s=5.0, max_age=23, pct=0.80):
    fdf = pd.read_csv(features_path)
    ldf = pd.read_csv(label_path)
    co = build_temporal_training_table(fdf, ldf, min_90s=min_90s, max_age=max_age, pct=pct)

    feats = [c for c in FEATURE_COLUMNS if c != "Role" and c in co.columns]
    cols = feats + (["Role"] if "Role" in co.columns else [])
    X = co[cols].copy()
    y = co["Future_Star"]

    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    print(f"Cohort: {len(co)} | positive rate: {y.mean():.3f}")
    print(f"Features: {num_cols + cat_cols}")
    return X, y, num_cols, cat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-data", dest="features",
                    default="data/players_data_light-2024_2025.csv",
                    help="season N CSV (model features)")
    ap.add_argument("--label-data", dest="label",
                    default="data/players_data_light-2025_2026.csv",
                    help="season N+1 CSV (next-season label)")
    ap.add_argument("--min-90s", type=float, default=5.0, dest="min_90s")
    ap.add_argument("--max-age", type=int, default=23, dest="max_age")
    ap.add_argument("--pct", type=float, default=0.80)
    args = ap.parse_args()

    X, y, num_cols, cat_cols = prepare_data(args.features, args.label,
                                            args.min_90s, args.max_age, args.pct)

    spw = (y == 0).sum() / max(1, (y == 1).sum())
    pipe = build_pipeline(build_preprocessor(num_cols, cat_cols),
                          build_xgb(scale_pos_weight=spw))

    # Honest cross-validated estimate
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {
        "cv_roc_auc": round(float(roc_auc_score(y, oof)), 4),
        "cv_pr_auc": round(float(average_precision_score(y, oof)), 4),
        "cv_f1_at_0.5": round(float(f1_score(y, (oof >= 0.5).astype(int), zero_division=0)), 4),
        "n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
    }
    print(json.dumps(metrics, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    json.dump(metrics, open("outputs/metrics.json", "w"), indent=2)

    # Fit final model on all data and save
    pipe.fit(X, y)
    Path("model").mkdir(exist_ok=True)
    save_model(pipe, "model/model.pkl")
    print("Model trained (temporal) and saved to model/model.pkl")


if __name__ == "__main__":
    main()
