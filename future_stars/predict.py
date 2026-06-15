# predict.py  (robust to any CSV schema -> no "upload" crashes)
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .model import load_model
from .preprocessing import (
    clean_data, normalize_country, normalize_positions,
    build_per90_features, FEATURE_COLUMNS,
)


def get_key_metric(row):
    role = row.get("Role")
    if role == "GK":
        if pd.notna(row.get("Save_Pct")):
            return f"Save%: {(row['Save_Pct'] * 100):.2f}%"
        return "No GK metric"
    elif role == "DF":
        return f"DefActions90: {row.get('Def_Actions_per90', 0):.2f} actions/90"
    elif role in ("MF", "FW"):
        ga = row.get("GA_per90")
        xga = row.get("xGA_per90")
        if pd.notna(ga) and (pd.isna(xga) or ga >= xga):
            return f"GA90: {ga:.2f} contrib/90"
        elif pd.notna(xga):
            return f"xGA90: {xga:.2f} expected contrib/90"
        return "No attacking metric"
    return "NA"


def predict(data=None, data_path=None, model_path="model/model.pkl", save_path=None):
    """Score players and return Player, Age, Pos, Role, Prediction, Probability, Key Metric.

    Robust to CSVs that are missing some columns: any model feature that can't be
    built is created as NaN and imputed by the pipeline, so the call never crashes
    on a schema mismatch.
    """
    model = load_model(model_path)

    if data is not None:
        df = data.copy()
    elif data_path is not None:
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Provide either a DataFrame (data) or a CSV path (data_path).")

    if df.empty:
        raise ValueError("No rows to score (empty input).")

    df = clean_data(df, min_90s=0)          # score everyone (just drop 0-minute rows)
    df = normalize_country(df)
    df = normalize_positions(df)
    df = build_per90_features(df)

    # --- Guarantee every model feature exists (the key fix for upload errors) ---
    if "Role" not in df.columns:
        df["Role"] = "Other"
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df_model = df[FEATURE_COLUMNS].copy()
    df_model["Role"] = df_model["Role"].fillna("Other").astype("object")

    preds = model.predict(df_model)
    proba = model.predict_proba(df_model)[:, 1]

    results = df.copy()
    results["Prediction"] = pd.Series(preds, index=results.index).map({1: "Future Star", 0: "Not Future Star"})
    results["Probability"] = [f"{p * 100:.2f}%" for p in proba]
    results["Key Metric"] = results.apply(get_key_metric, axis=1)

    cols = [c for c in ["Player", "Age", "Pos", "Role", "Prediction", "Probability", "Key Metric"]
            if c in results.columns]
    final_results = results[cols].reset_index(drop=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        final_results.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="test.csv")
    parser.add_argument("--model", type=str, default="model/model.pkl")
    parser.add_argument("--save", type=str, default="outputs/predictions.csv")
    args = parser.parse_args()
    predict(data_path=args.data, model_path=args.model, save_path=args.save)
