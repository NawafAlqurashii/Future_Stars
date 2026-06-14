# preprocessing.py  (temporal-ready)
#
# Adds to the previous version:
#   - build_per90_features() also creates KP_per90 / PrgC_per90 / PrgR_per90
#   - FEATURE_COLUMNS expanded to the 12-feature temporal set (+ Role)
#   - helpers for the TEMPORAL label: dedup_by_player, next_season_star_keys,
#     build_temporal_training_table  (features = season N, label = season N+1)

import numpy as np
import pandas as pd
import pycountry

SELECTED_COLUMNS = [
    "Player", "Nation", "Pos", "Age",
    "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "xG", "xAG", "G+A",
    "Tkl", "TklW", "Blocks_stats_defense", "Clr", "Err",
    "PrgP", "PrgC", "KP", "xA",
    "GA", "Saves", "Save%", "CS", "CS%", "PKA", "PKsv",
    "CrdY", "CrdR",
]

# Single source of truth for the model feature list (temporal model).
FEATURE_COLUMNS = [
    "GA_per90", "xGA_per90", "Def_Actions_per90", "ProgPass_per90", "Save_Pct",
    "Age", "90s", "Sh/90", "SoT/90", "KP_per90", "PrgC_per90", "PrgR_per90",
    "Role",
]


# ----------------------------- Validation & Cleaning -----------------------------
def validate_columns(df: pd.DataFrame):
    missing = [c for c in SELECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[SELECTED_COLUMNS]


def clean_data(df: pd.DataFrame, min_90s: float = 5.0) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    if "Min" in df.columns:
        df = df[df["Min"] > 0]
    if "90s" in df.columns:
        df = df[pd.to_numeric(df["90s"], errors="coerce") >= min_90s]
    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].fillna("Unknown")
    return df.reset_index(drop=True)


def normalize_country(df: pd.DataFrame) -> pd.DataFrame:
    if "Nation" not in df.columns:
        return df

    def full_country_name(alpha3):
        try:
            m = pycountry.countries.get(alpha_3=str(alpha3))
            return m.name if m else alpha3
        except Exception:
            return alpha3

    df = df.copy()
    df["Nation"] = df["Nation"].astype(str).str.split().str[-1].apply(full_country_name)
    return df


def map_role(pos: str) -> str:
    primary = str(pos).split(",")[0].strip()
    return primary if primary in {"GK", "DF", "MF", "FW"} else "Other"


def normalize_positions(df: pd.DataFrame) -> pd.DataFrame:
    if "Pos" not in df.columns:
        return df
    df = df.copy()
    df["Role"] = df["Pos"].apply(map_role).astype("object")
    return df


# ----------------------------- Feature engineering -----------------------------
def safe_per90(num, minutes):
    return (num / minutes.replace(0, np.nan)) * 90


def build_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"G+A", "Min"}.issubset(df.columns):
        df["GA_per90"] = safe_per90(df["G+A"], df["Min"])
    if {"xG", "xAG", "Min"}.issubset(df.columns):
        df["xGA_per90"] = safe_per90(df["xG"] + df["xAG"], df["Min"])

    comb = [c for c in ["Tkl", "Blocks_stats_defense", "Clr"] if c in df.columns]
    if comb and "Min" in df.columns:
        df["Def_Actions_per90"] = safe_per90(df[comb].sum(axis=1), df["Min"])

    # Progression per-90 (added for the temporal model)
    for raw, new in [("PrgP", "ProgPass_per90"), ("KP", "KP_per90"),
                     ("PrgC", "PrgC_per90"), ("PrgR", "PrgR_per90")]:
        if {raw, "Min"}.issubset(df.columns):
            df[new] = safe_per90(df[raw], df["Min"])

    # GK save % — structural NaN for outfielders -> 0 (Role separates positions)
    if "Save%" in df.columns:
        sp = pd.to_numeric(df["Save%"].astype(str).str.strip().str.replace("%", "", regex=False),
                           errors="coerce")
        df["Save_Pct"] = np.where(sp > 1.0, sp / 100.0, sp)
    else:
        df["Save_Pct"] = np.nan
    df["Save_Pct"] = df["Save_Pct"].fillna(0.0)
    return df


# ----------------------------- Same-season label (kept for reference) -----------------------------
def role_score_row(row):
    role = row.get("Role", "Other")
    if role == "GK":
        return row.get("Save_Pct")
    if role == "DF":
        return row.get("Def_Actions_per90")
    if role in ("MF", "FW"):
        g, x = row.get("GA_per90"), row.get("xGA_per90")
        if pd.isna(g) and pd.isna(x):
            return np.nan
        if pd.isna(g):
            return x
        if pd.isna(x):
            return g
        return max(g, x)
    return np.nan


def apply_future_star_label(df: pd.DataFrame, pct: float = 0.80, max_age: int = 21):
    if "Role" not in df.columns:
        raise ValueError("Required column 'Role' not found")
    df = df.copy()
    if "Age" in df.columns:
        df = df[df["Age"].notna() & (df["Age"] <= max_age)]
    df["Role_Score"] = df.apply(role_score_row, axis=1)
    df = df[df["Role_Score"].notna()].reset_index(drop=True)
    thresholds = {r: sub["Role_Score"].quantile(pct) for r, sub in df.groupby("Role")}
    df["role_threshold"] = df["Role"].map(thresholds)
    df["Future_Star"] = (df["Role_Score"] >= df["role_threshold"]).astype(int)
    return df, thresholds


# ----------------------------- Temporal label (the real model) -----------------------------
def dedup_by_player(df: pd.DataFrame) -> pd.DataFrame:
    """Players who transfer mid-season appear twice; keep their main spell."""
    df = df.copy()
    df["_player_key"] = df["Player"].astype(str) + "|" + df["Born"].astype(str)
    return df.sort_values("Min").drop_duplicates("_player_key", keep="last")


def next_season_star_keys(label_df: pd.DataFrame, pct: float = 0.80, min_90s: float = 5.0):
    """Top-(1-pct) per role in the LABEL season, using the best metrics it has."""
    df = dedup_by_player(normalize_positions(label_df.copy()))
    df = df[pd.to_numeric(df["90s"], errors="coerce") >= min_90s].copy()
    m = df["Min"].replace(0, np.nan)

    attack = df["G+A"] / m * 90
    if {"TklW", "Int"}.issubset(df.columns):
        defense = (df["TklW"].fillna(0) + df["Int"].fillna(0)) / m * 90
    else:
        cols = [c for c in ["Tkl", "Blocks_stats_defense", "Clr"] if c in df.columns]
        defense = df[cols].sum(axis=1) / m * 90 if cols else pd.Series(np.nan, index=df.index)
    if "Save%" in df.columns:
        sp = pd.to_numeric(df["Save%"].astype(str).str.replace("%", "", regex=False), errors="coerce")
        gk = np.where(sp > 1, sp / 100, sp)
    else:
        gk = np.nan

    df["_score"] = np.where(df["Role"] == "GK", gk,
                     np.where(df["Role"] == "DF", defense, attack))
    df = df[pd.notna(df["_score"])]
    thr = {r: sub["_score"].quantile(pct) for r, sub in df.groupby("Role")}
    return set(df.loc[df["_score"] >= df["Role"].map(thr), "_player_key"])


def build_temporal_training_table(feature_df: pd.DataFrame, label_df: pd.DataFrame,
                                  min_90s: float = 5.0, max_age: int = 23, pct: float = 0.80):
    """
    Features come from `feature_df` (season N); the Future_Star label is computed
    on `label_df` (season N+1). Returns the young cohort with FEATURE_COLUMNS + label.
    """
    feats = build_per90_features(normalize_positions(clean_data(feature_df, min_90s=min_90s)))
    feats = dedup_by_player(feats)
    keys = next_season_star_keys(label_df, pct=pct, min_90s=min_90s)
    co = feats[(pd.to_numeric(feats["Age"], errors="coerce") <= max_age) &
               (feats["Role"] != "Other")].copy()
    co["Future_Star"] = co["_player_key"].isin(keys).astype(int)
    return co.reset_index(drop=True)


def full_preprocessing(df: pd.DataFrame, min_90s: float = 5.0, pct: float = 0.80, max_age: int = 21):
    df = validate_columns(df)
    df = clean_data(df, min_90s=min_90s)
    df = normalize_country(df)
    df = normalize_positions(df)
    df = build_per90_features(df)
    df, thresholds = apply_future_star_label(df, pct=pct, max_age=max_age)
    return df, thresholds
