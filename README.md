# ⚽ Future Star — Football Talent Prediction

> Predict whether a **young footballer** will become a **top performer in their
> position next season** — to help Saudi academies scout objectively, aligned
> with **Vision 2030**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED)
![License](https://img.shields.io/badge/License-MIT-yellow)

### 🔗 [**▶ Try the live app: futurestars.streamlit.app**](https://futurestars.streamlit.app/)
🖼️ **Screenshots:** see `outputs/` (confusion matrix, feature importance, threshold curve)

> **TL;DR** — An early *same-season* version scored a suspicious **ROC-AUC ≈ 0.99**.
> That was **target leakage**. After diagnosing and fixing it, the honest
> **temporal** model predicts next-season breakout with **ROC-AUC ≈ 0.75** — and
> its top picks are genuine young stars (Saka, Wirtz, Palmer, Musiala).

---

## 📌 Table of Contents
- [Problem](#-problem)
- [Approach](#-approach)
- [Data](#-data)
- [Methodology](#-methodology)
- [The leakage story](#-the-leakage-story)
- [Results](#-results)
- [Project structure](#-project-structure)
- [How to run](#-how-to-run)
- [Using the app](#-using-the-app)
- [Limitations](#-limitations)
- [Future work](#-future-work)
- [Tech stack](#-tech-stack) · [License](#-license) · [Author](#-author)

---

## 🎯 Problem
Scouting relies on subjective judgment and limited match exposure, so promising
youth players get missed and evaluations carry bias. The goal is an objective,
repeatable way to flag young players likely to break out — **fairly within their
position** (a defender shouldn't be judged by a striker's goals).

## 💡 Approach
1. Data cleaning & preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering & a leakage-free target
4. Model training & honest evaluation
5. Deployment (API & Streamlit)

## 📊 Data
Per-90 player statistics from the **top-5 European leagues** (FBref):

| Season | Role | Notes |
|--------|------|-------|
| **2024-25** | model **features** | full: attacking, defensive, GK, xG (~2,850 players, 165 cols) |
| **2025-26** | **label** (next season) | partial export (used to define the outcome) |

- **Target:** binary "Future Star" label **+ a probability score (%)**.
- **Cohort:** young players (Age ≤ 23) with ≥ 5 full matches → **630 players**,
  of whom **~15%** become a top-20% performer in their position next season.

## 🧠 Methodology
- **Per-90 features** so playing time doesn't bias comparisons.
- **Position-fair label:** within each role, a position-specific score (goal
  contributions for FW/MF, defensive actions for DF, save% for GK); the **top 20%**
  of each position are "future stars".
- **Temporal (leakage-free) design:** features come from **2024-25**, the label is
  computed on **2025-26**. Because the label is from a *different season*, strong
  metrics like `GA_per90` / `Def_Actions_per90` are **legitimate predictors, not
  leakage**.
- **Data-quality fixes:** structural goalkeeper-NaN handling, a position-mapping
  fix (~349 forwards were mislabeled), a minimum-minutes filter (≈30% of players
  removed), and an age constraint.
- **Pipeline:** `SimpleImputer` + `OneHotEncoder` inside a `ColumnTransformer`
  (fit on training folds only), XGBoost with `scale_pos_weight`, evaluated with
  **5-fold cross-validation**.

## 🔍 The leakage story
The first model labeled the top 20% **this season** and trained on **this
season's** stats — scoring **ROC-AUC ≈ 0.99**. That's a red flag: the label was
computed from the same metrics fed to the model, so it was just re-deriving the
labeling rule.

**Proof:** removing the label-defining features (and keeping only independent
style-of-play features) collapses the score. That gap *is* the leakage.

| Model | Label | ROC-AUC | Verdict |
|-------|-------|---------|---------|
| Same-season (leaky) | same season | **≈ 0.99** | ❌ leakage — meaningless |
| Same-season, style only | same season | ≈ 0.85 | proves the leakage |
| **Temporal (final)** | **next season** | **≈ 0.75** | ✅ **honest prediction** |

The fix wasn't to drop the strong metrics (they're what make a player good) — it
was to change **what the label means** so they become valid predictors again.

## 📈 Results
Final temporal model, 5-fold cross-validation (out-of-fold):

| Metric | Score |
|--------|-------|
| ROC-AUC | **≈ 0.75** |
| PR-AUC | ≈ 0.36 |
| F1 (@ 0.5) | ≈ 0.44 |

**Operating point (scouting shortlist):** the decision threshold is tuned for
**recall ≈ 0.70** (catch most prospects; humans review the list). The resulting
**precision ≈ 0.30** is about **2× the 15% base rate** — a genuinely useful
shortlist.

**Face validity** — top-scored young prospects (out-of-fold) include
*Bukayo Saka, Florian Wirtz, Cole Palmer, Arda Güler, Xavi Simons, Jamal
Musiala* — real, highly-rated young talents.

> Predicting the *future* in football is hard (injuries, transfers, development
> variance). A transparent ROC-AUC ≈ 0.75 is an honest, useful result — worth far
> more than an inflated 0.99 that secretly knows the answer.

## 📁 Project structure
```
Future_Stars/
├── api/              # FastAPI app (serving)
├── data/             # season CSVs (git-ignored)
├── future_stars/     # core package: preprocessing, model, training, predict, evaluation
├── model/            # saved model.pkl (temporal)
├── notebooks/        # future_stars_final.ipynb (the main case study) + experiments
├── outputs/          # metrics.json, figures, predictions
├── tests/            # unit tests
├── Dockerfile
├── main.py           # entry point -> future_stars.training.main
├── requirements.txt
└── requirements-dev.txt
```

## ▶️ How to run

### Run locally
```bash
git clone https://github.com/NawafAlqurashii/Future_Stars.git
cd Future_Stars
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
 
# (optional) retrain the temporal model -> model/model.pkl
python main.py --features-data data/players_data_light-2024_2025.csv \
               --label-data    data/players_data_light-2025_2026.csv
 
# launch the web app (self-contained: loads the model directly)
streamlit run app.py
```
 
### REST API (optional)
A FastAPI service is also included for programmatic scoring:
```bash
uvicorn api.api:app --reload --port 8080   # POST /predict_one, /predict_file
```
 
## 🕹️ Using the app
 
The app ([futurestars.streamlit.app](https://futurestars.streamlit.app/)) has two modes:
 
**1) Predict a single player** — fill in a player's season stats and get a
**Future Star score (%)**. Tips for a meaningful score:
- pick a real **position** (GK/DF/MF/FW),
- enter **realistic minutes** (a full season is ~2,000–3,400, not tens of thousands),
- open **"More stats"** (shots, key passes, progressive carries/runs) — the model
  uses all of them, so filling them gives a far more accurate score.
**2) Upload a squad (CSV)** — score a whole league at once, then explore the
**Analysis Dashboard** (filters, KPIs, rankings, charts).
 
### What CSV to upload
One row per player, using **FBref** column names. The easiest source is an FBref
top-5-leagues per-90 export — the **same format as the included
`data/players_data_light-2024_2025.csv`**, which you can upload directly to try it.
Missing columns are handled automatically (filled and imputed), but more complete
data gives more accurate scores. Key columns the model reads:
 
| Column | Meaning |
|--------|---------|
| `Player`, `Pos`, `Age`, `Min` | name, position (e.g. `FW,MF`), age, minutes |
| `G+A`, `xG`, `xAG` | goals+assists, expected goals, expected assisted goals |
| `Sh/90`, `SoT/90` | shots / shots on target per 90 |
| `KP`, `PrgP`, `PrgC`, `PrgR` | key passes, progressive passes / carries / runs |
| `Tkl`, `Blocks_stats_defense`, `Clr` | tackles, blocks, clearances |
| `Save%` | goalkeeper save percentage |
 
### What you get back
For each player the app builds per-90 features (and `Role` from position), runs the
XGBoost temporal model, and returns: **Prediction** (Future Star / Not),
**Probability (%)** (the score), and a position-relevant **Key Metric**.

## ⚠️ Limitations
- **2025-26 is a partial export** (no xG / defensive detail), so the next-season
  label is approximate.
- **Survivorship:** players absent in 2025-26 are treated as "did not break out"
  (some may be injuries or moves to uncovered leagues).
- **A single season-to-season step;** more seasons would strengthen it.
- **Stats only** — ignores physical, tactical, psychological and injury context.

## 🚀 Future work
- More full seasons + growth-trajectory features.
- Probability calibration so the score % is well-calibrated.
- A Streamlit scout-facing app on top of the API.

## 🧰 Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · FastAPI · Docker · Google Cloud Run


## 👤 Author
**Nawaf Alqurashi** — Data Analyst
[Portfolio]⟨[link](https://nawafalqurashii.github.io/)⟩ · [LinkedIn](https://www.linkedin.com/in/nawafqurashi) · [GitHub](https://github.com/NawafAlqurashii)

> Built during the Saudi Digital Academy × Le Wagon Data Science & AI Bootcamp, Riyadh.
