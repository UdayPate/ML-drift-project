# DATA 245 Project — Classical ML Robustness Under Data Drift

**Research question:** When data changes over time, can drift-triggered retraining maintain better predictive performance than static or periodically retrained classical ML models?

Project rules and step order: see `AGENTS.md`, `PROGRESS.md`, and `DECISIONS.md`.

## Get the data (not stored in Git)
1. Download from UCI: https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Unzip `bank+marketing.zip`, then unzip `bank-additional.zip` inside it.
3. Copy **`bank-additional-full.csv`** (41,188 rows) into `data/raw/`.
   Do not use `bank-full.csv` (older version, no economic features) or `bank-additional.csv` (10% sample).
4. Load with `pd.read_csv("../data/raw/bank-additional-full.csv", sep=";")` — the file is semicolon-separated.

## Layout
```
data/raw/        bank-additional-full.csv (local only, ignored by Git)
notebooks/       01_EDA.ipynb, ... — run from inside notebooks/
figures/         saved plots for slides and the report
```

## Data notes
- `duration` is leakage (only known after the call) → excluded
- Rows are in time order (May 2008 – Nov 2010); there is no date column
- Missing values are coded as `"unknown"`; `pdays = 999` means never contacted before
