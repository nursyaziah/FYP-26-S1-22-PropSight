# PropSight

PropSight is a Singapore HDB resale analytics platform with three working parts in one repo:

- a Flask web app for price prediction, map views, analytics, saved predictions, and admin tooling
- a local ETL pipeline that stages raw HDB resale data in SQLite, geocodes addresses, and computes proximity features
- an ML pipeline that engineers features, trains multiple regressors, and produces versioned model artefacts for the web app

The important architecture change in the current codebase is this:

- the live website runs on Supabase plus local model artefacts
- SQLite is still used for local preprocessing and as a fallback ML source
- Supabase is the authoritative app database and the preferred ML training source

## What To Open First

If you are new to the repo, start here:

1. [`webapp/app.py`](webapp/app.py) for the runtime application
2. [`scripts/pipeline_orchestration.py`](scripts/pipeline_orchestration.py) for batch workflow entry points
3. [`ML/feature_engineering.py`](ML/feature_engineering.py) for feature creation and dataset splits
4. [`ML/model_training.py`](ML/model_training.py) for training, evaluation, and winner selection
5. [`Database/supabase_schema.sql`](Database/supabase_schema.sql) for the live database schema and RPCs

## Current Repo Snapshot

As committed in this repo today:

- `ML/model_assets/latest.txt` points to `model_assets/20260417_074102`
- that checked-in run declares `catboost` as the winner
- the recorded refit test metrics are `MAPE=3.6447%`, `RMSE=34,699.2`, `R2=0.973872`

Those values come from [`ML/model_assets/20260417_074102/metrics.json`](ML/model_assets/20260417_074102/metrics.json).

## Live Deployment

The project has a live hosted website at:

- https://propsight.onrender.com/

Based on the repo configuration, Render deployment is part of the automation path for model updates:

- the weekly retrain workflow can trigger a Render deploy hook
- that deploy step runs only when the retrain job succeeds and the `RENDER_DEPLOY_HOOK` secret is configured

## GitHub Actions Automation

This repo currently includes two production-style GitHub Actions workflows under [`.github/workflows/`](.github/workflows/):

- [`weekly-data-refresh.yml`](.github/workflows/weekly-data-refresh.yml)
- [`weekly-model-retrain.yml`](.github/workflows/weekly-model-retrain.yml)

What they do:

- `Weekly Data Refresh` runs on a schedule every Wednesday at `03:00 UTC`, which is `11:00 SGT`, and can also be started manually with `workflow_dispatch`
- it runs the preprocessing pipeline via `python run_data_preprocessing.py`
- `Weekly Model Retrain` runs after a successful `Weekly Data Refresh`, and can also be started manually
- it runs the ML pipeline, commits updated model artefacts, syncs model metadata to Supabase, and optionally triggers a Render deployment

## Quick Start

Run all commands from the project root.

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

There is no single root `requirements.txt` for the entire repo yet.

For the web app plus model loading:

```bash
pip install -r webapp/requirements.txt
pip install xgboost
```

For the full pipeline as well:

```bash
pip install requests optuna psycopg2-binary
```

Optional extras:

```bash
# only needed for the EDA script
pip install matplotlib seaborn scipy
```

Notes:

- [`webapp/requirements.txt`](webapp/requirements.txt) already includes `flask`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `catboost`, `shap`, `gunicorn`, and `pyarrow`
- `xgboost` is not in that file, but older and alternate model artefacts can still need it at load time
- the deployment runtime file is [`webapp/runtime.txt`](webapp/runtime.txt), currently `python-3.12.6`

### 3. Create a `.env`

`webapp/app.py` loads `.env` from the project root automatically.

Minimum variables to start the website:

```env
SECRET_KEY=change-me
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong-random-password>
```

Optional, for the current MAS API Gateway SORA endpoint:

```env
MAS_API_KEY=<mas-keyid>
```

Variables needed for Supabase SQL sync and preferred ML training source:

```env
SUPABASE_DB_URL=postgresql://...
```

Useful optional variables:

```env
SUPABASE_KEY=<alternate app key name accepted by webapp/app.py>
GEMINI_API_KEY=<optional; enables AI features>
MODEL_ASSETS_DIR=<override artefact root>
MODEL_ASSETS_RUN=<pin the app to a specific run folder>
MODEL_USE_LATEST_TXT=true
TRAINING_DATA_SOURCE=auto
HDB_SQLITE_PATH=Data Preprocessing/hdb_resale.db
HDB_REFERENCE_DATA_DIR=Data Preprocessing/reference_data
SUPABASE_USERS_TABLE=users
SUPABASE_PREDICTIONS_TABLE=saved_predictions
FLASK_HOST=127.0.0.1
FLASK_PORT=5001
HDB_BACKFILL_GEOCODING=1
HDB_BACKFILL_PROXIMITY=1
HDB_STRICT_EXTERNAL_STEPS=1
MAX_MAPE_REGRESSION_PCT=10
```

## Common Workflows

### Run the website only

Use this when you want the Flask app and already have model artefacts plus Supabase data:

```bash
python webapp/app.py
```

What this needs:

- `.env` with `SECRET_KEY`, `SUPABASE_URL`, a Supabase key, `ADMIN_EMAIL`, and `ADMIN_PASSWORD`
- model artefacts under `ML/model_assets/` or `MODEL_ASSETS_DIR`

Runtime details from the code:

- default host is `127.0.0.1`
- default port is `5001`
- the app reads model artefacts at startup
- the website does not read from SQLite at runtime

### Refresh data only

Use the orchestrated preprocessing path for routine data refreshes:

```bash
python scripts/run_data_preprocessing.py
```

This runs, in order:

1. [`Data Preprocessing/api_fetcher.py`](Data%20Preprocessing/api_fetcher.py)
2. [`Data Preprocessing/data_pipeline.py`](Data%20Preprocessing/data_pipeline.py)
3. Supabase-backed geocode cache warm-up when `SUPABASE_DB_URL` is set
4. [`Data Preprocessing/geocoding.py`](Data%20Preprocessing/geocoding.py)
5. [`Data Preprocessing/proximity_features.py`](Data%20Preprocessing/proximity_features.py)
6. [`Database/migrate_to_supabase.py`](Database/migrate_to_supabase.py) through the orchestration wrapper

Good to know:

- geocoding and proximity steps auto-skip when the pipeline reports no DB changes
- set `HDB_BACKFILL_GEOCODING=1` or `HDB_BACKFILL_PROXIMITY=1` to force retries
- this workflow syncs processed rows to Supabase when `SUPABASE_DB_URL` is available

### Run ML only

```bash
python scripts/run_ml_pipeline.py
```

This runs:

1. [`ML/feature_engineering.py`](ML/feature_engineering.py)
2. [`ML/model_training.py`](ML/model_training.py)

Training source behavior:

- `TRAINING_DATA_SOURCE=auto` prefers Supabase when `SUPABASE_DB_URL` is set
- in `auto` mode, the code falls back to local SQLite if Supabase extraction fails
- set `TRAINING_DATA_SOURCE=supabase` if you want the ML pipeline to fail fast instead of falling back

### Sync data and the current model version to Supabase

```bash
python scripts/sync_to_supabase.py
```

This does two things:

- syncs processed SQLite data into the normalized Supabase tables
- updates `model_versions` using the run pointed to by `ML/model_assets/latest.txt`

Deployment guard:

- the sync step refuses to promote a new model if its test MAPE is more than `MAX_MAPE_REGRESSION_PCT` worse than the currently active deployed model
- default guardrail is `10%`

### Run the full retrain and deploy flow

```bash
python scripts/retrain_and_deploy.py
```

This is the end-to-end workflow:

1. preprocessing
2. Supabase sync
3. feature engineering
4. model training
5. `model_versions` update in Supabase

### Refresh reference datasets

Most developers do not need this because the repo already includes checked-in JSON files under [`Data Preprocessing/reference_data/`](Data%20Preprocessing/reference_data/).

Run it only if those files are missing or you intentionally want a refresh:

```bash
python "Data Preprocessing/fetch_reference_data.py"
```

## How Data Flows Through The System

```text
data.gov.sg raw HDB CSVs
  -> local SQLite staging DB (Data Preprocessing/hdb_resale.db)
  -> geocoding via OneMap
  -> proximity feature enrichment
  -> normalized Supabase tables
  -> feature engineering
  -> timestamped ML run directory under ML/model_assets/
  -> Flask app loads the selected run and serves predictions against Supabase data
```

The repo also contains a checked-in raw snapshot at [`Data Preprocessing/raw hdb data/`](Data%20Preprocessing/raw%20hdb%20data/), but the active fetch pipeline writes to `Data Preprocessing/raw/` by default.

## Web App Summary

The Flask app in [`webapp/app.py`](webapp/app.py) currently provides:

- public landing page, pricing page, register, login, forgot password, and reset password flows
- authenticated prediction flow with saved predictions
- interactive map and analytics dashboards backed by Supabase RPCs
- property comparison with saved-prediction shortcuts
- admin dashboard for user management, platform stats, notifications, and model inventory
- optional Gemini-powered AI helpers when `GEMINI_API_KEY` is configured

### Main page routes

Public:

- `/`
- `/home`
- `/register`
- `/login`
- `/forgot-password`
- `/reset-password`
- `/pricing`

Authenticated:

- `/predict`
- `/save_prediction`
- `/my_predictions`
- `/comparison`
- `/map`
- `/analytics`
- `/upgrade`

Admin:

- `/admin`
- `/admin/dashboard`

### Tier behavior implemented in code

General users:

- can save up to `3` predictions
- can compare up to `2` properties at once
- are limited to `3` views per 7 days for each of `map`, `analytics`, and `comparison`
- when Gemini is configured, can request up to `3` AI answers per day

Premium users:

- get unlimited saved predictions
- can compare up to `5` properties at once
- do not have the weekly page-view cap
- can use premium AI endpoints such as comparison AI analysis and AI chat

Admin login details:

- admin access is controlled by `ADMIN_EMAIL` and `ADMIN_PASSWORD` in `.env`
- the app requires those variables at startup
- admin login is handled separately from normal Supabase user login

## ML Pipeline Summary

### Feature engineering

[`ML/feature_engineering.py`](ML/feature_engineering.py) currently:

- loads the canonical training dataframe from Supabase or SQLite
- filters the training dataset to `year >= 2015`
- keeps high-price valid transactions and only drops invalid-price rows
- adds block/location features such as MRT, CBD, school, mall, hawker, and high-demand-school distances
- adds temporal features such as `flat_age`, `month_sin`, `month_cos`, and prior-year appreciation signals
- adds rolling market features such as town and town-by-flat-type medians, PSF, transaction volume, and price momentum
- merges monthly SORA data from [`ML/fetch_sora.py`](ML/fetch_sora.py)
- target-encodes `town` and `flat_model`
- writes a timestamped run directory under [`ML/model_assets/`](ML/model_assets/)

Current split strategy:

- train: everything before the validation window
- validation: the 3 months immediately before the test window
- test: the most recent 3 months

This is a fully temporal holdout, not the older random validation split described in the previous README.

Current artefacts written by feature engineering:

- `X_train.parquet`, `X_val.parquet`, `X_test.parquet`
- `y_train.parquet`, `y_val.parquet`, `y_test.parquet`
- optional `future_holdout` split files for older or alternate flows
- diagnostic `outlier_bounds.json` under `ML/model_assets/`
- `target_encoders.pkl`
- `rolling_stats_snapshot.pkl`
- `scaler.pkl` as a compatibility artefact even though scaling is currently disabled
- `feature_cols.txt`
- `run_manifest.json`
- `metrics.json` stub

### Model training

[`ML/model_training.py`](ML/model_training.py) currently trains:

- XGBoost
- LightGBM
- CatBoost
- Random Forest baseline
- a weighted CatBoost + LightGBM ensemble

Important current behavior:

- Optuna tunes XGBoost, LightGBM, and CatBoost
- winner selection uses the best base model on validation MAPE
- the ensemble is evaluated and saved, but excluded from serving-winner selection
- after winner selection, the winning base model is refit on train + validation before the final saved test metrics are updated

### How the web app chooses a model run

`webapp/app.py` resolves artefacts in this order:

1. `MODEL_ASSETS_RUN` if set and valid
2. `MODEL_USE_LATEST_TXT=true` to honor `ML/model_assets/latest.txt`
3. otherwise, the newest valid run directory under `ML/model_assets/`

This is easy to miss:

- `latest.txt` is not used automatically unless `MODEL_USE_LATEST_TXT` is enabled
- the app expects at least `scaler.pkl`, `target_encoders.pkl`, and `metrics.json`
- it then loads the winner model pickle declared by the run metadata

## Database Notes

The live schema is defined in [`Database/supabase_schema.sql`](Database/supabase_schema.sql).

At a high level the runtime database contains:

- dimension tables such as `towns`, `flat_types`, and `flat_models`
- address/location rows in `blocks`
- transaction facts in `transactions`
- app user data in `users`
- saved prediction rows in `saved_predictions`
- usage and feature limits in `feature_view_log`
- model deployment history in `model_versions`

The Flask app depends heavily on Supabase RPC functions from that SQL file for:

- map data
- analytics charts
- lookup lists for towns, blocks, models, and storeys
- prediction context and trends
- public landing-page teaser data

## Repo Structure

```text
hdb_resale/
├── Data Preprocessing/
│   ├── api_fetcher.py
│   ├── data_pipeline.py
│   ├── geocoding.py
│   ├── proximity_features.py
│   ├── fetch_reference_data.py
│   ├── reference_data/
│   ├── raw hdb data/
│   └── hdb_resale.db              # created locally by the ETL flow
├── ML/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── training_data_source.py
│   ├── fetch_sora.py
│   └── model_assets/
├── Database/
│   ├── supabase_schema.sql
│   └── migrate_to_supabase.py
├── scripts/
│   ├── pipeline_orchestration.py
│   ├── run_data_preprocessing.py
│   ├── run_ml_pipeline.py
│   ├── sync_to_supabase.py
│   └── retrain_and_deploy.py
└── webapp/
    ├── app.py
    ├── templates/
    ├── static/
    ├── requirements.txt
    └── runtime.txt
```

## External Data Sources

- data.gov.sg HDB resale datasets
- OneMap Singapore geocoding API
- LTA static train station data
- NEA hawker centre data
- MOE / data.gov.sg school data
- MAS SORA data
- Supabase PostgreSQL, REST, RPC, and Auth

## Troubleshooting

- App fails at startup with a Supabase error:
  Check `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_KEY`.

- App fails at startup asking for admin credentials:
  `ADMIN_EMAIL` and `ADMIN_PASSWORD` are required by the current code, even in local dev.

- Training unexpectedly uses SQLite:
  Set `TRAINING_DATA_SOURCE=supabase` if you want the ML pipeline to fail instead of falling back.

- Geocoding or proximity steps keep skipping:
  Set `HDB_BACKFILL_GEOCODING=1` or `HDB_BACKFILL_PROXIMITY=1`.

- Supabase sync silently warns instead of failing:
  Set `HDB_STRICT_EXTERNAL_STEPS=1`.

- `libomp.dylib` import errors on macOS for gradient boosting libraries:
  Install OpenMP with `brew install libomp`.
