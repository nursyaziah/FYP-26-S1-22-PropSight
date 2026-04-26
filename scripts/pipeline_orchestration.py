"""
pipeline_orchestration.py
=========================
Shared orchestration helpers for the HDB resale batch pipelines.

This module keeps the step logic in one place so the project can expose
focused entry scripts without duplicating pipeline code:
  - scripts/run_data_preprocessing.py
  - scripts/run_ml_pipeline.py
  - scripts/sync_to_supabase.py
  - scripts/retrain_and_deploy.py
"""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PREPROCESSING_DIR = os.path.join(PROJECT_ROOT, "Data Preprocessing")
ML_DIR = os.path.join(PROJECT_ROOT, "ML")
DATABASE_DIR = os.path.join(PROJECT_ROOT, "Database")
MODEL_ASSETS_DIR = os.environ.get("MODEL_ASSETS_DIR", os.path.join(ML_DIR, "model_assets"))
LOCAL_DB_PATH = os.environ.get(
    "HDB_SQLITE_PATH",
    os.path.join(DATA_PREPROCESSING_DIR, "hdb_resale.db"),
)
PIPELINE_RUNTIME_STATE: dict[str, dict[str, object]] = {}


def step_banner(step_num: int, total: int, description: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}/{total}: {description}")
    print(f"{'=' * 60}")


def _project_banner(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Started at: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Project root: {PROJECT_ROOT}")


def _ensure_import_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_module(module_dir: str, module_name: str):
    os.chdir(module_dir)
    _ensure_import_path(module_dir)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def _run_module_main(module_dir: str, module_name: str) -> None:
    module = _load_module(module_dir, module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module '{module_name}' has no main() function.")
    module.main()


def run_api_fetcher() -> None:
    _run_module_main(DATA_PREPROCESSING_DIR, "api_fetcher")


def run_data_pipeline() -> None:
    module = _load_module(DATA_PREPROCESSING_DIR, "data_pipeline")
    if not hasattr(module, "main"):
        raise AttributeError("Module 'data_pipeline' has no main() function.")
    module.main()

    run_info = getattr(module, "LAST_RUN_INFO", {}) or {}
    PIPELINE_RUNTIME_STATE["data_pipeline"] = dict(run_info)


def _data_pipeline_changed() -> bool | None:
    """Return whether the current run changed SQLite rows, if known."""
    run_info = PIPELINE_RUNTIME_STATE.get("data_pipeline", {})
    changed = run_info.get("db_changed")
    if isinstance(changed, bool):
        return changed
    return None


def warm_cache_from_supabase() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    except ImportError:
        pass

    supabase_db_url = os.environ.get("SUPABASE_DB_URL", "")
    if not supabase_db_url:
        print("  WARNING: SUPABASE_DB_URL not set — skipping cache warm-up (geocoding will run normally).")
        return

    try:
        import psycopg2
    except ImportError:
        print("  WARNING: psycopg2 not installed — skipping cache warm-up.")
        return

    pg_conn = pg_cur = None
    try:
        pg_conn = psycopg2.connect(supabase_db_url)
        pg_cur = pg_conn.cursor()
        pg_cur.execute(
            "SELECT full_address, latitude, longitude, dist_mrt, dist_cbd, "
            "dist_primary_school, dist_major_mall FROM blocks WHERE latitude IS NOT NULL"
        )
        rows = pg_cur.fetchall()
    except Exception as exc:
        print(f"  WARNING: Could not query Supabase blocks table: {exc}")
        return
    finally:
        if pg_cur:
            pg_cur.close()
        if pg_conn:
            pg_conn.close()

    if not rows:
        print("  WARNING: Supabase blocks table returned 0 geocoded rows — skipping cache warm-up.")
        return

    sqlite_conn = sqlite3.connect(LOCAL_DB_PATH)
    try:
        sqlite_conn.executemany(
            "INSERT OR IGNORE INTO geocode_cache "
            "(full_address, latitude, longitude, fetched_at) VALUES (?, ?, ?, ?)",
            [(r[0], r[1], r[2], "supabase-sync") for r in rows],
        )
        changes_before_update = sqlite_conn.total_changes
        sqlite_conn.executemany(
            "UPDATE resale_prices SET latitude=?, longitude=?, dist_mrt=?, "
            "dist_cbd=?, dist_primary_school=?, dist_major_mall=? "
            "WHERE full_address=? AND latitude IS NULL",
            [(r[1], r[2], r[3], r[4], r[5], r[6], r[0]) for r in rows],
        )
        updated = sqlite_conn.total_changes - changes_before_update
        sqlite_conn.commit()
    finally:
        sqlite_conn.close()

    print(f"  Warmed {len(rows)} addresses from Supabase ({updated} resale rows updated)")


def run_geocoding() -> None:
    if _data_pipeline_changed() is False and not _env_flag("HDB_BACKFILL_GEOCODING"):
        print("  Data pipeline reported no DB changes — skipping backlog geocoding.")
        print("  Set HDB_BACKFILL_GEOCODING=1 to force retries for unresolved addresses.")
        return

    os.chdir(DATA_PREPROCESSING_DIR)
    _ensure_import_path(DATA_PREPROCESSING_DIR)

    conn = sqlite3.connect(LOCAL_DB_PATH)
    pending = conn.execute(
        "SELECT COUNT(*) FROM resale_prices WHERE latitude IS NULL"
    ).fetchone()[0]
    conn.close()

    if pending == 0:
        print("  All rows already geocoded — skipping.")
        return

    print(f"  {pending:,} rows pending geocoding.")
    import geocoding

    geocoding.main()


def run_proximity_features() -> None:
    if _data_pipeline_changed() is False and not _env_flag("HDB_BACKFILL_PROXIMITY"):
        print("  Data pipeline reported no DB changes — skipping backlog proximity features.")
        print("  Set HDB_BACKFILL_PROXIMITY=1 to force recomputation for pending rows.")
        return

    os.chdir(DATA_PREPROCESSING_DIR)
    _ensure_import_path(DATA_PREPROCESSING_DIR)

    conn = sqlite3.connect(LOCAL_DB_PATH)
    pending = conn.execute(
        "SELECT COUNT(*) FROM resale_prices "
        "WHERE latitude IS NOT NULL AND dist_mrt IS NULL"
    ).fetchone()[0]
    conn.close()

    if pending == 0:
        print("  All geocoded rows already have proximity features — skipping.")
        return

    print(f"  {pending:,} rows pending proximity features.")
    import proximity_features

    proximity_features.main()


def run_feature_engineering() -> None:
    _run_module_main(ML_DIR, "feature_engineering")


def run_model_training() -> None:
    # Auto-trigger fresh Optuna tuning once a month so hyperparameters stay
    # current as the market evolves.  We check the timestamp of the most recent
    # Optuna study file; if it is older than FRESH_TUNING_INTERVAL_DAYS days we
    # set FRESH_TUNING=1 for this run only.
    FRESH_TUNING_INTERVAL_DAYS = int(
        os.environ.get("FRESH_TUNING_INTERVAL_DAYS", "30")
    )
    if not _env_flag("FRESH_TUNING"):
        study_path = os.path.join(
            MODEL_ASSETS_DIR, "optuna_study_catboost.pkl"
        )
        if os.path.exists(study_path):
            age_days = (
                time.time() - os.path.getmtime(study_path)
            ) / 86400
            if age_days >= FRESH_TUNING_INTERVAL_DAYS:
                print(
                    f"  Optuna studies are {age_days:.0f} days old "
                    f"(>= {FRESH_TUNING_INTERVAL_DAYS}d threshold) — "
                    f"enabling FRESH_TUNING for this run."
                )
                os.environ["FRESH_TUNING"] = "1"
    _run_module_main(ML_DIR, "model_training")


def _load_project_env() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("  ERROR: python-dotenv not installed. Skipping Supabase sync.")
        return ""

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    return os.environ.get("SUPABASE_DB_URL", "")


def run_supabase_data_sync() -> None:
    supabase_db_url = _load_project_env()
    if not supabase_db_url:
        print("  WARNING: SUPABASE_DB_URL not set in .env — skipping Supabase data sync.")
        return

    try:
        import psycopg2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "psycopg2 is required to sync processed data to Supabase."
        ) from exc

    os.chdir(DATABASE_DIR)
    _ensure_import_path(DATABASE_DIR)

    import migrate_to_supabase

    try:
        migrate_to_supabase.migrate()
    except Exception as exc:
        if _env_flag("HDB_STRICT_EXTERNAL_STEPS"):
            raise
        print(f"  WARNING: Supabase data sync skipped after external error: {exc}")


class ModelPromotionBlocked(RuntimeError):
    """Raised when a new model should not be promoted to production."""


def _skip_model_version_sync(message: str) -> None:
    if _env_flag("HDB_STRICT_EXTERNAL_STEPS"):
        raise RuntimeError(message)
    print(f"  WARNING: {message}")


def run_supabase_model_version_sync(
    trigger_label: str = "scripts/sync_to_supabase.py",
    dry_run: bool = False,
) -> None:
    supabase_db_url = _load_project_env()
    if not supabase_db_url:
        _skip_model_version_sync("SUPABASE_DB_URL not set in .env; skipping model_versions update.")
        return

    try:
        import psycopg2
    except ImportError:
        _skip_model_version_sync("psycopg2 not installed; skipping model_versions update.")
        return

    run_dir = get_latest_run_dir()
    if run_dir is None:
        _skip_model_version_sync("Could not find latest run directory; skipping model_versions update.")
        return

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        _skip_model_version_sync(f"No metrics.json in {run_dir}; skipping model_versions update.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    winner = metrics.get("winner", {})
    winner_name = winner.get("winner", "unknown")
    test_mape = winner.get("test_mape")
    test_rmse = winner.get("test_rmse")
    test_r2 = winner.get("test_r2")

    mape_regression_threshold = float(os.environ.get("MAX_MAPE_REGRESSION_PCT", "10")) / 100

    try:
        pg_conn = psycopg2.connect(supabase_db_url)
        pg_cur = pg_conn.cursor()

        # Guard: only promote if new MAPE is not more than threshold% worse than deployed.
        if test_mape is not None:
            pg_cur.execute(
                "SELECT test_mape FROM model_versions WHERE is_active = TRUE ORDER BY trained_at DESC LIMIT 1"
            )
            current_row = pg_cur.fetchone()
            if current_row and current_row[0] is not None:
                current_mape = float(current_row[0])
                if test_mape > current_mape * (1 + mape_regression_threshold):
                    message = (
                        f"  ABORT: New model MAPE ({test_mape:.4f}%) is more than "
                        f"{mape_regression_threshold * 100:.0f}% worse than the deployed model "
                        f"({current_mape:.4f}%). Skipping deployment to protect production."
                    )
                    print(message)
                    raise ModelPromotionBlocked(message)

        if dry_run:
            metric_label = f"{float(test_mape):.4f}%" if test_mape is not None else "unknown MAPE"
            print(f"  model promotion guard passed: {winner_name} ({metric_label})")
            return

        pg_cur.execute("UPDATE model_versions SET is_active = FALSE WHERE is_active = TRUE")
        pg_cur.execute(
            """
            INSERT INTO model_versions (version, run_dir, test_mape, test_rmse, test_r2, notes, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            """,
            (
                winner_name,
                os.path.basename(run_dir),
                test_mape,
                test_rmse,
                test_r2,
                f"Deployed via {trigger_label} at {datetime.now(timezone.utc).isoformat()}",
            ),
        )
        pg_conn.commit()
        print(f"  model_versions updated: {winner_name} (active)")
    except ModelPromotionBlocked:
        raise
    except Exception as exc:
        if _env_flag("HDB_STRICT_EXTERNAL_STEPS"):
            raise
        print(f"  WARNING: model_versions update skipped after external error: {exc}")
    finally:
        if "pg_cur" in locals():
            pg_cur.close()
        if "pg_conn" in locals():
            pg_conn.close()


def get_latest_run_dir() -> str | None:
    latest_path = os.path.join(MODEL_ASSETS_DIR, "latest.txt")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path) as f:
        run_dir = f.read().strip()

    if not os.path.isabs(run_dir):
        run_dir = os.path.join(ML_DIR, run_dir)
    if not os.path.isdir(run_dir):
        return None
    return run_dir


def _get_db_counts() -> dict[str, int] | None:
    if not os.path.exists(LOCAL_DB_PATH):
        return None

    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        counts = {
            "total_rows": conn.execute(
                "SELECT COUNT(*) FROM resale_prices"
            ).fetchone()[0],
            "geocoded_rows": conn.execute(
                "SELECT COUNT(*) FROM resale_prices WHERE latitude IS NOT NULL"
            ).fetchone()[0],
            "proximity_rows": conn.execute(
                "SELECT COUNT(*) FROM resale_prices "
                "WHERE dist_mrt IS NOT NULL AND dist_cbd IS NOT NULL "
                "AND dist_primary_school IS NOT NULL AND dist_major_mall IS NOT NULL"
            ).fetchone()[0],
        }
        conn.close()
        return counts
    except sqlite3.OperationalError:
        return None


def print_data_preprocessing_summary() -> None:
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING SUMMARY")
    print("=" * 60)

    counts = _get_db_counts()
    if counts is None:
        print("  Local SQLite DB not found.")
    else:
        print(f"  Total rows in DB:      {counts['total_rows']:,}")
        print(f"  Geocoded rows:         {counts['geocoded_rows']:,}")
        print(f"  Rows with proximity:   {counts['proximity_rows']:,}")

    print(f"\n  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


def print_ml_summary() -> None:
    print("\n" + "=" * 60)
    print("  ML PIPELINE SUMMARY")
    print("=" * 60)

    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("  No latest ML run directory found.")
    else:
        print(f"  Run directory: {os.path.basename(run_dir)}")
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            winner = metrics.get("winner", {})
            print(f"  Model winner: {winner.get('winner', 'N/A')}")
            print(f"  Val RMSE:     {winner.get('val_rmse', 'N/A')}")
            print(f"  Test RMSE:    {winner.get('test_rmse', 'N/A')}")
            print(f"  Test R²:      {winner.get('test_r2', 'N/A')}")
            print(f"  Test MAPE:    {winner.get('test_mape', 'N/A')}%")

    print(f"\n  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


def print_deploy_summary() -> None:
    print("\n" + "=" * 60)
    print("  SUPABASE DEPLOY SUMMARY")
    print("=" * 60)

    counts = _get_db_counts()
    if counts is not None:
        print(f"  Local DB rows available for deploy: {counts['total_rows']:,}")

    run_dir = get_latest_run_dir()
    if run_dir is not None:
        print(f"  Latest ML run: {os.path.basename(run_dir)}")

    print(f"\n  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


def print_full_summary() -> None:
    print("\n" + "=" * 60)
    print("  RETRAIN & DEPLOY SUMMARY")
    print("=" * 60)

    counts = _get_db_counts()
    if counts is not None:
        print(f"  Total rows in DB: {counts['total_rows']:,}")

    run_dir = get_latest_run_dir()
    if run_dir is not None:
        print(f"  Run directory: {os.path.basename(run_dir)}")
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            winner = metrics.get("winner", {})
            print(f"  Model winner: {winner.get('winner', 'N/A')}")
            print(f"  Test RMSE:    {winner.get('test_rmse', 'N/A')}")
            print(f"  Test R²:      {winner.get('test_r2', 'N/A')}")
            print(f"  Test MAPE:    {winner.get('test_mape', 'N/A')}%")

    print(f"\n  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


PREPROCESSING_STEPS = [
    ("Fetch latest data from HDB API", run_api_fetcher),
    ("Data cleaning & preprocessing", run_data_pipeline),
    ("Warm geocode cache from Supabase", warm_cache_from_supabase),
    ("Geocoding addresses", run_geocoding),
    ("Computing proximity features", run_proximity_features),
    ("Syncing processed data to Supabase", run_supabase_data_sync),
]

ML_STEPS = [
    ("Feature engineering", run_feature_engineering),
    ("Model training", run_model_training),
]


def run_pipeline(
    title: str,
    steps: list[tuple[str, Callable[[], None]]],
    summary_callback: Callable[[], None] | None = None,
) -> int:
    PIPELINE_RUNTIME_STATE.clear()
    _project_banner(title)
    t_start = time.time()

    try:
        total_steps = len(steps)
        for idx, (description, func) in enumerate(steps, start=1):
            step_banner(idx, total_steps, description)
            func()
    except Exception as exc:
        print(f"\n  FATAL ERROR: {exc}")
        traceback.print_exc()
        return 1

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if summary_callback is not None:
        summary_callback()
    return 0


def run_data_preprocessing_pipeline() -> int:
    return run_pipeline(
        "HDB Resale — Data Preprocessing Pipeline",
        PREPROCESSING_STEPS,
        print_data_preprocessing_summary,
    )


def run_ml_pipeline() -> int:
    return run_pipeline(
        "HDB Resale — ML Pipeline",
        ML_STEPS,
        print_ml_summary,
    )


def run_supabase_sync_pipeline() -> int:
    return run_pipeline(
        "HDB Resale — Supabase Sync",
        [
            ("Syncing processed data to Supabase", run_supabase_data_sync),
            (
                "Updating deployed model version in Supabase",
                lambda: run_supabase_model_version_sync("scripts/sync_to_supabase.py"),
            ),
        ],
        print_deploy_summary,
    )


def run_full_pipeline() -> int:
    full_steps = list(PREPROCESSING_STEPS) + list(ML_STEPS) + [
        (
            "Updating deployed model version in Supabase",
            lambda: run_supabase_model_version_sync("scripts/retrain_and_deploy.py"),
        ),
    ]
    return run_pipeline(
        "HDB Resale — Retrain & Deploy Pipeline",
        full_steps,
        print_full_summary,
    )
