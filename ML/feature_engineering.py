import json
import os
import pickle
import time
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training_data_source import load_training_dataframe
from fetch_sora import load_sora_monthly


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("MODEL_ASSETS_DIR", os.path.join(BASE_DIR, "model_assets"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

QUANTILES = [0.10, 0.50, 0.90]

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "HOUGANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON", "TAMPINES",
    "TOA PAYOH",
}

FLAT_TYPE_ORDINAL = {
    "1 Room": 1,
    "2 Room": 2,
    "3 Room": 3,
    "4 Room": 4,
    "5 Room": 5,
    "Executive": 6,
    "Multi-Generation": 7,
}

SCALE_COLS = [
    "floor_area_sqm",
    "storey_midpoint",
    "flat_age",
    "remaining_lease",
    "lease_commence_date",
    "month_sin",
    "month_cos",
    "year",
    "dist_mrt",
    "dist_cbd",
    "dist_primary_school",
    "dist_major_mall",
    "dist_hawker_centre",
    "hawker_count_1km",
    "dist_high_demand_primary_school",
    "high_demand_primary_count_1km",
    "town_yoy_appreciation_lag1",
    "town_5yr_cagr_lag1",
    "sora_3m",
    "sora_3m_change_3m",
]

FEATURE_COLS = [
    "flat_type_ordinal",
    "town_enc",
    "flat_model_enc",
    "floor_area_sqm",
    "storey_midpoint",
    "flat_age",
    "remaining_lease",
    "lease_commence_date",
    "month_sin",
    "month_cos",
    "year",
    "is_mature_estate",
    "dist_mrt",
    "dist_cbd",
    "dist_primary_school",
    "dist_major_mall",
    "dist_hawker_centre",
    "hawker_count_1km",
    "dist_high_demand_primary_school",
    "high_demand_primary_count_1km",
    "town_yoy_appreciation_lag1",
    "town_5yr_cagr_lag1",
    # Rolling comparable features — must be computed before temporal split
    "town_flattype_median_1m",
    "town_flattype_median_3m",
    "town_flattype_median_6m",
    "town_flattype_psf_1m",
    "town_flattype_psf_3m",
    "town_median_3m",
    "town_txn_volume_3m",
    "price_momentum_3m",
    "psf_change_1m_vs_3m",
    "national_median_psf_1m",
    "national_median_psf_3m",
    "sora_3m",
    "sora_3m_change_3m",
]

TARGET_COL = "log_price"

REQUIRED_MODEL_COLS = [
    "town",
    "flat_type",
    "flat_model",
    "month_num",
    "year",
    "floor_area_sqm",
    "storey_midpoint",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
    "dist_mrt",
    "dist_cbd",
    "dist_primary_school",
    "dist_major_mall",
    "dist_hawker_centre",
    "hawker_count_1km",
    "dist_high_demand_primary_school",
    "high_demand_primary_count_1km",
    "sora_3m",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, str]:
    print("Loading training data ...", flush=True)
    t0 = time.time()
    df, data_source = load_training_dataframe()
    print(f"  Source: {data_source}")
    print(f"  Source query completed in {time.time() - t0:.1f}s.")

    numeric_cols = [
        "floor_area_sqm",
        "storey_midpoint",
        "remaining_lease",
        "remaining_lease_months",
        "lease_commence_date",
        "resale_price",
        "year",
        "month_num",
        "latitude",
        "longitude",
        "dist_mrt",
        "dist_cbd",
        "dist_primary_school",
        "dist_major_mall",
        "dist_hawker_centre",
        "hawker_count_1km",
        "dist_high_demand_primary_school",
        "high_demand_primary_count_1km",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["town", "flat_type", "flat_model"]:
        # Preserve missing values so required-field cleanup can drop them later.
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    print(f"  Loaded {len(df):,} rows.")

    min_year = 2015
    before = len(df)
    df = df[df["year"] >= min_year].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Year filter (>= {min_year}): dropped {dropped:,} pre-{min_year} rows. Remaining: {len(df):,}")

    return df, data_source


# ---------------------------------------------------------------------------
# Outlier diagnostics / conservative cleanup
# ---------------------------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    bounds: dict[str, dict[str, float]] = {}

    for flat_type, idx in df.groupby("flat_type").groups.items():
        prices = df.loc[idx, "resale_price"]
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bounds[str(flat_type)] = {
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2),
        }

    # Keep premium but valid transactions in training. With a log-price target,
    # a simple flat_type IQR fence is too aggressive and can remove legitimate
    # high-end sales that the model needs to learn from.
    keep_mask = df["resale_price"] > 0
    df = df[keep_mask].copy()

    dropped = before - len(df)
    print("\nOutlier handling: IQR bounds saved for diagnostics only.")
    print("  High-price transactions are retained to preserve premium-market signal.")
    print(
        f"  Invalid-price cleanup: dropped {dropped:,} rows "
        f"({dropped / before * 100:.1f}%). Remaining: {len(df):,}"
    )

    bounds_path = os.path.join(OUTPUT_DIR, "outlier_bounds.json")
    with open(bounds_path, "w") as f:
        json.dump(bounds, f, indent=2)

    print(f"  outlier_bounds.json saved to {bounds_path}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    print("\nEngineering features ...")

    df["flat_age"] = (df["year"] - df["lease_commence_date"]).clip(lower=0)

    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df["is_mature_estate"] = df["town"].isin(MATURE_ESTATES).astype(int)

    df["flat_type_ordinal"] = df["flat_type"].map(FLAT_TYPE_ORDINAL)

    unmapped = df["flat_type_ordinal"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped:,} unmapped flat_type rows — filling with median.")
        df["flat_type_ordinal"] = df["flat_type_ordinal"].fillna(
            df["flat_type_ordinal"].median()
        )

    # Use a direct log-price target so holdout rows do not depend on a
    # market index estimated from future transactions.
    df[TARGET_COL] = np.log1p(df["resale_price"])

    # Historical trend signals are computed from prior completed years only,
    # so each row only sees market information that would have been available
    # at prediction time. Use town + flat_type granularity to better capture
    # segment-specific price momentum.
    #
    # Exclude the last 6 months (val + test window) from the yearly average
    # so no future transactions contaminate the appreciation signal.
    # A placeholder row for the current year is then injected so that
    # shift(1) carries the most recent clean annual average forward to
    # current-year rows, giving them a real signal instead of 0.0.
    _max_ym = int((df["year"] * 100 + df["month_num"]).max())
    _max_ym_year, _max_ym_month = _max_ym // 100, _max_ym % 100
    _cutoff_ym = (
        _max_ym_year * 100 + (_max_ym_month - 6)
        if _max_ym_month > 6
        else (_max_ym_year - 1) * 100 + (_max_ym_month + 6)
    )
    _yearly_avg_df = df[(df["year"] * 100 + df["month_num"]) <= _cutoff_ym]
    yearly_avg = (
        _yearly_avg_df.groupby(["town", "flat_type", "year"], as_index=False)["resale_price"]
        .mean()
        .rename(columns={"resale_price": "avg_resale_price"})
    )
    # Inject a placeholder for the current year when the cutoff excluded it
    # entirely, so shift(1) can propagate the last clean avg to current-year rows.
    if _max_ym_year not in yearly_avg["year"].values:
        _placeholder = (
            yearly_avg[["town", "flat_type"]]
            .drop_duplicates()
            .assign(year=_max_ym_year, avg_resale_price=np.nan)
        )
        yearly_avg = (
            pd.concat([yearly_avg, _placeholder], ignore_index=True)
            .sort_values(["town", "flat_type", "year"])
            .reset_index(drop=True)
        )
    yearly_avg["prev_avg_1y"] = yearly_avg.groupby(["town", "flat_type"])["avg_resale_price"].shift(1)
    yearly_avg["prev_avg_2y"] = yearly_avg.groupby(["town", "flat_type"])["avg_resale_price"].shift(2)
    yearly_avg["prev_avg_6y"] = yearly_avg.groupby(["town", "flat_type"])["avg_resale_price"].shift(6)

    yearly_avg["town_yoy_appreciation_lag1"] = np.where(
        yearly_avg["prev_avg_2y"].gt(0),
        (yearly_avg["prev_avg_1y"] - yearly_avg["prev_avg_2y"]) / yearly_avg["prev_avg_2y"],
        np.nan,
    )
    yearly_avg["town_5yr_cagr_lag1"] = np.where(
        yearly_avg["prev_avg_6y"].gt(0) & yearly_avg["prev_avg_1y"].gt(0),
        np.power(yearly_avg["prev_avg_1y"] / yearly_avg["prev_avg_6y"], 1 / 5) - 1,
        np.nan,
    )

    df = df.merge(
        yearly_avg[
            [
                "town",
                "flat_type",
                "year",
                "town_yoy_appreciation_lag1",
                "town_5yr_cagr_lag1",
            ]
        ],
        on=["town", "flat_type", "year"],
        how="left",
    )
    df["town_yoy_appreciation_lag1"] = df["town_yoy_appreciation_lag1"].fillna(0.0)
    df["town_5yr_cagr_lag1"] = df["town_5yr_cagr_lag1"].fillna(0.0)

    print(
        "  Features added: flat_age, month_sin, month_cos, "
        "is_mature_estate, flat_type_ordinal, "
        "town_yoy_appreciation_lag1, town_5yr_cagr_lag1, log_price"
    )

    df = add_rolling_market_features(df)

    # ---- SORA integration -----------------------------------------------
    sora_df = load_sora_monthly()
    sora_df = sora_df.sort_values("year_month").reset_index(drop=True)
    sora_df["sora_3m_change_3m"] = sora_df["sora_3m"] - sora_df["sora_3m"].shift(3)
    sora_df["sora_3m_change_3m"] = sora_df["sora_3m_change_3m"].fillna(0.0)
    df["_sora_ym"] = df["year"] * 100 + df["month_num"]
    df = df.merge(sora_df, left_on="_sora_ym", right_on="year_month", how="left")
    sora_median = df["sora_3m"].dropna().median()
    if pd.isna(sora_median):
        sora_median = 0.0
    df["sora_3m"] = df["sora_3m"].fillna(sora_median)
    df["sora_3m_change_3m"] = df["sora_3m_change_3m"].fillna(0.0)
    df = df.drop(columns=["_sora_ym", "year_month"], errors="ignore")
    print(f"  sora_3m null count after fill: {df['sora_3m'].isna().sum():,}")

    return df


# ---------------------------------------------------------------------------
# Rolling comparable sale features (computed before temporal split)
# ---------------------------------------------------------------------------

def add_rolling_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling comparable sale features on the FULL dataset before any
    temporal split.  For each feature:
      1. Aggregate individual rows to monthly-level per group.
      2. Apply a rolling window over those monthly buckets.
      3. Shift by 1 month so each row only sees data up to the previous month.
      4. Merge back to individual rows by (group keys, year_month).

    This avoids rolling over individual transactions (slow and semantically
    wrong) and prevents look-ahead leakage.
    """
    print("\nComputing rolling market features (pre-split, full dataset) ...")

    df = df.sort_values(["year", "month_num"]).reset_index(drop=True)
    df["year_month"] = df["year"] * 100 + df["month_num"]
    df["_psf"] = df["resale_price"] / df["floor_area_sqm"]

    def _fill_nan(s: pd.Series) -> pd.Series:
        """Fill NaN with expanding mean within the group, then 0.0."""
        filled = s.fillna(s.expanding(min_periods=1).mean())
        return filled.fillna(0.0)

    # ---- Features 1 & 2: town_flattype_median_3m / 6m --------------------
    tf_monthly = (
        df.groupby(["town", "flat_type", "year_month"])["resale_price"]
        .median()
        .reset_index()
        .rename(columns={"resale_price": "tf_med"})
        .sort_values(["town", "flat_type", "year_month"])
    )
    tf_monthly["town_flattype_median_3m"] = (
        tf_monthly.groupby(["town", "flat_type"])["tf_med"]
        .transform(lambda x: x.rolling(3, min_periods=1).median().shift(1))
    )
    tf_monthly["town_flattype_median_6m"] = (
        tf_monthly.groupby(["town", "flat_type"])["tf_med"]
        .transform(lambda x: x.rolling(6, min_periods=1).median().shift(1))
    )
    tf_monthly["town_flattype_median_1m"] = (
        tf_monthly.groupby(["town", "flat_type"])["tf_med"]
        .transform(lambda x: x.rolling(1, min_periods=1).median().shift(1))
    )
    tf_monthly["town_flattype_median_1m"] = (
        tf_monthly.groupby(["town", "flat_type"])["town_flattype_median_1m"].transform(_fill_nan)
    )
    for col in ["town_flattype_median_3m", "town_flattype_median_6m"]:
        tf_monthly[col] = (
            tf_monthly.groupby(["town", "flat_type"])[col].transform(_fill_nan)
        )

    # ---- Feature 3: town_flattype_psf_3m ---------------------------------
    tf_psf_monthly = (
        df.groupby(["town", "flat_type", "year_month"])["_psf"]
        .median()
        .reset_index()
        .rename(columns={"_psf": "tf_psf_med"})
        .sort_values(["town", "flat_type", "year_month"])
    )
    tf_psf_monthly["town_flattype_psf_3m"] = (
        tf_psf_monthly.groupby(["town", "flat_type"])["tf_psf_med"]
        .transform(lambda x: x.rolling(3, min_periods=1).median().shift(1))
    )
    tf_psf_monthly["town_flattype_psf_3m"] = (
        tf_psf_monthly.groupby(["town", "flat_type"])["town_flattype_psf_3m"]
        .transform(_fill_nan)
    )
    tf_psf_monthly["town_flattype_psf_1m"] = (
        tf_psf_monthly.groupby(["town", "flat_type"])["tf_psf_med"]
        .transform(lambda x: x.rolling(1, min_periods=1).median().shift(1))
    )
    tf_psf_monthly["town_flattype_psf_1m"] = (
        tf_psf_monthly.groupby(["town", "flat_type"])["town_flattype_psf_1m"].transform(_fill_nan)
    )

    # Merge all three town×flat_type features at once
    tf_lookup = tf_monthly[
        ["town", "flat_type", "year_month",
         "town_flattype_median_1m", "town_flattype_median_3m", "town_flattype_median_6m"]
    ].merge(
        tf_psf_monthly[
            ["town", "flat_type", "year_month", "town_flattype_psf_1m", "town_flattype_psf_3m"]
        ],
        on=["town", "flat_type", "year_month"],
        how="left",
    )
    df = df.merge(tf_lookup, on=["town", "flat_type", "year_month"], how="left")

    df["psf_change_1m_vs_3m"] = np.where(
        df["town_flattype_psf_3m"] > 0,
        (df["town_flattype_psf_1m"] - df["town_flattype_psf_3m"]) / df["town_flattype_psf_3m"],
        0.0,
    )
    df["psf_change_1m_vs_3m"] = (
        df["psf_change_1m_vs_3m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # ---- Feature 4: town_median_3m ----------------------------------------
    t_monthly = (
        df.groupby(["town", "year_month"])["resale_price"]
        .median()
        .reset_index()
        .rename(columns={"resale_price": "t_med"})
        .sort_values(["town", "year_month"])
    )
    t_monthly["town_median_3m"] = (
        t_monthly.groupby("town")["t_med"]
        .transform(lambda x: x.rolling(3, min_periods=1).median().shift(1))
    )
    t_monthly["town_median_3m"] = (
        t_monthly.groupby("town")["town_median_3m"].transform(_fill_nan)
    )
    df = df.merge(
        t_monthly[["town", "year_month", "town_median_3m"]],
        on=["town", "year_month"],
        how="left",
    )

    # ---- Feature 5: town_txn_volume_3m ------------------------------------
    t_vol_monthly = (
        df.groupby(["town", "year_month"])["resale_price"]
        .count()
        .reset_index()
        .rename(columns={"resale_price": "t_count"})
        .sort_values(["town", "year_month"])
    )
    t_vol_monthly["town_txn_volume_3m"] = (
        t_vol_monthly.groupby("town")["t_count"]
        .transform(lambda x: x.rolling(3, min_periods=1).sum().shift(1))
    )
    t_vol_monthly["town_txn_volume_3m"] = (
        t_vol_monthly.groupby("town")["town_txn_volume_3m"].transform(_fill_nan)
    )
    df = df.merge(
        t_vol_monthly[["town", "year_month", "town_txn_volume_3m"]],
        on=["town", "year_month"],
        how="left",
    )

    # ---- Feature 6: price_momentum_3m -------------------------------------
    df["price_momentum_3m"] = (
        (df["town_flattype_median_3m"] - df["town_flattype_median_6m"])
        / df["town_flattype_median_6m"]
    )
    df["price_momentum_3m"] = (
        df["price_momentum_3m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # ---- Feature 7: national_median_psf_3m --------------------------------
    nat_monthly = (
        df.groupby("year_month")["_psf"]
        .median()
        .reset_index()
        .rename(columns={"_psf": "nat_psf_med"})
        .sort_values("year_month")
    )
    nat_monthly["national_median_psf_3m"] = (
        nat_monthly["nat_psf_med"].rolling(3, min_periods=1).median().shift(1)
    )
    nat_monthly["national_median_psf_3m"] = (
        nat_monthly["national_median_psf_3m"]
        .fillna(nat_monthly["national_median_psf_3m"].expanding(min_periods=1).mean())
        .fillna(0.0)
    )
    nat_monthly["national_median_psf_1m"] = (
        nat_monthly["nat_psf_med"].rolling(1, min_periods=1).median().shift(1)
    )
    nat_monthly["national_median_psf_1m"] = (
        nat_monthly["national_median_psf_1m"]
        .fillna(nat_monthly["national_median_psf_1m"].expanding(min_periods=1).mean())
        .fillna(0.0)
    )
    df = df.merge(
        nat_monthly[["year_month", "national_median_psf_1m", "national_median_psf_3m"]],
        on="year_month",
        how="left",
    )

    # Drop temporary columns
    df = df.drop(columns=["_psf", "year_month"])

    # Null summary
    new_features = [
        "town_flattype_median_1m",
        "town_flattype_median_3m",
        "town_flattype_median_6m",
        "town_flattype_psf_1m",
        "town_flattype_psf_3m",
        "town_median_3m",
        "town_txn_volume_3m",
        "price_momentum_3m",
        "psf_change_1m_vs_3m",
        "national_median_psf_1m",
        "national_median_psf_3m",
    ]
    print("  Rolling feature null counts after filling:")
    for feat in new_features:
        print(f"    {feat}: {df[feat].isna().sum():,} nulls")

    return df


# ---------------------------------------------------------------------------
# Remove rows with missing model inputs
# ---------------------------------------------------------------------------

def drop_missing_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=REQUIRED_MODEL_COLS).copy()
    dropped = before - len(df)

    print(
        f"\nMissing-value cleanup: dropped {dropped:,} rows with missing "
        f"required model fields. Remaining: {len(df):,}"
    )
    return df


# Temporal split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    test_months: int = 3,
    val_months: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Split data using a fully temporal strategy:
    - Test set:  most recent `test_months` months
    - Val set:   `val_months` months immediately before the test cutoff
    - Train set: everything before the val cutoff
    No random sampling anywhere; all boundaries are date-based.
    """
    df["_ym"] = df["year"] * 100 + df["month_num"]
    unique_ym = sorted(df["_ym"].dropna().unique())

    if len(unique_ym) <= test_months + val_months:
        raise ValueError(
            f"Not enough distinct months ({len(unique_ym)}) to carve out "
            f"test ({test_months}) + val ({val_months}) windows."
        )

    test_cutoff_ym = unique_ym[-test_months]
    val_cutoff_ym  = unique_ym[-(test_months + val_months)]

    test  = df[df["_ym"] >= test_cutoff_ym].copy()
    val   = df[(df["_ym"] >= val_cutoff_ym) & (df["_ym"] < test_cutoff_ym)].copy()
    train = df[df["_ym"] < val_cutoff_ym].copy()

    for split in (train, val, test):
        split.drop(columns=["_ym"], inplace=True)
    df.drop(columns=["_ym"], inplace=True)

    def ym_str(ym: int) -> str:
        return f"{ym // 100}-{ym % 100:02d}"

    total     = len(df)
    train_end = unique_ym[-(test_months + val_months) - 1]
    val_end   = unique_ym[-test_months - 1]
    test_end  = unique_ym[-1]

    print(f"\nFully temporal split (total {total:,} rows):")
    print(f"  Train : {len(train):>8,}  ({len(train)/total*100:.1f}%)  {ym_str(unique_ym[0])} – {ym_str(train_end)}")
    print(f"  Val   : {len(val):>8,}  ({len(val)/total*100:.1f}%)  {ym_str(val_cutoff_ym)} – {ym_str(val_end)}")
    print(f"  Test  : {len(test):>8,}  ({len(test)/total*100:.1f}%)  {ym_str(test_cutoff_ym)} – {ym_str(test_end)}")

    return {"train": train, "val": val, "test": test}


def build_split_metadata(
    splits: dict[str, pd.DataFrame],
) -> dict[str, int | str | None]:
    train = splits["train"]
    val   = splits["val"]
    test  = splits["test"]

    def ym_range(split: pd.DataFrame) -> tuple[int | None, int | None]:
        if split.empty:
            return None, None
        ym = split["year"] * 100 + split["month_num"]
        return int(ym.min()), int(ym.max())

    train_ym_min, train_ym_max = ym_range(train)
    val_ym_min,   val_ym_max   = ym_range(val)
    test_ym_min,  test_ym_max  = ym_range(test)

    return {
        "train_rows": len(train),
        "train_min_year": int(train["year"].min()) if not train.empty else None,
        "train_max_year": int(train["year"].max()) if not train.empty else None,
        "train_ym_range": [train_ym_min, train_ym_max],
        "val_rows": len(val),
        "val_min_year": int(val["year"].min()) if not val.empty else None,
        "val_max_year": int(val["year"].max()) if not val.empty else None,
        "val_ym_range": [val_ym_min, val_ym_max],
        "test_rows": len(test),
        "test_min_year": int(test["year"].min()) if not test.empty else None,
        "test_max_year": int(test["year"].max()) if not test.empty else None,
        "test_ym_range": [test_ym_min, test_ym_max],
        "split_strategy": "rolling_temporal_holdout",
    }


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

def target_encode(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Time-aware target encoding for town and flat_model.

    Train rows: each row's encoding is the expanding mean of log_price for
    that category using only chronologically prior rows (groupby + expanding
    + shift(1)).  This prevents a 2005 row from absorbing 2020 price levels.

    Val / test rows: encoded with the snapshot map built from the entire
    training period (mean of all train rows per category), which is what
    would be known at the train cutoff date.

    NaN handling (first occurrence of a category in train):
      1. Fall back to the global expanding mean at that point in time.
      2. If that is also NaN (very first row overall), fall back to the
         overall training mean.
    """
    print("\nTarget encoding town and flat_model (time-aware expanding mean) ...")

    # Sort train chronologically so expanding() operates in time order.
    train = splits["train"].sort_values(["year", "month_num"]).copy()

    global_mean = float(train[TARGET_COL].mean())

    # Global expanding mean shifted by 1: used as fallback for group-NaNs.
    global_expanding = train[TARGET_COL].expanding().mean().shift(1)

    encoders = {}

    for col in ["town", "flat_model"]:
        enc_col = f"{col}_enc"

        # Within each category, compute the expanding mean of all prior rows.
        # sort=False preserves the temporal order established above.
        group_expanding = (
            train.groupby(col, sort=False)[TARGET_COL]
            .transform(lambda x: x.expanding().mean().shift(1))
        )

        # Fill NaN in order of priority:
        #   1st occurrence of a category → use global expanding at that point.
        #   Very first row overall        → fall back to whole-train global mean.
        train[enc_col] = (
            group_expanding
            .fillna(global_expanding)
            .fillna(global_mean)
        )

        # Snapshot map for val / test: mean of all training rows per category.
        # This is the encoding any new row would receive after training ends.
        final_map = train.groupby(col)[TARGET_COL].mean()
        encoders[col] = {
            "means": final_map,
            "global_mean": global_mean,
        }

        for split_name, split in splits.items():
            if split_name == "train":
                continue
            split[enc_col] = split[col].map(final_map).fillna(global_mean)

        print(f"  {col}: {len(final_map)} categories  "
              f"(train: expanding prior-row mean, val/test: end-of-train snapshot)")

    splits["train"] = train
    return splits, encoders


# ---------------------------------------------------------------------------
# Scale numeric features
# ---------------------------------------------------------------------------

def scale_features(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], StandardScaler]:
    print("\nScaling numeric features ...")

    scaler = StandardScaler()
    train = splits["train"]

    train[SCALE_COLS] = scaler.fit_transform(train[SCALE_COLS])
    for name, split in splits.items():
        if name == "train":
            continue
        split[SCALE_COLS] = scaler.transform(split[SCALE_COLS])

    applied_to = ", ".join(name for name in splits if name != "train")
    print(f"  StandardScaler fitted on train, applied to {applied_to}.")
    return splits, scaler


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    splits: dict[str, pd.DataFrame],
    target_encoders: dict,
    split_metadata: dict[str, int | None],
    data_source: str,
) -> str:
    run_dir = os.path.join(
        OUTPUT_DIR,
        datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nSaving artefacts to '{run_dir}/' ...")

    for name in ("train", "val", "test", "future_holdout"):
        if name not in splits:
            continue
        split = splits[name]
        X = split[FEATURE_COLS]
        y_cols = [TARGET_COL, "resale_price"]
        if name == "train" and "year" in split.columns and "month_num" in split.columns:
            split = split.copy()
            split["year_month_raw"] = split["year"] * 100 + split["month_num"]
            y_cols = [TARGET_COL, "resale_price", "year_month_raw"]
        y = split[y_cols]

        X.to_parquet(os.path.join(run_dir, f"X_{name}.parquet"), index=False)
        y.to_parquet(os.path.join(run_dir, f"y_{name}.parquet"), index=False)

        print(f"  X_{name}.parquet {X.shape}   y_{name}.parquet {y.shape}")

    with open(os.path.join(run_dir, "target_encoders.pkl"), "wb") as f:
        pickle.dump(target_encoders, f)
    print("  target_encoders.pkl")

    # Rolling stats snapshot — keyed by (town, flat_type) for serving-time lookup.
    # Use val+test splits (most recent data) so the snapshot reflects current market.
    rolling_cols = [
        "town_flattype_median_1m", "town_flattype_median_3m", "town_flattype_median_6m",
        "town_flattype_psf_1m", "town_flattype_psf_3m", "town_median_3m",
        "town_txn_volume_3m", "price_momentum_3m", "psf_change_1m_vs_3m",
        "national_median_psf_1m", "national_median_psf_3m",
    ]
    recent_parts = [splits[k] for k in ("val", "test", "future_holdout") if k in splits]
    if recent_parts:
        recent_df = pd.concat(recent_parts, ignore_index=True)
        snap_df = (
            recent_df.groupby(["town", "flat_type"])[rolling_cols].median()
        )
        rolling_snapshot = {}
        for (town, flat_type), row in snap_df.iterrows():
            rolling_snapshot[(town, flat_type)] = {col: float(row[col]) for col in rolling_cols}
        global_defaults = recent_df[rolling_cols].median().to_dict()
        rolling_snapshot["_global_defaults"] = {col: float(v) for col, v in global_defaults.items()}
        with open(os.path.join(run_dir, "rolling_stats_snapshot.pkl"), "wb") as f:
            pickle.dump(rolling_snapshot, f)
        print("  rolling_stats_snapshot.pkl")

    # Save a no-op placeholder for backward compatibility; tree models don't need scaling.
    with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(None, f)
    print("  scaler.pkl (no-op, scaling disabled)")

    with open(os.path.join(run_dir, "feature_cols.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))
    print("  feature_cols.txt")

    manifest = {
        "run_at": datetime.now(UTC).isoformat(),
        "train_rows": len(splits["train"]),
        "val_rows": len(splits["val"]),
        "test_rows": len(splits["test"]),
        "feature_cols": FEATURE_COLS,
        "scale_cols": SCALE_COLS,
        "quantiles": QUANTILES,
        "split_metadata": split_metadata,
        "data_source": data_source,
        "target_transform": "log1p_resale_price",
        "outlier_strategy": "diagnostic_iqr_bounds_keep_nonzero_prices",
        "scaling_enabled": False,
    }

    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  run_manifest.json")

    metrics_stub = {
        "note": "Populate with actual model metrics after training.",
    }

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_stub, f, indent=2)
    print("  metrics.json (stub)")

    latest_path = os.path.join(OUTPUT_DIR, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(os.path.relpath(run_dir, BASE_DIR))
    print(f"  latest.txt → {os.path.relpath(run_dir, BASE_DIR)}")

    return run_dir


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(train: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY (train set)")
    print("=" * 60)

    stats = train[FEATURE_COLS + [TARGET_COL]].describe().T
    print(stats[["mean", "std", "min", "max"]].to_string())

    null_count = train[FEATURE_COLS].isnull().sum().sum()
    print(f"\n  Null values: {null_count}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("HDB Resale — Feature Engineering")
    print("=" * 60)

    df, data_source = load_data()
    df = engineer_features(df)
    df = drop_missing_model_rows(df)

    splits = stratified_split(df, test_months=3)
    split_metadata = build_split_metadata(splits)
    splits["train"] = remove_outliers(splits["train"])
    splits, target_encoders = target_encode(splits)

    print_summary(splits["train"])
    run_dir = save_artefacts(
        splits,
        target_encoders,
        split_metadata,
        data_source,
    )

    print(f"\nDone. ML-ready data saved to: {run_dir}/")


if __name__ == "__main__":
    main()
