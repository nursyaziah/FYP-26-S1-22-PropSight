"""
fetch_sora.py — MAS SORA data loader
=====================================
Provides load_sora_monthly() which returns a DataFrame with columns:
  year_month  (int, YYYYMM)
  sora_3m     (float, monthly average of daily 3-month compounded SORA)

Primary source: MAS API Gateway (chunked, retried).
  - Fetches SORA from Jan 2015, matching the project training window.
  - Uses MAS_API_KEY / MAS_KEYID when available.

Fallback: ML/data/sora_monthly.csv (covers Jan 2015 – present).
  - First 6 rows are metadata; data starts at row 7.
  - Header row columns used: "SORA Publication Date", "Compound SORA - 3 month"
  - The API is known to be intermittently unavailable (scheduled maintenance).
  - If both sources fail, returns an empty DataFrame; feature engineering
    handles this with median fill.
"""

import json
import os
import time
from datetime import date
from urllib import parse, request as urllib_request

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SORA_CSV_PATH = os.path.join(BASE_DIR, "data", "sora_monthly.csv")

_MAS_SORA_VIEW_URL = (
    "https://eservices.mas.gov.sg/apimg-gw/server/"
    "monthly_statistical_bulletin_non610mssql/domestic_interest_rates_daily/"
    "views/domestic_interest_rates_daily"
)
_MAS_LEGACY_SORA_BASE = (
    "https://eservices.mas.gov.sg/api/action/datastore/search.json"
    "?resource_id=9a0bf149-308c-4bd2-832d-76c8e6cb47ed"
)
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_TIMEOUT = 30
_CHUNK_YEARS = 2
_MAX_RETRIES = 3
_RETRY_SLEEP = 5
_START_YEAR = 2015  # Project training data starts at Jan 2015


def _mas_keyid() -> str:
    """Return the MAS API Gateway KeyId from the environment, if configured."""
    return (
        os.environ.get("MAS_API_KEY")
        or os.environ.get("MAS_KEYID")
        or ""
    ).strip()


def _extract_records(data) -> list[dict]:
    """
    Extract rows from either the current MAS API Gateway response or the legacy
    datastore response. MAS has changed wrappers before, so keep this tolerant.
    """
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if not isinstance(data, dict):
        return []

    result = data.get("result")
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        return [row for row in result["records"] if isinstance(row, dict)]

    for key in ("records", "items", "data", "value", "elements", "rows"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    if {"end_of_day", "comp_sora_3m"}.issubset(data):
        return [data]

    return []


def _read_json_response(req: urllib_request.Request) -> dict | list:
    with urllib_request.urlopen(req, timeout=_TIMEOUT) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read()

    if "html" in content_type.lower() or "xml" in content_type.lower():
        raise ValueError(
            f"API returned {content_type or 'non-JSON'} instead of JSON"
        )

    return json.loads(raw.decode("utf-8-sig"))


def _load_sora_from_csv() -> pd.DataFrame:
    """
    Load daily SORA data from the local CSV file.

    The file has 6 metadata rows before the actual header (row index 6).
    Columns used:
      "SORA Publication Date"  — date string like "05 Jan 2015"
      "Compound SORA - 3 month" — daily rate as float
    """
    df = pd.read_csv(SORA_CSV_PATH, skiprows=6, header=0)
    df = df.rename(columns={
        "SORA Publication Date": "end_of_day",
        "Compound SORA - 3 month": "comp_sora_3m",
    })
    df["end_of_day"] = pd.to_datetime(df["end_of_day"], errors="coerce")
    df["comp_sora_3m"] = pd.to_numeric(df["comp_sora_3m"], errors="coerce")
    df = df.dropna(subset=["end_of_day", "comp_sora_3m"])
    return df[["end_of_day", "comp_sora_3m"]]


def _fetch_sora_from_gateway() -> pd.DataFrame:
    """
    Fetch daily SORA records from the current MAS API Gateway.

    MAS requires a KeyId header for this endpoint. The view API also asks for a
    filtering parameter, so each request is chunked by end_of_day.
    """
    keyid = _mas_keyid()
    if not keyid:
        raise ValueError("MAS_API_KEY / MAS_KEYID is not configured.")

    today = date.today()
    current_year = today.year
    all_records: list[dict] = []

    for chunk_start in range(_START_YEAR, current_year + 1, _CHUNK_YEARS):
        chunk_end = min(chunk_start + _CHUNK_YEARS - 1, current_year)
        start_date = f"{chunk_start}-01-01"
        end_date = f"{chunk_end}-12-31"

        query = parse.urlencode({
            "$select": "end_of_day,comp_sora_3m",
            "$filter": (
                f"end_of_day >= '{start_date}' "
                f"and end_of_day <= '{end_date}' "
                "and comp_sora_3m is not null"
            ),
            "$orderby": "end_of_day asc",
        })
        url = f"{_MAS_SORA_VIEW_URL}?{query}"

        chunk_records = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                req = urllib_request.Request(
                    url,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept": "application/json",
                        "KeyId": keyid,
                    },
                )
                data = _read_json_response(req)
                chunk_records = _extract_records(data)
                break
            except Exception as exc:
                print(
                    f"  SORA API Gateway: chunk {start_date}–{end_date} "
                    f"attempt {attempt}/{_MAX_RETRIES} failed: {exc}"
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP)

        if chunk_records is None:
            raise ValueError(
                f"MAS API Gateway failed for chunk {start_date}–{end_date} "
                f"after {_MAX_RETRIES} attempts."
            )

        all_records.extend(chunk_records)

    if not all_records:
        raise ValueError("MAS API Gateway returned no SORA records.")

    df = pd.DataFrame(all_records)[["end_of_day", "comp_sora_3m"]]
    df["end_of_day"] = pd.to_datetime(df["end_of_day"], errors="coerce")
    df["comp_sora_3m"] = pd.to_numeric(df["comp_sora_3m"], errors="coerce")
    df = df.dropna(subset=["end_of_day", "comp_sora_3m"])
    return df


def _fetch_sora_from_legacy_api() -> pd.DataFrame:
    """
    Fetch daily SORA records from the older MAS datastore API in 2-year chunks.

    This is kept as a fallback because it does not require a KeyId.
    """
    today = date.today()
    current_year = today.year
    all_records: list[dict] = []

    for chunk_start in range(_START_YEAR, current_year + 1, _CHUNK_YEARS):
        chunk_end = min(chunk_start + _CHUNK_YEARS - 1, current_year)
        start_date = f"{chunk_start}-01-01"
        end_date = f"{chunk_end}-12-31"

        # Literal brackets are valid in query strings (RFC 3986 gen-delims)
        url = (
            f"{_MAS_LEGACY_SORA_BASE}"
            f"&fields=end_of_day,comp_sora_3m"
            f"&between[end_of_day]={start_date},{end_date}"
            f"&limit=10000"
        )

        chunk_records = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                req = urllib_request.Request(
                    url, headers={"User-Agent": _USER_AGENT}
                )
                data = _read_json_response(req)
                chunk_records = _extract_records(data)
                break
            except Exception as exc:
                print(
                    f"  SORA legacy API: chunk {start_date}–{end_date} "
                    f"attempt {attempt}/{_MAX_RETRIES} failed: {exc}"
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP)

        if chunk_records is None:
            raise ValueError(
                f"MAS legacy API failed for chunk {start_date}–{end_date} "
                f"after {_MAX_RETRIES} attempts."
            )

        all_records.extend(chunk_records)

    if not all_records:
        raise ValueError("MAS legacy API returned no SORA records.")

    df = pd.DataFrame(all_records)[["end_of_day", "comp_sora_3m"]]
    df["end_of_day"] = pd.to_datetime(df["end_of_day"], errors="coerce")
    df["comp_sora_3m"] = pd.to_numeric(df["comp_sora_3m"], errors="coerce")
    df = df.dropna(subset=["end_of_day", "comp_sora_3m"])
    return df


def _fetch_sora_from_api() -> pd.DataFrame:
    """Fetch daily SORA rows from MAS, preferring the current API Gateway."""
    try:
        return _fetch_sora_from_gateway()
    except Exception as exc:
        print(f"  SORA: API Gateway load failed ({exc}), trying legacy API.")
        return _fetch_sora_from_legacy_api()


def load_sora_monthly() -> pd.DataFrame:
    """
    Return monthly average 3-month compounded SORA.

    Tries the current MAS API first so retraining can use the latest published
    SORA values. Falls back to the legacy MAS API, then the local CSV. Returns
    an empty DataFrame if all sources fail.

    Returns DataFrame with columns:
      year_month  int   YYYYMM (e.g. 202401)
      sora_3m     float monthly average of daily comp_sora_3m
    """
    daily: pd.DataFrame | None = None

    # --- Primary: MAS API ---
    try:
        daily = _fetch_sora_from_api()
        print(f"  SORA: loaded {len(daily):,} daily rows from MAS API.")
    except Exception as exc:
        print(f"  SORA: API load failed ({exc}), trying CSV fallback.")
        daily = None

    # --- Fallback: local CSV ---
    if daily is None and os.path.exists(SORA_CSV_PATH):
        try:
            daily = _load_sora_from_csv()
            if daily.empty:
                print("  SORA: CSV is empty.")
                daily = None
            else:
                print(
                    f"  SORA: loaded {len(daily):,} daily rows from CSV "
                    f"({daily['end_of_day'].min().date()} – "
                    f"{daily['end_of_day'].max().date()})."
                )
        except Exception as exc:
            print(f"  SORA: CSV load failed ({exc}).")
            daily = None

    if daily is None:
        print("  SORA: no API or CSV data available; median fill will be applied.")
        return pd.DataFrame(columns=["year_month", "sora_3m"])

    daily["year_month"] = (
        daily["end_of_day"].dt.year * 100 + daily["end_of_day"].dt.month
    ).astype(int)

    monthly = (
        daily.groupby("year_month")["comp_sora_3m"]
        .mean()
        .reset_index()
        .rename(columns={"comp_sora_3m": "sora_3m"})
    )
    return monthly
