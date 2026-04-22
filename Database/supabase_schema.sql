-- ============================================================
-- Supabase Normalized Schema — HDB Resale Analytics Platform
-- Run this in Supabase SQL Editor (Settings > SQL Editor)
-- ============================================================

-- ── 1. Dimension Tables ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS towns (
    id               SERIAL PRIMARY KEY,
    name             TEXT    UNIQUE NOT NULL,
    is_mature_estate BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS flat_types (
    id      SERIAL PRIMARY KEY,
    name    TEXT    UNIQUE NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS flat_models (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- ── 2. Address / Location Table ───────────────────────────────

CREATE TABLE IF NOT EXISTS blocks (
    id                  SERIAL PRIMARY KEY,
    block               TEXT   NOT NULL,
    street_name         TEXT   NOT NULL,
    town_id             INTEGER REFERENCES towns(id),
    full_address        TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    dist_mrt            DOUBLE PRECISION,
    dist_cbd            DOUBLE PRECISION,
    dist_primary_school DOUBLE PRECISION,
    dist_major_mall     DOUBLE PRECISION,
    dist_hawker_centre  DOUBLE PRECISION,
    hawker_count_1km    INTEGER,
    dist_high_demand_primary_school DOUBLE PRECISION,
    high_demand_primary_count_1km INTEGER,
    UNIQUE (block, street_name)
);

ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS dist_hawker_centre DOUBLE PRECISION;

ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS hawker_count_1km INTEGER;

ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS dist_high_demand_primary_school DOUBLE PRECISION;

ALTER TABLE blocks
    ADD COLUMN IF NOT EXISTS high_demand_primary_count_1km INTEGER;

CREATE INDEX IF NOT EXISTS idx_blocks_town     ON blocks(town_id);
CREATE INDEX IF NOT EXISTS idx_blocks_location ON blocks(latitude, longitude);

-- ── 3. Fact Table (Transactions) ─────────────────────────────

CREATE TABLE IF NOT EXISTS transactions (
    id                     SERIAL PRIMARY KEY,
    block_id               INTEGER REFERENCES blocks(id),
    flat_type_id           INTEGER REFERENCES flat_types(id),
    flat_model_id          INTEGER REFERENCES flat_models(id),
    storey_range           TEXT             NOT NULL,
    storey_midpoint        DOUBLE PRECISION,
    floor_area_sqm         DOUBLE PRECISION NOT NULL,
    lease_commence_date    INTEGER          NOT NULL,
    remaining_lease        DOUBLE PRECISION,
    remaining_lease_months DOUBLE PRECISION,
    resale_price           DOUBLE PRECISION NOT NULL,
    month                  TEXT             NOT NULL,
    month_num              INTEGER,
    year                   INTEGER          NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_txn_year      ON transactions(year);
CREATE INDEX IF NOT EXISTS idx_txn_block     ON transactions(block_id);
CREATE INDEX IF NOT EXISTS idx_txn_flat_type ON transactions(flat_type_id);
CREATE INDEX IF NOT EXISTS idx_txn_year_block ON transactions(year, block_id);

-- ── 4. User Tables ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id                SERIAL PRIMARY KEY,
    username          TEXT UNIQUE NOT NULL,
    email             TEXT UNIQUE NOT NULL,
    password_hash     TEXT NOT NULL,
    subscription_tier TEXT NOT NULL DEFAULT 'general' CHECK (subscription_tier IN ('general', 'premium')),
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feature_view_log (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    feature    TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE SEQUENCE IF NOT EXISTS feature_view_log_id_seq;

ALTER TABLE feature_view_log
    ALTER COLUMN id SET DEFAULT nextval('feature_view_log_id_seq');

ALTER SEQUENCE feature_view_log_id_seq OWNED BY feature_view_log.id;

SELECT setval(
    'feature_view_log_id_seq',
    COALESCE((SELECT MAX(id) FROM feature_view_log), 0) + 1,
    false
);

CREATE TABLE IF NOT EXISTS saved_predictions (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    town            TEXT             NOT NULL,
    flat_type       TEXT             NOT NULL,
    flat_model      TEXT             NOT NULL,
    floor_area      DOUBLE PRECISION NOT NULL,
    storey_range    TEXT             NOT NULL,
    lease_commence  INTEGER          NOT NULL,
    predicted_price DOUBLE PRECISION NOT NULL,
    price_low       DOUBLE PRECISION NOT NULL,
    price_high      DOUBLE PRECISION NOT NULL,
    street_name     TEXT             NOT NULL DEFAULT '',
    block           TEXT             NOT NULL DEFAULT '',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

ALTER TABLE IF EXISTS saved_predictions
    ADD COLUMN IF NOT EXISTS street_name TEXT NOT NULL DEFAULT '';

ALTER TABLE IF EXISTS saved_predictions
    ADD COLUMN IF NOT EXISTS block TEXT NOT NULL DEFAULT '';

-- Public reviews submitted from landing/review page (one row per logged-in user)
CREATE TABLE IF NOT EXISTS reviews (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL CHECK (char_length(name) <= 80),
    role        TEXT NOT NULL CHECK (char_length(role) <= 80),
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    content     TEXT NOT NULL CHECK (char_length(content) BETWEEN 20 AND 1200),
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_approved BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_reviews_public_quality
    ON reviews (is_approved, rating, created_at DESC);

-- Upgrade: reviews table created before user_id (drops anonymous rows without a user)
ALTER TABLE reviews ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;

DELETE FROM reviews WHERE user_id IS NULL;

ALTER TABLE reviews ALTER COLUMN user_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_one_per_user ON reviews (user_id);

-- ── 5. Model Versions Table ───────────────────────────────────

CREATE TABLE IF NOT EXISTS model_versions (
    id         SERIAL PRIMARY KEY,
    version    TEXT    NOT NULL,
    run_dir    TEXT,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    test_mape  DOUBLE PRECISION,
    test_rmse  DOUBLE PRECISION,
    test_r2    DOUBLE PRECISION,
    notes      TEXT,
    is_active  BOOLEAN DEFAULT FALSE
);

-- ── 6. RPC Functions (called by Flask app) ────────────────────

-- Towns list
CREATE OR REPLACE FUNCTION rpc_get_towns()
RETURNS TABLE(town TEXT) LANGUAGE SQL STABLE AS $$
    SELECT name FROM towns ORDER BY name;
$$;

-- Flat models list
CREATE OR REPLACE FUNCTION rpc_get_flat_models()
RETURNS TABLE(flat_model TEXT) LANGUAGE SQL STABLE AS $$
    SELECT name FROM flat_models ORDER BY name;
$$;

-- Town average distances (used by prediction engine)
CREATE OR REPLACE FUNCTION rpc_get_town_avg_distances()
RETURNS TABLE(
    town TEXT, avg_dist_mrt DOUBLE PRECISION, avg_dist_cbd DOUBLE PRECISION,
    avg_dist_school DOUBLE PRECISION, avg_dist_mall DOUBLE PRECISION,
    avg_dist_hawker DOUBLE PRECISION, avg_hawker_count_1km DOUBLE PRECISION,
    avg_dist_high_demand_school DOUBLE PRECISION,
    avg_high_demand_primary_count_1km DOUBLE PRECISION,
    avg_lat DOUBLE PRECISION, avg_lng DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        AVG(b.dist_mrt),
        AVG(b.dist_cbd),
        AVG(b.dist_primary_school),
        AVG(b.dist_major_mall),
        AVG(b.dist_hawker_centre),
        AVG(b.hawker_count_1km),
        AVG(b.dist_high_demand_primary_school),
        AVG(b.high_demand_primary_count_1km),
        AVG(b.latitude),
        AVG(b.longitude)
    FROM blocks b
    JOIN towns t ON b.town_id = t.id
    WHERE b.dist_mrt IS NOT NULL
    GROUP BY t.name;
$$;

-- Recent transactions for map
CREATE OR REPLACE FUNCTION rpc_api_transactions(
    p_town TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 500,
    p_min_year INTEGER DEFAULT NULL
)
RETURNS TABLE(
    town TEXT, flat_type TEXT, block TEXT, street_name TEXT,
    storey_range TEXT, floor_area_sqm DOUBLE PRECISION,
    resale_price DOUBLE PRECISION, month TEXT, year INTEGER,
    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name, ft.name, b.block, b.street_name,
        tx.storey_range, tx.floor_area_sqm,
        tx.resale_price, tx.month, tx.year,
        b.latitude, b.longitude
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE b.latitude IS NOT NULL AND b.longitude IS NOT NULL
      AND (p_town IS NULL OR t.name = p_town)
      AND (p_min_year IS NULL OR tx.year >= p_min_year)
    ORDER BY tx.year DESC, tx.month_num DESC
    LIMIT p_limit;
$$;

-- District summary for heatmap
CREATE OR REPLACE FUNCTION rpc_api_district_summary()
RETURNS TABLE(
    town TEXT, avg_price DOUBLE PRECISION, recent_avg DOUBLE PRECISION,
    total_txns BIGINT, recent_txns BIGINT,
    lat DOUBLE PRECISION, lng DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(AVG(CASE WHEN tx.year >= 2023 THEN tx.resale_price END)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        SUM(CASE WHEN tx.year >= 2023 THEN 1 ELSE 0 END),
        AVG(b.latitude),
        AVG(b.longitude)
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns  t ON b.town_id   = t.id
    WHERE b.latitude IS NOT NULL
    GROUP BY t.name
    ORDER BY t.name;
$$;

-- Yearly price trend (simple, no percentile)
CREATE OR REPLACE FUNCTION rpc_api_price_trend_simple(
    p_town TEXT DEFAULT NULL,
    p_flat_type TEXT DEFAULT NULL,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL
)
RETURNS TABLE(
    year INTEGER, avg_price DOUBLE PRECISION,
    min_price DOUBLE PRECISION, max_price DOUBLE PRECISION, txn_count BIGINT
) LANGUAGE SQL STABLE AS $$
    SELECT
        tx.year,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(MIN(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(MAX(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*)
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE (p_town IS NULL OR t.name = p_town)
      AND (p_flat_type IS NULL OR ft.name = p_flat_type)
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block)
    GROUP BY tx.year
    ORDER BY tx.year;
$$;

-- Yearly street-level price trend within a town
CREATE OR REPLACE FUNCTION rpc_api_street_price_trend(
    p_town TEXT,
    p_flat_type TEXT DEFAULT NULL,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL
)
RETURNS TABLE(
    street_name TEXT,
    year INTEGER,
    avg_price DOUBLE PRECISION,
    min_price DOUBLE PRECISION,
    max_price DOUBLE PRECISION,
    txn_count BIGINT,
    avg_area DOUBLE PRECISION,
    psf DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        b.street_name,
        tx.year,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(MIN(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        ROUND(MAX(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION,
        ROUND(AVG(tx.resale_price / NULLIF(tx.floor_area_sqm, 0))::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town
      AND (p_flat_type IS NULL OR ft.name = p_flat_type)
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block)
    GROUP BY b.street_name, tx.year
    ORDER BY b.street_name, tx.year;
$$;

-- District comparison (most recent year)
CREATE OR REPLACE FUNCTION rpc_api_district_comparison()
RETURNS TABLE(
    town TEXT, avg_price DOUBLE PRECISION, txn_count BIGINT,
    avg_area DOUBLE PRECISION, psf DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION,
        ROUND(AVG(tx.resale_price / NULLIF(tx.floor_area_sqm, 0))::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns  t ON b.town_id   = t.id
    WHERE tx.year = (SELECT MAX(year) FROM transactions)
    GROUP BY t.name
    ORDER BY AVG(tx.resale_price) DESC;
$$;

-- Flat type breakdown for a town or specific address scope.
-- Town-only requests stay recent-focused, while street/block requests use
-- full available history so address-specific option lists stay complete.
CREATE OR REPLACE FUNCTION rpc_api_flat_type_breakdown(
    p_town TEXT DEFAULT NULL,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL
)
RETURNS TABLE(
    flat_type TEXT, avg_price DOUBLE PRECISION,
    txn_count BIGINT, avg_area DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        ft.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE (p_town IS NULL OR t.name = p_town)
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block)
      AND (
          p_street_name IS NOT NULL
          OR p_block IS NOT NULL
          OR tx.year >= 2023
      )
    GROUP BY ft.name
    ORDER BY ft.name;
$$;

-- Monthly transaction volume
CREATE OR REPLACE FUNCTION rpc_api_monthly_volume(p_town TEXT DEFAULT NULL)
RETURNS TABLE(
    month INTEGER, txn_count BIGINT, avg_price DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        tx.month_num,
        COUNT(*),
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns  t ON b.town_id   = t.id
    WHERE (p_town IS NULL OR t.name = p_town)
    GROUP BY tx.month_num
    ORDER BY tx.month_num;
$$;

-- Price trend for predict page
CREATE OR REPLACE FUNCTION rpc_predict_trend(p_town TEXT, p_flat_type TEXT)
RETURNS TABLE(year INTEGER, avg_price DOUBLE PRECISION, txn_count BIGINT)
LANGUAGE SQL STABLE AS $$
    SELECT
        tx.year,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*)
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND ft.name = p_flat_type
    GROUP BY tx.year
    ORDER BY tx.year;
$$;

-- Benchmarks for predict page
CREATE OR REPLACE FUNCTION rpc_predict_benchmarks(p_town TEXT)
RETURNS TABLE(
    flat_type TEXT, avg_price DOUBLE PRECISION,
    txn_count BIGINT, avg_area DOUBLE PRECISION
) LANGUAGE SQL STABLE AS $$
    SELECT
        ft.name,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*),
        ROUND(AVG(tx.floor_area_sqm)::NUMERIC)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town AND tx.year >= 2023
    GROUP BY ft.name
    ORDER BY ft.name;
$$;

-- Resolve floor area for prediction (town + flat_type average)
CREATE OR REPLACE FUNCTION rpc_resolve_floor_area(
    p_town TEXT,
    p_flat_type TEXT,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL
)
RETURNS DOUBLE PRECISION LANGUAGE SQL STABLE AS $$
    SELECT ROUND(AVG(tx.floor_area_sqm)::NUMERIC, 1)::DOUBLE PRECISION
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town
      AND ft.name = p_flat_type
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block);
$$;

-- Resolve lease commence for prediction (town + flat_type average)
CREATE OR REPLACE FUNCTION rpc_resolve_lease_commence(
    p_town TEXT,
    p_flat_type TEXT,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL
)
RETURNS INTEGER LANGUAGE SQL STABLE AS $$
    SELECT ROUND(AVG(tx.lease_commence_date))::INTEGER
    FROM transactions tx
    JOIN blocks     b  ON tx.block_id     = b.id
    JOIN towns      t  ON b.town_id       = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town
      AND ft.name = p_flat_type
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block);
$$;

-- ── 7. Block-Level & Analytics RPC Functions ────────────────────

-- Streets for a town (block-level comparison)
CREATE OR REPLACE FUNCTION rpc_available_streets(p_town TEXT)
RETURNS TABLE(street_name TEXT) LANGUAGE SQL STABLE AS $$
    SELECT DISTINCT b.street_name
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town
    ORDER BY b.street_name;
$$;

-- Blocks for a town + street
CREATE OR REPLACE FUNCTION rpc_available_blocks(p_town TEXT, p_street TEXT)
RETURNS TABLE(block TEXT) LANGUAGE SQL STABLE AS $$
    SELECT DISTINCT b.block
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town AND b.street_name = p_street
    ORDER BY b.block;
$$;

-- Block-level distances for prediction
CREATE OR REPLACE FUNCTION rpc_block_distances(p_town TEXT, p_street TEXT, p_block TEXT)
RETURNS TABLE(dist_mrt DOUBLE PRECISION, dist_cbd DOUBLE PRECISION,
              dist_school DOUBLE PRECISION, dist_mall DOUBLE PRECISION,
              dist_hawker DOUBLE PRECISION, hawker_count_1km INTEGER,
              dist_high_demand_school DOUBLE PRECISION,
              high_demand_primary_count_1km INTEGER)
LANGUAGE SQL STABLE AS $$
    SELECT
        b.dist_mrt,
        b.dist_cbd,
        b.dist_primary_school,
        b.dist_major_mall,
        b.dist_hawker_centre,
        b.hawker_count_1km,
        b.dist_high_demand_primary_school,
        b.high_demand_primary_count_1km
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town
      AND upper(trim(b.street_name)) = upper(trim(p_street))
      AND trim(b.block) = trim(p_block)
    LIMIT 1;
$$;

-- When an exact block row is missing, use averages for that street (still far more precise than town averages).
CREATE OR REPLACE FUNCTION rpc_street_avg_distances(p_town TEXT, p_street TEXT)
RETURNS TABLE(dist_mrt DOUBLE PRECISION, dist_cbd DOUBLE PRECISION,
              dist_school DOUBLE PRECISION, dist_mall DOUBLE PRECISION,
              dist_hawker DOUBLE PRECISION, hawker_count_1km DOUBLE PRECISION,
              dist_high_demand_school DOUBLE PRECISION,
              high_demand_primary_count_1km DOUBLE PRECISION)
LANGUAGE SQL STABLE AS $$
    SELECT
        AVG(b.dist_mrt),
        AVG(b.dist_cbd),
        AVG(b.dist_primary_school),
        AVG(b.dist_major_mall),
        AVG(b.dist_hawker_centre),
        AVG(b.hawker_count_1km)::DOUBLE PRECISION,
        AVG(b.dist_high_demand_primary_school),
        AVG(b.high_demand_primary_count_1km)::DOUBLE PRECISION
    FROM blocks b JOIN towns t ON b.town_id = t.id
    WHERE t.name = p_town
      AND upper(trim(b.street_name)) = upper(trim(p_street));
$$;

-- Lease decay data for analytics
CREATE OR REPLACE FUNCTION rpc_lease_decay(
    p_town TEXT,
    p_flat_type TEXT DEFAULT NULL,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL
)
RETURNS TABLE(lease_bucket INTEGER, avg_price DOUBLE PRECISION, txn_count BIGINT)
LANGUAGE SQL STABLE AS $$
    SELECT
        (CAST(tx.remaining_lease AS INT) / 10) * 10,
        ROUND(AVG(tx.resale_price)::NUMERIC)::DOUBLE PRECISION,
        COUNT(*)
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town
      AND (p_flat_type IS NULL OR ft.name = p_flat_type)
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block)
    GROUP BY (CAST(tx.remaining_lease AS INT) / 10) * 10
    ORDER BY 1 DESC;
$$;

-- Recent similar transactions for prediction context
DROP FUNCTION IF EXISTS rpc_recent_similar_transactions(TEXT, TEXT, INTEGER, TEXT, TEXT, TEXT);
CREATE OR REPLACE FUNCTION rpc_recent_similar_transactions(
    p_town TEXT,
    p_flat_type TEXT,
    p_limit INTEGER DEFAULT 20,
    p_street_name TEXT DEFAULT NULL,
    p_block TEXT DEFAULT NULL,
    p_storey_range TEXT DEFAULT NULL,
    p_min_year INTEGER DEFAULT NULL
)
RETURNS TABLE(
    block TEXT, street_name TEXT, storey_range TEXT,
    floor_area_sqm DOUBLE PRECISION, resale_price DOUBLE PRECISION, month TEXT, year INTEGER
) LANGUAGE SQL STABLE AS $$
    SELECT b.block, b.street_name, tx.storey_range,
           tx.floor_area_sqm, tx.resale_price, tx.month, tx.year
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE t.name = p_town
      AND ft.name = p_flat_type
      AND (p_street_name IS NULL OR b.street_name = p_street_name)
      AND (p_block IS NULL OR b.block = p_block)
      AND (p_storey_range IS NULL OR tx.storey_range = p_storey_range)
      AND (p_min_year IS NULL OR tx.year >= p_min_year)
    ORDER BY tx.year DESC, tx.month_num DESC
    LIMIT p_limit;
$$;

-- Total transaction count for homepage stats
CREATE OR REPLACE FUNCTION rpc_count_transactions()
RETURNS BIGINT
LANGUAGE SQL STABLE AS $$
    SELECT COUNT(*) FROM transactions;
$$;

-- Available flat models for a town + flat type
CREATE OR REPLACE FUNCTION rpc_api_available_models(
    p_town TEXT,
    p_flat_type TEXT
)
RETURNS TABLE(flat_model TEXT)
LANGUAGE SQL STABLE AS $$
    SELECT DISTINCT fm.name
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    JOIN flat_models fm ON tx.flat_model_id = fm.id
    WHERE t.name = p_town
      AND ft.name = p_flat_type
    ORDER BY fm.name;
$$;

-- Available storey ranges for a town + flat type
CREATE OR REPLACE FUNCTION rpc_api_available_storey_ranges(
    p_town TEXT DEFAULT NULL,
    p_flat_type TEXT DEFAULT NULL
)
RETURNS TABLE(storey_range TEXT)
LANGUAGE SQL STABLE AS $$
    SELECT DISTINCT tx.storey_range
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE (p_town IS NULL OR t.name = p_town)
      AND (p_flat_type IS NULL OR ft.name = p_flat_type)
    ORDER BY tx.storey_range;
$$;

-- Floor area bounds for UI sliders
CREATE OR REPLACE FUNCTION rpc_api_floor_area_stats(
    p_town TEXT DEFAULT NULL,
    p_flat_type TEXT DEFAULT NULL
)
RETURNS TABLE(
    min_area INTEGER,
    max_area INTEGER,
    avg_area INTEGER
)
LANGUAGE SQL STABLE AS $$
    SELECT
        ROUND(MIN(tx.floor_area_sqm))::INTEGER,
        ROUND(MAX(tx.floor_area_sqm))::INTEGER,
        ROUND(AVG(tx.floor_area_sqm))::INTEGER
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    WHERE (p_town IS NULL OR t.name = p_town)
      AND (p_flat_type IS NULL OR ft.name = p_flat_type);
$$;

-- Lease commence year bounds for UI sliders
CREATE OR REPLACE FUNCTION rpc_api_lease_year_range(p_town TEXT DEFAULT NULL)
RETURNS TABLE(
    min_year INTEGER,
    max_year INTEGER,
    avg_year INTEGER
)
LANGUAGE SQL STABLE AS $$
    SELECT
        MIN(tx.lease_commence_date)::INTEGER,
        MAX(tx.lease_commence_date)::INTEGER,
        ROUND(AVG(tx.lease_commence_date))::INTEGER
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    WHERE (p_town IS NULL OR t.name = p_town);
$$;

-- Guest teaser map without exposing actual prices
CREATE OR REPLACE FUNCTION rpc_api_public_location_summary()
RETURNS TABLE(
    town TEXT,
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    total_txns BIGINT,
    price_bucket INTEGER
)
LANGUAGE SQL STABLE AS $$
    WITH town_stats AS (
        SELECT
            t.name AS town,
            AVG(b.latitude) AS lat,
            AVG(b.longitude) AS lng,
            COUNT(*) AS total_txns,
            AVG(tx.resale_price) AS avg_price
        FROM transactions tx
        JOIN blocks b ON tx.block_id = b.id
        JOIN towns t ON b.town_id = t.id
        WHERE b.latitude IS NOT NULL
          AND b.longitude IS NOT NULL
        GROUP BY t.name
    )
    SELECT
        town,
        lat,
        lng,
        total_txns,
        NTILE(5) OVER (ORDER BY avg_price)::INTEGER AS price_bucket
    FROM town_stats
    ORDER BY town;
$$;

-- Public homepage ticker
CREATE OR REPLACE FUNCTION rpc_api_public_recent_ticker()
RETURNS TABLE(
    town TEXT,
    flat_type TEXT,
    resale_price DOUBLE PRECISION,
    year INTEGER
)
LANGUAGE SQL STABLE AS $$
    SELECT
        t.name,
        ft.name,
        tx.resale_price,
        tx.year
    FROM transactions tx
    JOIN blocks b ON tx.block_id = b.id
    JOIN towns t ON b.town_id = t.id
    JOIN flat_types ft ON tx.flat_type_id = ft.id
    ORDER BY tx.year DESC, tx.month_num DESC
    LIMIT 20;
$$;

-- Public random high-quality reviews for landing page
-- DROP required when OUT/RETURNS shape changes (CREATE OR REPLACE is not enough).
DROP FUNCTION IF EXISTS rpc_public_reviews(integer, integer);

CREATE OR REPLACE FUNCTION rpc_public_reviews(
    p_limit INTEGER DEFAULT 5,
    p_min_rating INTEGER DEFAULT 4
)
RETURNS TABLE(
    id BIGINT,
    user_id INTEGER,
    name TEXT,
    role TEXT,
    rating INTEGER,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    subscription_tier TEXT
)
LANGUAGE SQL STABLE AS $$
    SELECT
        r.id,
        r.user_id::INTEGER,
        r.name,
        r.role,
        r.rating,
        r.content,
        r.created_at,
        COALESCE(u.subscription_tier, 'general')::TEXT AS subscription_tier
    FROM reviews r
    LEFT JOIN users u ON u.id = r.user_id
    WHERE r.is_approved = TRUE
      AND r.rating >= GREATEST(1, LEAST(p_min_rating, 5))
    ORDER BY random()
    LIMIT GREATEST(1, LEAST(p_limit, 20));
$$;
