from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
train_dir = ROOT / "data" / "training"

if not train_dir.exists():
    train_dir = ROOT / "data"

files = sorted(train_dir.glob("training_*.parquet"))
if not files:
    raise FileNotFoundError(f"No training files in {train_dir}")

paths = ", ".join(
    "'" + str(p.resolve()).replace("\\", "/").replace("'", "''") + "'"
    for p in files
)

con = duckdb.connect()
con.execute(f"""
    CREATE VIEW movements AS
    SELECT *
    FROM read_parquet([{paths}], union_by_name=true)
""")

# 1. Examine the negative taxi-out times.
print("\n--- NEGATIVE DEPARTURE TARGETS ---")
print(con.execute("""
    SELECT
        ADEP_mvt,
        MVT_ID_mvt,
        MVT_TIME_UTC_mvt,
        BLOCK_TIME_UTC_mvt,
        TAXITIME_SEC_mvt
    FROM movements
    WHERE PHASE_mvt = 'DEP'
      AND TAXITIME_SEC_mvt < 0
    ORDER BY TAXITIME_SEC_mvt
    LIMIT 30
""").fetchdf().to_string(index=False))

# 2. Inspect the 25 largest departure targets.
print("\n--- LARGEST DEPARTURE TARGETS ---")
print(con.execute("""
    SELECT
        ADEP_mvt,
        MVT_ID_mvt,
        MVT_TIME_UTC_mvt,
        BLOCK_TIME_UTC_mvt,
        TAXITIME_SEC_mvt
    FROM movements
    WHERE PHASE_mvt = 'DEP'
      AND TAXITIME_SEC_mvt IS NOT NULL
    ORDER BY TAXITIME_SEC_mvt DESC
    LIMIT 25
""").fetchdf().to_string(index=False))

# 3. Quantify the target distribution by airport.
print("\n--- TARGET DISTRIBUTION BY AIRPORT ---")
print(con.execute("""
    SELECT
        ADEP_mvt,
        COUNT(*) AS n,
        ROUND(AVG(TAXITIME_SEC_mvt), 1) AS mean_sec,
        MEDIAN(TAXITIME_SEC_mvt) AS median_sec,
        ROUND(
            QUANTILE_CONT(TAXITIME_SEC_mvt, 0.99), 1
        ) AS p99_sec,
        MAX(TAXITIME_SEC_mvt) AS max_sec,
        COUNT(*) FILTER (
            WHERE TAXITIME_SEC_mvt < 0
        ) AS negative_count,
        COUNT(*) FILTER (
            WHERE TAXITIME_SEC_mvt > 7200
        ) AS over_2_hours
    FROM movements
    WHERE PHASE_mvt = 'DEP'
    GROUP BY ADEP_mvt
    ORDER BY ADEP_mvt
""").fetchdf().to_string(index=False))

con.close()