from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]


train_dir = ROOT / "data" / "training"

files = sorted(train_dir.glob("training_*.parquet"))

if not files:
    raise FileNotFoundError(
        f"No monthly training files found in {train_dir}"
    )

print(f"Found {len(files)} monthly files:")
for file in files:
    print(f"  {file.name}: {file.stat().st_size / 1e6:.1f} MB")


paths_sql = ", ".join(
    "'" + str(p.resolve()).replace("\\", "/").replace("'", "''") + "'"
    for p in files
)

con = duckdb.connect()
con.execute(f"""
    CREATE VIEW movements AS
    SELECT *
    FROM read_parquet([{paths_sql}], union_by_name=true)
""")

print("\n--- SCHEMA ---")
print(con.execute("DESCRIBE movements").fetchdf().to_string(index=False))

print("\n--- SAMPLE ROWS ---")
print(con.execute("SELECT * FROM movements LIMIT 5").fetchdf().to_string(index=False))

print("\n--- ROW COUNTS BY PHASE ---")
print(con.execute("""
    SELECT PHASE_mvt, COUNT(*) AS rows
    FROM movements
    GROUP BY PHASE_mvt
    ORDER BY PHASE_mvt
""").fetchdf().to_string(index=False))

print("\n--- DEPARTURE TAXI-TIME QUALITY ---")
print(con.execute("""
    SELECT
        COUNT(*) AS departures,
        COUNT(*) FILTER (
            WHERE TAXITIME_SEC_mvt IS NULL
        ) AS missing_target,
        COUNT(*) FILTER (
            WHERE TAXITIME_SEC_mvt < 0
        ) AS negative_target,
        ROUND(AVG(TAXITIME_SEC_mvt), 1) AS mean_seconds,
        MEDIAN(TAXITIME_SEC_mvt) AS median_seconds,
        ROUND(QUANTILE_CONT(TAXITIME_SEC_mvt, 0.99), 1)
            AS p99_seconds,
        MAX(TAXITIME_SEC_mvt) AS max_seconds
    FROM movements
    WHERE PHASE_mvt = 'DEP'
""").fetchdf().to_string(index=False))

print("\n--- DEPARTURES BY AIRPORT ---")
print(con.execute("""
    SELECT ADEP_mvt, COUNT(*) AS departures
    FROM movements
    WHERE PHASE_mvt = 'DEP'
    GROUP BY ADEP_mvt
    ORDER BY departures DESC
""").fetchdf().to_string(index=False))

con.close()