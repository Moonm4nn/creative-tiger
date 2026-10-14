from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
ranking = ROOT / "data" / "final_ranking.parquet"
submission = ROOT / "data" / "final_submitting.parquet"

for path in (ranking, submission):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

con = duckdb.connect()

for label, path in [
    ("RANKING", ranking),
    ("SUBMISSION TEMPLATE", submission),
]:
    p = str(path.resolve()).replace("\\", "/").replace("'", "''")

    print(f"\n--- {label}: {path.name} ---")

    print(con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{p}')"
    ).fetchdf().to_string(index=False))

    print(con.execute(
        f"SELECT COUNT(*) AS rows FROM read_parquet('{p}')"
    ).fetchdf().to_string(index=False))

    print(con.execute(
        f"SELECT * FROM read_parquet('{p}') LIMIT 5"
    ).fetchdf().to_string(index=False))

# Check the required output format.
p = str(submission.resolve()).replace("\\", "/").replace("'", "''")

print("\n--- SUBMISSION CHECK ---")
print(con.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT MVT_ID_mvt) AS unique_ids,
        COUNT(*) FILTER (
            WHERE TAXITIME_SEC_mvt IS NULL
        ) AS missing_predictions
    FROM read_parquet('{p}')
""").fetchdf().to_string(index=False))

con.close()