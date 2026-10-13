import duckdb

con = duckdb.connect()

file = "data/training/training_2025-01-01_2025-02-01.parquet"

# Inspect column names and data types
print(con.execute(f"""
    DESCRIBE SELECT * FROM read_parquet('{file}')
""").df())

# Preview 5 rows
print(con.execute(f"""
    SELECT * FROM read_parquet('{file}') LIMIT 5
""").df().to_string())