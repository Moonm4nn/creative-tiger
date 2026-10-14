
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "training"

if not TRAIN_DIR.exists():
    TRAIN_DIR = ROOT / "data"

FILES = sorted(TRAIN_DIR.glob("training_*.parquet"))

if not FILES:
    raise FileNotFoundError(f"No training files found in {TRAIN_DIR}")

# Set to None later to use all training rows in each fold.
MAX_TRAIN_ROWS = 300_000

RANDOM_SEED = 42
ITERATIONS = 400
DEPTH = 8
LEARNING_RATE = 0.08

VALIDATION_MONTHS = ["2025-10", "2025-11", "2025-12"]

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading parquet files...")

df = pd.concat(
    [pd.read_parquet(path) for path in FILES],
    ignore_index=True,
)

df = df.loc[
    df["PHASE_mvt"].eq("DEP")
    & df["TAXITIME_SEC_mvt"].notna()
].copy()

df["target"] = df["TAXITIME_SEC_mvt"].astype(float)

# Actual event timestamp is used only for chronological splitting.
df["event_time"] = pd.to_datetime(
    df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce"
)

df = df.loc[df["event_time"].notna()].copy()

df["event_month"] = (
    df["event_time"]
    .dt.tz_convert("UTC")
    .dt.tz_localize(None)
    .dt.to_period("M")
)

# Candidate calendar features are derived from scheduled time,
# not the actual movement timestamp.
scheduled = pd.to_datetime(
    df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce"
)

df["scheduled_hour"] = scheduled.dt.hour
df["scheduled_dow"] = scheduled.dt.dayofweek
df["scheduled_month"] = scheduled.dt.month

# Airport and other categorical features.
df["airport"] = df["ADEP_mvt"].fillna("__MISSING__").astype(str)
df["aircraft_type"] = (
    df["AIRCRAFT_TYPE_mvt"].fillna("__MISSING__").astype(str)
)
df["operator"] = (
    df["AIRCRAFT_OPERATOR_flt"].fillna("__MISSING__").astype(str)
)
df["flight_rule"] = (
    df["FLIGHT_RULE_mvt"].fillna("__MISSING__").astype(str)
)

# Make calendar features categorical so CatBoost learns
# non-linear differences between hours, weekdays, and months.
for col in ["scheduled_hour", "scheduled_dow", "scheduled_month"]:
    df[col] = df[col].apply(
        lambda x: str(int(x)) if pd.notna(x) else "__MISSING__"
    )

# --------------------------------------------------
# Cumulative feature versions
# --------------------------------------------------

versions = {
    "V0_airport": [
        "airport",
    ],
    "V1_hour": [
        "airport",
        "scheduled_hour",
    ],
    "V2_weekday": [
        "airport",
        "scheduled_hour",
        "scheduled_dow",
    ],
    "V3_month": [
        "airport",
        "scheduled_hour",
        "scheduled_dow",
        "scheduled_month",
    ],
    "V4_aircraft": [
        "airport",
        "scheduled_hour",
        "scheduled_dow",
        "scheduled_month",
        "aircraft_type",
    ],
    "V5_operator": [
        "airport",
        "scheduled_hour",
        "scheduled_dow",
        "scheduled_month",
        "aircraft_type",
        "operator",
    ],
    "V6_flight_rule": [
        "airport",
        "scheduled_hour",
        "scheduled_dow",
        "scheduled_month",
        "aircraft_type",
        "operator",
        "flight_rule",
    ],
}

categorical_columns = [
    "airport",
    "scheduled_hour",
    "scheduled_dow",
    "scheduled_month",
    "aircraft_type",
    "operator",
    "flight_rule",
]

# --------------------------------------------------
# Train and evaluate each version
# --------------------------------------------------

results = []
all_predictions = []

for month in VALIDATION_MONTHS:
    validation_period = pd.Period(month, freq="M")

    train = df.loc[df["event_month"] < validation_period]
    valid = df.loc[df["event_month"] == validation_period]

    if train.empty or valid.empty:
        print(f"Skipping {month}: missing train or validation rows.")
        continue

    # Keep the same training sample for every version in this fold.
    if (
        MAX_TRAIN_ROWS is not None
        and len(train) > MAX_TRAIN_ROWS
    ):
        train = train.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_SEED,
        )

    print(
        f"\n{month}: training={len(train):,}, "
        f"validation={len(valid):,}"
    )

    y_train = train["target"]
    y_valid = valid["target"]

    for version, features in versions.items():
        model = CatBoostRegressor(
            iterations=ITERATIONS,
            depth=DEPTH,
            learning_rate=LEARNING_RATE,
            loss_function="RMSE",
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

        model.fit(
            train[features],
            y_train,
            cat_features=[
                col for col in features
                if col in categorical_columns
            ],
        )

        predictions = model.predict(valid[features])

        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
        mae = mean_absolute_error(y_valid, predictions)

        results.append({
            "month": month,
            "version": version,
            "n_train": len(train),
            "n_valid": len(valid),
            "rmse": rmse,
            "mae": mae,
        })

        all_predictions.append(pd.DataFrame({
            "month": month,
            "version": version,
            "actual": y_valid.to_numpy(),
            "prediction": predictions,
        }))

        print(
            f"{version:<18} RMSE={rmse:8.2f} sec "
            f"MAE={mae:8.2f} sec"
        )

# --------------------------------------------------
# Save results and calculate pooled RMSE
# --------------------------------------------------

if not results:
    raise RuntimeError("No validation results were produced.")

results_df = pd.DataFrame(results)

predictions_df = pd.concat(all_predictions, ignore_index=True)

pooled = (
    predictions_df.groupby("version")
    .apply(
        lambda group: pd.Series({
            "pooled_rmse": np.sqrt(
                mean_squared_error(
                    group["actual"], group["prediction"]
                )
            ),
            "pooled_mae": mean_absolute_error(
                group["actual"], group["prediction"]
            ),
        }),
        include_groups=False,
    )
    .reset_index()
)

summary = results_df.pivot(
    index="version",
    columns="month",
    values="rmse",
).reset_index()

summary = summary.merge(pooled, on="version")

output_dir = ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "feature_experiment_monthly.csv",
    index=False,
)

summary.to_csv(
    output_dir / "feature_experiment_summary.csv",
    index=False,
)

print("\n--- SUMMARY: RMSE BY MONTH AND POOLED METRIC ---")
print(summary.round(2).to_string(index=False))

print(f"\nSaved results to {output_dir}")
