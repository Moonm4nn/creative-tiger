
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
# MAX_TRAIN_ROWS = 300_000
MAX_TRAIN_ROWS = None

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
    "A4_operator": [
        "airport",
        "scheduled_hour",
        "scheduled_dow",
        "aircraft_type",
        "operator",
    ]
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

        # Keep original fields that may help explain extreme targets.
        diagnostic_cols = [
            "MVT_ID_mvt",
            "SCHED_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "MVT_TIME_UTC_mvt",
            "AIRCRAFT_TYPE_mvt",
            "AIRCRAFT_OPERATOR_flt",
            "RUNWAY_mvt",
            "STAND_mvt",
        ]

        available_cols = [
            c for c in diagnostic_cols if c in valid.columns
        ]

        error_df = valid[available_cols].reset_index(drop=True).copy()
        error_df["airport"] = valid["airport"].to_numpy()
        error_df["actual"] = np.asarray(y_valid)
        error_df["predicted"] = np.asarray(predictions)
        error_df["error"] = error_df["predicted"] - error_df["actual"]
        error_df["abs_error"] = error_df["error"].abs()

        # Review the most extreme LIRF targets, not only the largest model errors.
        lirf_extreme = (
            error_df[
                (error_df["airport"] == "LIRF")
                & (error_df["actual"] > 3 * 60 * 60)
            ]
            .sort_values("actual", ascending=False)
        )

        print("\n--- LIRF TARGETS LONGER THAN 3 HOURS ---")
        print(f"Number of records: {len(lirf_extreme)}")
        print(lirf_extreme.to_string(index=False))

        error_df = pd.DataFrame({
            "airport": valid["airport"].to_numpy(),
            "actual": np.asarray(y_valid),
            "predicted": np.asarray(predictions),
        })

        
        error_df["error"] = error_df["predicted"] - error_df["actual"]
        error_df["abs_error"] = error_df["error"].abs()
        error_df["squared_error"] = error_df["error"] ** 2

        airport_errors = (
            error_df.groupby("airport")
            .agg(
                n=("actual", "size"),
                actual_mean=("actual", "mean"),
                predicted_mean=("predicted", "mean"),
                mae=("abs_error", "mean"),
                mean_error=("error", "mean"),
                mse=("squared_error", "mean"),
                p95_abs_error=(
                    "abs_error",
                    lambda x: x.quantile(0.95),
                ),
            )
            .reset_index()
        )

        airport_errors["rmse"] = np.sqrt(airport_errors["mse"])
        airport_errors = airport_errors.drop(columns="mse")
        airport_errors = airport_errors.sort_values("rmse", ascending=False)

        print(airport_errors.to_string(index=False))

        # Inspect the largest errors for Rome in November
        if str(month) == "2025-11":
            worst_lirf = (
                error_df[error_df["airport"] == "LIRF"]
                .assign(abs_error=lambda x: x["error"].abs())
                .nlargest(30, "abs_error")
            )

            print("\n--- WORST LIRF PREDICTIONS: NOVEMBER 2025 ---")
            print(
                worst_lirf[
                    ["airport", "actual", "predicted", "error", "abs_error"]
                ].to_string(index=False)
            )
            
            if {
                "BLOCK_TIME_UTC_mvt",
                "MVT_TIME_UTC_mvt",
            }.issubset(error_df.columns):

                block_time = pd.to_datetime(
                    error_df["BLOCK_TIME_UTC_mvt"], utc=True, errors="coerce"
                )
                movement_time = pd.to_datetime(
                    error_df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce"
                )

                error_df["timestamp_diff_sec"] = (
                    movement_time - block_time
                ).dt.total_seconds()

                print("\n--- TARGET VS TIMESTAMP DIFFERENCE ---")
                print(
                    error_df.loc[
                        (error_df["airport"] == "LIRF")
                        & (error_df["actual"] > 3 * 60 * 60),
                        [
                            "MVT_ID_mvt",
                            "BLOCK_TIME_UTC_mvt",
                            "MVT_TIME_UTC_mvt",
                            "actual",
                            "timestamp_diff_sec",
                        ],
                    ].to_string(index=False)
                )

        audit = df.copy()

        # Use the existing target definition.
        audit["target_sec"] = pd.to_numeric(
            audit["TAXITIME_SEC_mvt"], errors="coerce"
        )

        audit["scheduled_time"] = pd.to_datetime(
            audit["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce"
        )
        audit["block_time"] = pd.to_datetime(
            audit["BLOCK_TIME_UTC_mvt"], utc=True, errors="coerce"
        )
        audit["movement_time"] = pd.to_datetime(
            audit["MVT_TIME_UTC_mvt"], utc=True, errors="coerce"
        )

        audit["month"] = audit["scheduled_time"].dt.strftime("%Y-%m")

        # Keep rows with an observed target.
        audit = audit[audit["target_sec"].notna()].copy()

        # Target-distribution audit by month and airport.
        audit["over_1h"] = audit["target_sec"] > 3600
        audit["over_3h"] = audit["target_sec"] > 10800
        audit["negative"] = audit["target_sec"] < 0

        summary = (
            audit.groupby(["month", "airport"], observed=True)
            .agg(
                n=("target_sec", "size"),
                median_sec=("target_sec", "median"),
                p99_sec=("target_sec", lambda s: s.quantile(0.99)),
                max_sec=("target_sec", "max"),
                over_1h=("over_1h", "sum"),
                over_3h=("over_3h", "sum"),
                negative=("negative", "sum"),
            )
            .reset_index()
        )

        print(
            summary[
                (summary["over_3h"] > 0) | (summary["negative"] > 0)
            ].to_string(index=False)
        )

        audit["timestamp_diff_sec"] = (
            audit["movement_time"] - audit["block_time"]
        ).dt.total_seconds()
        
        audit["target_minus_timestamp_diff"] = (
            audit["target_sec"] - audit["timestamp_diff_sec"]
        )
        
        print(
            audit.loc[
                audit["target_sec"] > 10800,
                [
                    "month",
                    "airport",
                    "MVT_ID_mvt",
                    "SCHED_TIME_UTC_mvt",
                    "BLOCK_TIME_UTC_mvt",
                    "MVT_TIME_UTC_mvt",
                    "target_sec",
                    "timestamp_diff_sec",
                    "target_minus_timestamp_diff",
                ],
            ]
            .sort_values("target_sec", ascending=False)
            .head(50)
            .to_string(index=False)
        )


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
