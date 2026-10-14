
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
train_dir = ROOT / "data" / "training"
if not train_dir.exists():
    train_dir = ROOT / "data"

files = sorted(train_dir.glob("training_*.parquet"))
if not files:
    raise FileNotFoundError(f"No training parquet files found in {train_dir}")

df = pd.concat(
    [pd.read_parquet(path) for path in files],
    ignore_index=True,
)

# Taxi-out targets exist only for departures.
df = df.loc[df["PHASE_mvt"].eq("DEP")].copy()
df = df.loc[df["TAXITIME_SEC_mvt"].notna()].copy()

# Event time is used ONLY to create historical folds.
df["event_time"] = pd.to_datetime(
    df["MVT_TIME_UTC_mvt"], utc=True, errors="coerce"
)
df = df.loc[df["event_time"].notna()].copy()
df["event_month"] = df["event_time"].dt.to_period("M")

# Scheduled time is a candidate predictor only if it is populated
# and known at the intended prediction time.
df["scheduled_time"] = pd.to_datetime(
    df["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce"
)
df["scheduled_hour"] = df["scheduled_time"].dt.hour
df["airport"] = df["ADEP_mvt"].fillna("__MISSING__").astype(str)

def score_fold(train, valid, fold_name):
    y_train = train["TAXITIME_SEC_mvt"].astype(float)
    y_valid = valid["TAXITIME_SEC_mvt"].astype(float)

    global_median = y_train.median()

    airport_median = train.groupby("airport")[
        "TAXITIME_SEC_mvt"
    ].median()

    airport_hour_median = train.groupby(
        ["airport", "scheduled_hour"],
        dropna=False,
    )["TAXITIME_SEC_mvt"].median()

    # Baseline A: airport median.
    pred_a = valid["airport"].map(airport_median).fillna(global_median)

    # Baseline B: airport + scheduled hour, with fallbacks.
    keys = pd.MultiIndex.from_frame(
        valid[["airport", "scheduled_hour"]]
    )
    pred_b = pd.Series(
        airport_hour_median.reindex(keys).to_numpy(),
        index=valid.index,
    )
    pred_b = pred_b.fillna(valid["airport"].map(airport_median))
    pred_b = pred_b.fillna(global_median)

    for name, pred in [
        ("airport_median", pred_a),
        ("airport_hour_median", pred_b),
    ]:
        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        mae = mean_absolute_error(y_valid, pred)
        print(
            f"{fold_name:>7} | {name:<22} | "
            f"RMSE={rmse:8.2f} sec | MAE={mae:8.2f} sec"
        )

    print(
        f"         validation rows={len(valid):,}; "
        f"negative targets={(y_valid < 0).sum():,}; "
        f"targets >2h={(y_valid > 7200).sum():,}"
    )

# Expanding-window folds: all training dates precede validation dates.
for month in ["2025-10", "2025-11", "2025-12"]:
    validation_month = pd.Period(month, freq="M")

    train = df.loc[df["event_month"] < validation_month]
    valid = df.loc[df["event_month"] == validation_month]

    if train.empty or valid.empty:
        print(f"Skipping {month}: empty training or validation fold")
        continue

    score_fold(train, valid, month)
