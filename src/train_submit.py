from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "training"
OUTPUT_DIR = ROOT / "outputs"

TARGET = "TAXITIME_SEC_mvt"
ID_COLUMN = "MVT_ID_mvt"
OUTLIER_THRESHOLD = 7_200.0
MAX_TRAIN_ROWS = 300_000
RANDOM_SEED = 42
ITERATIONS = 400
DEPTH = 8
LEARNING_RATE = 0.08
VALIDATION_MONTHS = ("2025-10", "2025-11", "2025-12")
FEATURES = (
    "airport",
    "scheduled_hour",
    "scheduled_dow",
    "aircraft_type",
    "operator",
)
CATEGORICAL_FEATURES = list(FEATURES)
REQUIRED_TRAINING_COLUMNS = {
    "PHASE_mvt",
    TARGET,
    ID_COLUMN,
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "ADEP_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate, train, and generate PRC taxi-time submissions."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run chronological strategy validation without writing submissions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for submissions and validation reports.",
    )
    return parser.parse_args()


def discover_training_files() -> list[Path]:
    files = sorted(TRAIN_DIR.glob("training_*.parquet"))
    if len(files) != 12:
        raise FileNotFoundError(
            f"Expected 12 training files in {TRAIN_DIR}, found {len(files)}"
        )
    return files


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_training_data() -> pd.DataFrame:
    files = discover_training_files()
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        require_columns(frame, REQUIRED_TRAINING_COLUMNS, path.name)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data["PHASE_mvt"].eq("DEP")].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data["event_time"] = pd.to_datetime(
        data["MVT_TIME_UTC_mvt"], utc=True, errors="coerce"
    )
    data["event_month"] = (
        data["event_time"].dt.tz_localize(None).dt.to_period("M")
    )

    invalid_target = data[TARGET].isna() | ~np.isfinite(data[TARGET])
    missing_event_time = data["event_month"].isna()
    print(
        f"Loaded {len(data):,} departure rows from {len(files)} training files."
    )
    print(
        f"Invalid targets: {invalid_target.sum():,}; "
        f"negative targets: {(data[TARGET] < 0).sum():,}; "
        f"missing event timestamps: {missing_event_time.sum():,}"
    )

    data = data.loc[~invalid_target & ~missing_event_time].copy()
    if data.empty:
        raise ValueError("No usable dated departure training rows remain.")
    return data


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "SCHED_TIME_UTC_mvt",
        "ADEP_mvt",
        "AIRCRAFT_TYPE_mvt",
        "AIRCRAFT_OPERATOR_flt",
    }
    require_columns(frame, required, "Prediction data")

    scheduled = pd.to_datetime(
        frame["SCHED_TIME_UTC_mvt"], utc=True, errors="coerce"
    )
    features = pd.DataFrame(index=frame.index)
    features["airport"] = frame["ADEP_mvt"].fillna("__MISSING__").astype(str)
    features["scheduled_hour"] = scheduled.dt.hour.map(
        lambda value: str(int(value)) if pd.notna(value) else "__MISSING__"
    )
    features["scheduled_dow"] = scheduled.dt.dayofweek.map(
        lambda value: str(int(value)) if pd.notna(value) else "__MISSING__"
    )
    features["aircraft_type"] = (
        frame["AIRCRAFT_TYPE_mvt"].fillna("__MISSING__").astype(str)
    )
    features["operator"] = (
        frame["AIRCRAFT_OPERATOR_flt"].fillna("__MISSING__").astype(str)
    )
    return features.loc[:, FEATURES]


def make_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def prepare_strategy_training(
    train: pd.DataFrame, strategy: str
) -> tuple[pd.DataFrame, pd.Series, int]:
    usable = train.loc[train[TARGET] >= 0].copy()
    excluded = 0
    if strategy == "exclude":
        excluded = int((usable[TARGET] > OUTLIER_THRESHOLD).sum())
        usable = usable.loc[usable[TARGET] <= OUTLIER_THRESHOLD].copy()
    elif strategy == "cap":
        excluded = int((usable[TARGET] > OUTLIER_THRESHOLD).sum())
        usable[TARGET] = usable[TARGET].clip(upper=OUTLIER_THRESHOLD)
    elif strategy != "keep":
        raise ValueError(f"Unknown strategy: {strategy}")

    if usable.empty:
        raise ValueError(f"Strategy {strategy} has no usable training rows.")
    if len(usable) > MAX_TRAIN_ROWS:
        usable = usable.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    return usable, usable[TARGET].astype(float), excluded


def score_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    result = {
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted))),
        "mae": float(mean_absolute_error(actual_values, predicted)),
    }
    typical = actual_values <= OUTLIER_THRESHOLD
    result["typical_rmse"] = (
        float(np.sqrt(mean_squared_error(actual_values[typical], predicted[typical])))
        if typical.any()
        else float("nan")
    )
    return result


def validate_strategies(data: pd.DataFrame, output_dir: Path) -> str:
    monthly_results: list[dict[str, object]] = []
    prediction_records: list[pd.DataFrame] = []

    for month in VALIDATION_MONTHS:
        validation_period = pd.Period(month, freq="M")
        train = data.loc[data["event_month"] < validation_period]
        valid = data.loc[data["event_month"] == validation_period]
        if train.empty or valid.empty:
            raise ValueError(
                f"Validation month {month} has training={len(train):,}, "
                f"validation={len(valid):,}."
            )

        valid_features = build_features(valid)
        for strategy in ("keep", "exclude", "cap"):
            strategy_train, y_train, excluded = prepare_strategy_training(
                train, strategy
            )
            model = make_model()
            model.fit(
                build_features(strategy_train),
                y_train,
                cat_features=CATEGORICAL_FEATURES,
            )
            predictions = model.predict(valid_features)
            scores = score_predictions(valid[TARGET], predictions)
            monthly_results.append(
                {
                    "month": month,
                    "strategy": strategy,
                    "n_train": len(strategy_train),
                    "n_valid": len(valid),
                    "n_excluded_or_capped": excluded,
                    **scores,
                }
            )
            prediction_records.append(
                pd.DataFrame(
                    {
                        "month": month,
                        "strategy": strategy,
                        "actual": valid[TARGET].to_numpy(dtype=float),
                        "prediction": predictions,
                    }
                )
            )
            print(
                f"{month} {strategy:>7}: train={len(strategy_train):,}, "
                f"RMSE={scores['rmse']:.2f}, MAE={scores['mae']:.2f}, "
                f"typical_RMSE={scores['typical_rmse']:.2f}"
            )

    monthly = pd.DataFrame(monthly_results)
    predictions = pd.concat(prediction_records, ignore_index=True)
    summary_rows = []
    for strategy, group in predictions.groupby("strategy"):
        scores = score_predictions(group["actual"], group["prediction"].to_numpy())
        summary_rows.append({"strategy": strategy, **scores})
    summary = pd.DataFrame(summary_rows).sort_values(
        ["rmse", "mae"], ignore_index=True
    )
    selected = str(summary.iloc[0]["strategy"])

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "strategy_monthly.csv", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    print("\nPooled strategy results:")
    print(summary.to_string(index=False))
    print(f"Selected strategy: {selected}")
    return selected


def train_final_model(data: pd.DataFrame, strategy: str) -> CatBoostRegressor:
    train, targets, _ = prepare_strategy_training(data, strategy)
    model = make_model()
    model.fit(
        build_features(train),
        targets,
        cat_features=CATEGORICAL_FEATURES,
    )
    return model


def load_prediction_pair(
    ranking_path: Path, template_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = pd.read_parquet(ranking_path)
    template = pd.read_parquet(template_path)
    require_columns(ranking, {ID_COLUMN}, ranking_path.name)
    require_columns(template, {ID_COLUMN, TARGET}, template_path.name)
    build_features(ranking)

    for frame, label in ((ranking, ranking_path.name), (template, template_path.name)):
        if frame[ID_COLUMN].isna().any():
            raise ValueError(f"{label} contains null movement IDs.")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{label} contains duplicate movement IDs.")

    missing = template.loc[
        ~template[ID_COLUMN].isin(ranking[ID_COLUMN]), ID_COLUMN
    ]
    if not missing.empty:
        raise ValueError(
            f"{template_path.name} has {len(missing):,} IDs missing from "
            f"{ranking_path.name}."
        )
    return ranking, template


def predict_submission(
    model: CatBoostRegressor,
    ranking_path: Path,
    template_path: Path,
    output_path: Path,
) -> None:
    ranking, template = load_prediction_pair(ranking_path, template_path)
    ranking_features = build_features(ranking)
    predictions = model.predict(ranking_features)
    if not np.isfinite(predictions).all():
        raise ValueError(f"Model produced non-finite predictions for {ranking_path.name}.")
    predictions = np.maximum(predictions, 0.0)

    predicted_by_id = pd.Series(
        predictions, index=ranking[ID_COLUMN].to_numpy(), name=TARGET
    )
    aligned = template[ID_COLUMN].map(predicted_by_id)
    if aligned.isna().any():
        raise ValueError(f"Missing aligned predictions for {template_path.name}.")

    output = template.copy()
    output[TARGET] = aligned.to_numpy(dtype=np.float64)
    if list(output.columns) != list(template.columns):
        raise ValueError("Output columns differ from the submission template.")
    if len(output) != len(template) or output[ID_COLUMN].tolist() != template[ID_COLUMN].tolist():
        raise ValueError("Output does not preserve template row count or ID order.")
    if output[TARGET].isna().any() or not np.isfinite(output[TARGET]).all():
        raise ValueError(f"Output contains invalid predictions: {output_path}")
    if (output[TARGET] < 0).any():
        raise ValueError(f"Output contains negative predictions: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    print(
        f"Wrote {output_path} with {len(output):,} rows; "
        f"prediction range={output[TARGET].min():.1f}-{output[TARGET].max():.1f}"
    )


def main() -> None:
    args = parse_args()
    data = load_training_data()
    strategy = validate_strategies(data, args.output_dir)
    if args.validate_only:
        return

    model = train_final_model(data, strategy)
    predict_submission(
        model,
        ROOT / "data" / "ranking.parquet",
        ROOT / "data" / "submitting.parquet",
        args.output_dir / "submitting.parquet",
    )
    predict_submission(
        model,
        ROOT / "data" / "final_ranking.parquet",
        ROOT / "data" / "final_submitting.parquet",
        args.output_dir / "final_submitting.parquet",
    )


if __name__ == "__main__":
    main()