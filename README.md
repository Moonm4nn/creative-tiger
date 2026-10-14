# PRC Data Challenge 2026

A reproducible Python pipeline for predicting departure taxi time and generating submission-ready Parquet files for the PRC Data Challenge 2026.

## Project aim

The target is `TAXITIME_SEC_mvt`, the taxi time in seconds. The model uses historical flight and movement data from 2025 and predicts the later ranking period in 2026.

The project has two goals:

- Build a useful taxi-time regression model.
- Produce valid submissions without changing the template IDs, columns, row count, or row order.

The local ranking data covers 2026-01-01 through 2026-07-31. The training labels end on 2025-12-31, so the actual 2026 competition RMSE is not available locally.

## Repository layout

```text
.
├── data/
│   ├── training/                  # 12 monthly training Parquet files
│   ├── ranking.parquet            # Standard prediction features
│   ├── submitting.parquet         # Standard submission template
│   ├── final_ranking.parquet      # Final prediction features
│   └── final_submitting.parquet   # Final submission template
├── outputs/                       # Reports and generated submissions
├── src/
│   ├── train_submit.py            # Reproducible end-to-end pipeline
│   ├── baseline_validation.py     # Median baseline comparison
│   ├── feature_experiments.py     # Progressive CatBoost features
│   ├── feature_experiment2.py    # Alternative feature experiment
│   ├── feature_experiment3.py    # Operator model and anomaly inspection
│   ├── inspect_anomally.py        # Extreme-target investigation
│   ├── inspect_data.py            # Training-data inspection
│   └── inspect_ranking.py         # Ranking/template inspection
├── NOTES.md                       # Exploratory narrative and decisions
└── explore.py                     # Initial data exploration
```

## Environment

The project currently uses the existing local virtual environment at `.venv`. The main runtime dependencies are:

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- PyArrow

Activate the environment in PowerShell from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
```

If a new environment is needed, install the dependencies with:

```powershell
python -m pip install pandas numpy scikit-learn catboost pyarrow
```

## Methodology

### Training population

The pipeline loads all twelve monthly files in `data/training/`, keeps departure rows where `PHASE_mvt == "DEP"`, and parses `MVT_TIME_UTC_mvt` only to create chronological validation folds.

Invalid or missing targets are removed. Negative targets are reported and excluded from fitting because a negative taxi time is physically implausible. Validation observations are not filtered on that basis, so the reported scores still reflect the available data as it is.

### Features

The production model uses five categorical features:

- Departure airport from `ADEP_mvt`
- Scheduled hour from `SCHED_TIME_UTC_mvt`
- Scheduled weekday from `SCHED_TIME_UTC_mvt`
- Aircraft type from `AIRCRAFT_TYPE_mvt`
- Aircraft operator from `AIRCRAFT_OPERATOR_flt`

Missing categorical values are represented by `__MISSING__`. The actual movement timestamp is used for time ordering only. It is not used as a feature because it would not be known at the intended prediction point.

### Model

The model is `CatBoostRegressor` with:

- 400 iterations
- Depth 8
- Learning rate 0.08
- RMSE loss
- Random seed 42
- CatBoost file writing disabled

### Outlier strategies

Targets above 7,200 seconds were investigated rather than automatically deleted. The pipeline compares three strategies on the same chronological validation rows:

- `keep`: retain all nonnegative targets.
- `exclude`: remove training targets above 7,200 seconds.
- `cap`: replace training targets above 7,200 seconds with 7,200 seconds.

The validation target is unchanged. The selected strategy is the one with the lowest pooled RMSE, with MAE reported as a complementary measure.

The current recorded comparison is:

| Strategy | Pooled RMSE | Pooled MAE | Typical RMSE |
| --- | ---: | ---: | ---: |
| `cap` | 420.63 sec | 229.55 sec | 336.04 sec |
| `keep` | 423.01 sec | 230.69 sec | 345.74 sec |
| `exclude` | 428.40 sec | 229.49 sec | 336.30 sec |

The selected strategy is currently `cap`.

### Chronological validation

Validation uses expanding historical folds:

- October 2025: train on earlier months, validate on October.
- November 2025: train on earlier months, validate on November.
- December 2025: train on earlier months, validate on December.

Each comparison fold uses the same 300,000-row training sample for a fair strategy comparison. The final model is trained using all usable historical training rows under the selected strategy.

These scores are historical backtest scores. They are not the final 2026 competition score because the 2026 target values are not included in the local ranking files.

## Reproduce the analysis

Run commands from the repository root using the project environment.

Inspect the data and schemas:

```powershell
.\.venv\Scripts\python.exe .\src\inspect_data.py
.\.venv\Scripts\python.exe .\src\inspect_ranking.py
.\.venv\Scripts\python.exe .\src\inspect_anomally.py
```

Run the exploratory baselines and feature experiments:

```powershell
.\.venv\Scripts\python.exe .\src\baseline_validation.py
.\.venv\Scripts\python.exe .\src\feature_experiments.py
.\.venv\Scripts\python.exe .\src\feature_experiment2.py
.\.venv\Scripts\python.exe .\src\feature_experiment3.py
```

Run the production validation without generating submissions:

```powershell
.\.venv\Scripts\python.exe .\src\train_submit.py --validate-only
```

Run the complete pipeline:

```powershell
.\.venv\Scripts\python.exe .\src\train_submit.py
```

The output directory can be changed when needed:

```powershell
.\.venv\Scripts\python.exe .\src\train_submit.py --output-dir .\outputs
```

## Generated artifacts

The complete run creates:

- `outputs/submitting.parquet` — standard submission.
- `outputs/final_submitting.parquet` — final submission.
- `outputs/strategy_monthly.csv` — fold-by-fold strategy metrics.
- `outputs/strategy_summary.csv` — pooled strategy comparison.

The submission writer validates that:

- Every template ID is present in the matching ranking data.
- IDs are non-null and unique.
- Template row order is unchanged.
- Output columns match the template.
- Output row counts match the template.
- Predictions are finite, non-null, and nonnegative.

## Presentation notes

The main story of the project is the progression from inspection to a defensible production decision:

1. Inspect the schema and target distribution.
2. Establish simple airport-based baselines.
3. Add scheduled timing and aircraft context to a categorical CatBoost model.
4. Investigate negative and unusually long taxi times.
5. Compare keeping, excluding, and capping extreme training targets.
6. Select the strategy using chronological validation.
7. Train on all available history and preserve the submission contract.

More of the exploratory reasoning is recorded in [NOTES.md](NOTES.md).

## Limitations

- The final 2026 target values are unavailable, so actual submission RMSE cannot be calculated locally.
- Only the final quarter of 2025 is used for the current strategy comparison.
- The validation sample cap may affect the exact ranking of close strategies.
- Extreme values may represent real operational delays or unresolved timestamp/data-definition issues.
- There is no dependency lockfile or serialized model artifact yet.
