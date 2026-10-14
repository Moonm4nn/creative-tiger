# PRC Data Challenge 2026 - Working Notes

## Project aim

The aim is to predict `TAXITIME_SEC_mvt`, the taxi time in seconds, for the movement IDs in the challenge submission templates.

The training data is split into twelve monthly Parquet files from January through December 2025. The ranking data starts on 2026-01-01 and runs through 2026-07-31, so the final model is trained on 2025 history and predicts a later 2026 period.

The important practical requirement is not only to get a reasonable prediction. The output must keep the submission template's IDs, columns, row count, and original row order.

## First look at the data

The first inspection showed about 30 columns in the movement data. The useful fields are a mixture of movement information, scheduled flight information, airport information, and the taxi-time target.

The target is only relevant for departures, so the modeling work uses rows where `PHASE_mvt == "DEP"`.

The first distribution checks raised three questions:

- Some taxi-time values are negative. These are physically suspicious and should not be treated as ordinary training targets.
- The mean is higher than the median, which suggests a right-skewed target distribution.
- There are unusually long taxi times, including values above two hours. These could be genuine delays, timestamp problems, or another data-quality issue.

The negative values were not deleted from the validation view. They are reported as a data-quality finding and excluded from model fitting because they are not physically plausible taxi times.

## What the experiments were trying to answer

The experiments followed a fairly ordinary progression. First, I wanted to know whether a simple airport-level estimate was already enough. The baseline compared an airport median with an airport plus scheduled-hour median. This gives a useful reference point before adding a more complicated model.

The next question was whether the scheduled flight context and aircraft information improved the estimate. CatBoost was tested with progressively richer categorical feature sets. The useful features were:

- Departure airport: `ADEP_mvt`
- Scheduled hour, derived from `SCHED_TIME_UTC_mvt`
- Scheduled weekday, also derived from scheduled time
- Aircraft type: `AIRCRAFT_TYPE_mvt`
- Aircraft operator: `AIRCRAFT_OPERATOR_flt`

The strongest saved feature experiment was the five-feature `A4_operator` version. It produced a pooled RMSE of 414.29 seconds and a pooled MAE of 228.28 seconds in its own experiment. That experiment used a different training-row policy, so its score should not be compared as if it were the final production score.

The movement timestamp was deliberately used for chronological splitting, not as a prediction feature. The scheduled timestamp is available as planned flight information, while the actual movement timestamp is only known after the event and would leak future information into a real prediction.

## The outlier conversation

The very long targets created the main modeling decision. There were three reasonable choices:

1. Keep every valid nonnegative target.
2. Exclude training targets above 7,200 seconds.
3. Keep those rows but cap their training target at 7,200 seconds.

The validation comparison was designed to answer this with data rather than assuming that every extreme observation was wrong. The validation rows were kept unchanged, including extreme values, so that the difficult cases still affected the score.

The final comparison was:

| Strategy | Pooled RMSE (seconds) | Pooled MAE (seconds) | Typical RMSE (seconds) |
| --- | ---: | ---: | ---: |
| Cap above 7,200 | 420.63 | 229.55 | 336.04 |
| Keep all targets | 423.01 | 230.69 | 345.74 |
| Exclude above 7,200 | 428.40 | 229.49 | 336.30 |

The capped strategy won on pooled RMSE, so it is the strategy used by `train_submit.py`. The difference is not huge, which is a reminder that this is a modeling choice based on the available historical folds, not proof that every long taxi time is an error.

## Validation versus submission testing

The October, November, and December 2025 folds are backtests. For each fold, the model trains only on earlier months and predicts the selected month.

They are not the actual challenge test score. The ranking data contains 2026 rows, beginning on 2026-01-01, and the true taxi-time labels for those rows are not available locally. Therefore the current RMSE of 420.63 seconds is the best recorded historical validation estimate, not the final competition RMSE.

## Current pipeline

The final workflow is in [src/train_submit.py](src/train_submit.py). It:

1. Loads and validates all twelve monthly training files.
2. Keeps departure rows and removes invalid or negative targets from fitting.
3. Builds the five selected categorical features.
4. Compares keep, exclude, and cap strategies on chronological folds.
5. Selects the strategy with the lowest pooled RMSE.
6. Trains the selected model on all usable 2025 history.
7. Predicts both ranking files.
8. Aligns predictions to each submission template by `MVT_ID_mvt`.
9. Checks IDs, row order, schema, uniqueness, and missing predictions before writing output.

The generated files are:

- `outputs/submitting.parquet`
- `outputs/final_submitting.parquet`

The validation reports are:

- `outputs/strategy_monthly.csv`
- `outputs/strategy_summary.csv`

## Remaining limitations

- The actual 2026 submission RMSE cannot be measured without challenge labels.
- Validation covers only October through December 2025.
- Validation folds use a 300,000-row sample for a consistent comparison; the final model uses the full usable training history.
- Extreme taxi times may be genuine operational events or timestamp/data-definition problems. The current pipeline reduces their influence but does not explain their cause.
- The saved feature experiment outputs represent selected runs rather than a complete experiment log.
- There is no locked dependency file yet. The existing `.venv` is the reproducible environment used for the current results.