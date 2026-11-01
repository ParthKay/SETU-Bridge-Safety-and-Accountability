# Zero-Trust Bridge Survival & Audit Verification Engine

Climate-adaptive survival analysis and paper-inspection anomaly detection for
Indian bridge portfolios. The project is a strictly classical-ML pipeline: Cox
counting-process models, survival trees, isolation forests, and SHAP. No
deep-learning framework is used.

## Why this exists

Roughly 80% of Indian bridge failures are hydraulic: monsoon flooding, riverbed
scouring, and inadequate waterway capacity. Many failures occur on bridges whose
paper inspection still reports a healthy condition. This project separates two
questions:

1. **What does physics and climate history imply about survival?**  
   A longitudinal, climate-aware survival engine estimates time-to-failure from
   objective structural, traffic, rainfall, discharge, and scour data.
2. **Does the human inspection claim agree?**  
   A zero-trust audit compares the physics-implied survival curve with the
   reported condition rating and flags healthy-looking bridges with high
   physical collapse risk.

After a flood or cyclone, the same engine updates the climate state per district,
recomputes survival curves, and quantifies the drop in remaining useful life
(RUL). SHAP force plots explain which objective factors drive each immediate NDT
recommendation.

## Architecture

```
src/bridge_engine/
├── data_pipeline.py       Module 1: synthetic NBI/IBMS panel, Andersen-Gill
│                          splitter, climate joins, leakage-safe preprocessing,
│                          sksurv targets and lifelines counting-process frames
├── survival_models.py     Module 2: CoxTimeVaryingFitter, RandomSurvivalForest,
│                          GradientBoostingSurvivalAnalysis, rank ensemble,
│                          Harrell/Uno C-index and Integrated Brier Score
├── audit_verifier.py      Module 3: conditional survival audit horizon, robust
│                          Z-residual, IsolationForest, paper-inspection flags
└── triage_and_xai.py      Module 4: post-disaster shock updater, survival-curve
                           RUL, priority triage, SHAP force plots
```

### Module 1 — longitudinal climate counting process

- Generates a realistic synthetic NBI-style panel for 24 Indian districts
  including Morbi, Mumbai Suburban, Darbhanga, Cachar, Wayanad, and others.
- Models deck, superstructure, and substructure deterioration from age,
  material, design, traffic, truck share, waterway adequacy, and hydraulic
  exposure.
- Defines failure when the minimum condition rating reaches `<= 4`.
- Splits each bridge into one-year Andersen-Gill intervals
  `(id, t_start, t_stop, event)` and attaches point-in-time monsoon anomaly,
  rainfall, peak discharge, velocity, scour index, flood exceedance, and
  cumulative exposure.
- Performs bridge-level train/validation/test splitting *before* imputation and
  encoding. `IterativeImputer`, scaling, and one-hot encoding are fitted on the
  training split only.
- Emits leakage assertions, counting-process validity checks, `sksurv` structured
  targets, and `lifelines` counting-process frames.

### Module 2 — survival engine

- `CoxTimeVaryingFitter` with elastic-net stabilisation, collinearity pruning,
  hazard ratios, confidence intervals, and a standardised +1 SD / +2 SD
  monsoon/scour acceleration report. The composite shock is
  `exp(sum_i beta_i * delta_i)`, so it cannot fall below its positive individual
  components.
- `RandomSurvivalForest` with log-rank splitting and
  `GradientBoostingSurvivalAnalysis` with the Cox partial-likelihood loss.
- A rank-averaged survival ensemble.
- Evaluation uses only survival metrics: Harrell's C-index, Uno's IPCW C-index,
  Brier scores, and Integrated Brier Score at 5, 10, 15, 25, and 40 years.

### Module 3 — paper-inspection audit

- Uses only objective features (age, material, traffic, observed climate) to
  estimate conditional survival over the next audit horizon.
- Compares `9 * S(t0 + h | t0)` with the reported NBI condition.
- Fits a median/MAD Z-residual on the training split and an IsolationForest on
  objective climate/structural features plus physics scores.
- Primary flag: reported condition 7–9 while physics-implied survival is below
  `0.68` (`DEFAULT_CRITICAL_SURVIVAL`).
- Broader `AUDIT_ALERT` combines the primary rule with an Audit Z-score of
  `>= 1.5` (`DEFAULT_Z_FLAG_THRESHOLD`) and the IsolationForest flag.
- Ground truth is used only for post-hoc evaluation and never enters a feature
  matrix.

### Module 4 — post-disaster triage and XAI

- Applies district-, state-, or bridge-scoped flood/cyclone/scour shocks to the
  latest observable climate vector.
- Recomputes conditional survival curves and derives RUL from the first crossing
  of the survival threshold.
- Reports baseline RUL, post-event RUL, RUL loss, survival drop, priority, and
  immediate NDT recommendation.
- Joins Module 3 audit findings as escalation metadata only.
- Uses a model-agnostic `shap.Explainer` over survival risk predictions, writes
  local force plots as standalone HTML, and verifies SHAP additivity.

## Installation

Python 3.10+ is required; the project has been verified on Python 3.13.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`scikit-survival`, `lifelines`, `shap`, `pyarrow`, `joblib`, `matplotlib`.
`xgboost`, `lightgbm`, `statsmodels`, and `mapie` are available for extensions.

## Interactive portal

The Streamlit portal reads only persisted artifacts and fitted preprocessors.
It does not refit models.

```powershell
streamlit run app.py
```

Open `http://localhost:8501`. The portal contains the executive benchmark view,
landmark collapse archetypes, the live audit inspector and custom bridge scorer,
and the disaster simulator with SHAP explanations.

## Docker deployment

Docker Desktop must be running. Build and start the portal with Compose:

```powershell
docker compose up --build -d
```

Open `http://localhost:8501`. The container exposes a health endpoint at
`http://localhost:8501/_stcore/health`.

```powershell
docker compose logs -f
docker compose down
```

The Docker image uses `requirements-docker.txt`, which pins the runtime
versions used to generate and verify the persisted artifacts.

## Run the complete pipeline

Fast smoke run (about 40 seconds on a laptop):

```powershell
python scripts/run_end_to_end.py --quick
```

Standard run from `config/default.json` (1,200 bridges, 120 RSF trees, 100
boosting trees):

```powershell
python scripts/run_end_to_end.py
```

Custom run:

```powershell
python scripts/run_end_to_end.py `
  --config config/default.json `
  --n-bridges 2500 `
  --seed 42 `
  --district Darbhanga `
  --output-root artifacts
```

The script executes and persists all four modules, then reloads the fitted
artifacts from disk before running post-disaster triage. This verifies the
serialisation boundary as part of the end-to-end run.

## Technical report

The current Markdown technical report (version 0.2.0) is
`docs/TECHNICAL_REPORT.md`. It covers the folder map, data generation physics,
schema, module design, survival mathematics, the corrected shock scale, the
recalibrated audit thresholds, the landmark archetypes, inference, validation
results, and limitations.

A detailed 22-page PDF report can also be generated from the persisted run
artifacts:

```powershell
python scripts/generate_report_pdf.py
```

It covers the problem context, data generation physics, schema, module design,
survival model mathematics and rationale, audit logic, post-disaster RUL and
SHAP, validation results, limitations, hyperparameters, and an artifact
inventory. The output is
`Zero_Trust_Bridge_Survival_Engine_Technical_Report.pdf`.

## Outputs

```
artifacts/
├── data/
│   ├── bridges.parquet
│   ├── climate.parquet
│   ├── inspections.parquet
│   ├── EVAL_ONLY_ground_truth.parquet
│   ├── static_{train,validation,test}.parquet
│   ├── counting_{train,validation,test}.parquet
│   ├── sksurv_targets.npz
│   ├── static_preprocessor.joblib
│   ├── time_varying_preprocessor.joblib
│   ├── scoring/                # roster template + climate panel for new inputs
│   └── pipeline_audit.json
├── module2/
│   ├── survival_engine.joblib
│   └── survival_report.json
├── module3/
│   ├── audit_findings.parquet
│   ├── audit_models.joblib
│   └── audit_report.json
├── module4/
│   ├── triage_findings.parquet
│   ├── xai_contributions.parquet
│   ├── triage_report.json
│   ├── archetype_validation.json
│   └── force_plots/*.html
└── end_to_end_summary.json
```

Parquet is used when `pyarrow` is available; writers fall back to CSV
automatically.

## Module CLIs

The modules can also be run individually:

```powershell
$env:PYTHONPATH = "src"
python -m bridge_engine.data_pipeline --n-bridges 2000 --output-dir data/processed
python -m bridge_engine.survival_models --data-dir data/processed --output-dir artifacts/module2
python -m bridge_engine.audit_verifier --data-dir data/processed --engine-dir artifacts/module2 --output-dir artifacts/module3
python -m bridge_engine.triage_and_xai --data-dir data/processed --engine-dir artifacts/module2 --audit-dir artifacts/module3 --district Darbhanga --output-dir artifacts/module4
```

If the survival engine artifacts are absent, the Module 4 CLI generates a small
synthetic cohort and fits a compact engine before triage.

## Landmark collapse archetypes

`inject_collapse_archetypes` appends three deterministic held-out cases to the
benchmark after the seeded generator has run:

| Bridge ID | Case | Design | Key stress |
|---|---|---|---|
| `ARCHETYPE-CSMT-2019` | Mumbai footbridge | Steel beam/girder | Coastal monsoon, pedestrian loading |
| `ARCHETYPE-MORBI-2022` | Morbi suspension bridge | Suspension, 1910 | Age > 100 years, deferred NDT |
| `ARCHETYPE-BIHAR-2024` | Darbhanga monsoon scour | Concrete girder | Waterway adequacy 3, scour-critical |

All three are forced into the test split, carry `REPORTED_LAST_MIN_COND = 8`
while their latent condition is <= 4, and are scored in
`artifacts/module4/archetype_validation.json`. The end-to-end summary reports
their five-year conditional survival, Audit Z-score, suspect flag, post-disaster
RUL loss, triage priority, and top SHAP risk driver under
`archetype_validation_table`.

## Score a bridge (inference)

The trained engine is reused through `scripts/score_bridges.py`. There are two
input modes.

Existing cohort rows (already preprocessed, no transform needed):

```powershell
python scripts/score_bridges.py --bridge-ids BR-0000038,BR-0000162 --rul
```

New inventory roster (raw columns, transformed with the training-fitted
imputer/encoder):

```powershell
python scripts/score_bridges.py --export-sidecar
# edit artifacts/data/scoring/bridges_template.csv, then:
python scripts/score_bridges.py `
  --bridges artifacts/data/scoring/bridges_template.csv `
  --climate artifacts/data/scoring/climate.csv `
  --horizons 5,10,25,40 --rul-threshold 0.68 --output scores.csv
```

Output columns: `RISK_SCORE`, `SURVIVAL_AT_{h}Y` for each horizon, `RUL_YEARS`,
and `RUL_CENSORED`. `--model` accepts `cox_time_varying`,
`random_survival_forest`, `gradient_boosting`, or `ensemble`.
`--event-district Darbhanga --severity 1.35` applies a post-disaster shock before
scoring. `--print-schema` documents the exact roster and climate columns.

Scoring reuses the project's own `conditional_survival_curve` and
`compute_rul_from_survival`, so values match the audit and triage reports. For
artifacts generated before this script existed, re-run
`python scripts/run_end_to_end.py` once to persist the two preprocessor files.

## Tests

```powershell
python -m pytest -q
```

The suite covers:

- leakage allowlists and pre-entry climate joins;
- Andersen-Gill interval validity and event placement;
- survival-model metrics and probability monotonicity;
- audit calibration, flag rules, and ground-truth separation;
- disaster scoping, shock application, RUL interpolation, SHAP additivity, and
  artifact persistence;
- end-to-end artifact reload in the orchestration test path.

## Evaluation discipline

- No classification accuracy or standard RMSE is used for survival models.
- Harrell and Uno concordance and Brier/IBS are computed at fixed horizons.
- Ground-truth paper-inspection labels are never model inputs.
- Reported condition ratings never enter the survival feature matrix.
- SHAP features are checked against the same truth/audit leakage prefixes.
- All random operations use explicit seeds (`random_state=42` by default).

## Reproducibility notes

- The synthetic generator is deterministic for a fixed seed and cohort size.
- The default seed is `42`.
- Imputation and encoding are fitted only on the training bridge split.
- The audit Z-residual and IsolationForest are fitted on the training split and
  evaluated on validation/test rows.
- The Module 4 ensemble SHAP predictor uses fixed background-calibrated risk
  scaling so SHAP attributions are additive and stable across explanation
  batches.

## Scope and limitations

The synthetic generator is a calibrated engineering benchmark, not a substitute
for a cleaned NBI/IBMS extract. It intentionally encodes the dominant failure
physics (hydraulic scour, flood exceedance, overload, aging materials) so the
entire pipeline can be exercised before replacing the generator with raw
inventory CSVs. The model outputs are decision support for NDT prioritisation,
not a certified safety determination.
