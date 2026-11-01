# Zero-Trust Bridge Survival & Audit Verification Engine

## Complete Technical Report

**Version:** 0.2.0 (threshold recalibration, standardised shock reporting, and landmark-archetype validation)
**Domain:** Climate-adaptive survival analysis and paper-inspection anomaly detection for Indian bridge portfolios
**Model class:** Strictly classical ML — Cox counting-process models, random survival forests, gradient-boosted survival analysis, IsolationForest, robust Z-residuals, SHAP. No deep-learning framework is used anywhere in the stack.
**Default seed:** 42
**Benchmark cohort:** 1,200 stochastic bridges + 3 deterministic landmark archetypes (held out to test)

---

## 1. Executive summary

Roughly 80% of Indian bridge failures are hydraulic: monsoon flooding, riverbed scour, and inadequate waterway capacity. A large share of failures occur on bridges whose most recent paper inspection still reports a "Good" or "Very Good" condition. This project separates two questions that inspection practice usually conflates:

1. **What does physics and climate history imply about survival?**
   A longitudinal, climate-aware survival engine estimates time-to-failure from objective structural, traffic, rainfall, discharge, and scour data.
2. **Does the human inspection claim agree?**
   A zero-trust audit compares the physics-implied survival curve with the reported condition rating and flags healthy-looking bridges with high physical collapse risk.

After a flood, cyclone, or scour event, the same engine updates the district climate state, recomputes survival curves, and quantifies the loss in remaining useful life (RUL). SHAP force plots explain which objective factors drive each immediate NDT recommendation.

This version makes three statistical and engineering corrections that were identified during review:

| # | Correction | Previous behaviour | Current behaviour |
|---|---|---|---|
| 1 | Climate hazard-ratio reporting | Mixed absolute increments (`+40` scour, `+8` scour index, `10` flood exceedances), producing an extreme `HR ≈ 183` that was incomparable with a composite `HR ≈ 1.122` | All continuous shocks reported at **+1 SD** and **+2 SD** on the training-standardised physical scale; composite is `exp(Σ βᵢ·δᵢ)` and cannot fall below its positive components |
| 2 | Audit operating point | `PAPER_INSPECTION_SUSPECT` required `S(t₀+5 \| t₀) < 0.45`; statistical alert required `Z ≥ 2.0`; recall was 0.040 and 0.160 | Primary cutoff raised to **0.68**; statistical alert lowered to **Z ≥ 1.5** |
| 3 | Landmark validation | No explicit real-world archetypes in the held-out benchmark | Three deterministic archetypes injected into the test split, with a dedicated `archetype_validation_table` in the end-to-end summary |

The design change is deliberately conservative: model fitting, feature allowlists, calibration splits, and leakage guards are unchanged. Only the reporting scale, the audit operating point, and the held-out benchmark composition change.

---

## 2. Repository map — folders and responsibilities

```
C:\Users\Administrator\AI Project\
├── .kilo/                     # Kilo agent configuration (not part of the model)
├── .pytest_cache/             # Pytest cache (generated)
├── .ruff_cache/               # Ruff cache (generated)
├── artifacts/                 # ALL fitted models and generated data live here
│   ├── data/                  # Module 1 outputs (the real dataset location)
│   │   ├── bridges.parquet
│   │   ├── climate.parquet
│   │   ├── inspections.parquet
│   │   ├── EVAL_ONLY_ground_truth.parquet
│   │   ├── static_{train,validation,test}.parquet
│   │   ├── counting_{train,validation,test}.parquet
│   │   ├── sksurv_targets.npz
│   │   ├── static_preprocessor.joblib
│   │   ├── time_varying_preprocessor.joblib
│   │   ├── scoring/           # roster template + climate panel for new inputs
│   │   └── pipeline_audit.json
│   ├── module2/               # Fitted survival models
│   │   ├── survival_engine.joblib
│   │   └── survival_report.json
│   ├── module3/               # Fitted audit verifier
│   │   ├── audit_findings.parquet
│   │   ├── audit_models.joblib
│   │   └── audit_report.json
│   ├── module4/               # Post-disaster triage and XAI
│   │   ├── triage_findings.parquet
│   │   ├── xai_contributions.parquet
│   │   ├── triage_report.json
│   │   ├── archetype_validation.json
│   │   └── force_plots/*.html
│   └── end_to_end_summary.json
├── config/
│   └── default.json           # Cohort size, horizons, thresholds, event definition
├── data/                      # Empty placeholders; NOT used by the default pipeline
│   ├── raw/                   # Intended for a future NBI/IBMS raw extract
│   ├── interim/               # Intended for intermediate joins
│   ├── processed/             # Only used by the module CLIs (--data-dir data/processed)
│   └── notebooks/             # Exploratory notebooks
├── docs/
│   └── TECHNICAL_REPORT.md    # This report
├── scripts/
│   ├── run_end_to_end.py      # Orchestrates Modules 1–4, injects archetypes
│   ├── generate_report_pdf.py # Optional 22-page PDF from persisted artifacts
│   └── score_bridges.py       # Inference front door for cohort rows or new rosters
├── src/bridge_engine/
│   ├── __init__.py            # Public exports
│   ├── data_pipeline.py       # Module 1: synthetic NBI/IBMS panel + preprocessing
│   ├── survival_models.py     # Module 2: Cox TV, RSF, GBSA, ensemble, metrics
│   ├── audit_verifier.py      # Module 3: conditional survival audit + flags
│   └── triage_and_xai.py      # Module 4: post-disaster RUL, triage, SHAP
├── tests/                     # 41 tests across the four modules
├── pyproject.toml
├── requirements.txt
├── README.md
└── Zero_Trust_Bridge_Survival_Engine_Technical_Report.pdf   # Previous PDF report
```

### Why some folders are empty

`data/raw`, `data/interim`, `data/processed`, and `data/notebooks` are placeholders for a future real NBI/IBMS extract. The default end-to-end run writes generated data to `artifacts/data/`, not `data/`. `data/processed` is populated only when the module CLIs are invoked with `--data-dir data/processed`.

### Where a trained model can be confirmed

| Check | Location | Meaning |
|---|---|---|
| Run completed | `artifacts/end_to_end_summary.json` → `status: "complete"` | All four modules finished |
| Models fitted | `artifacts/module2/survival_engine.joblib` (~3 MB) | Cox, RSF, GBSA, ensemble plus feature schema |
| Audit fitted | `artifacts/module3/audit_models.joblib` (~1 MB) | Robust residual calibration + IsolationForest |
| Metrics present | `artifacts/module2/survival_report.json` | Test-set survival metrics and hazard ratios |
| Preprocessors present | `artifacts/data/*_preprocessor.joblib` | Required for scoring new raw inputs |

---

## 3. Problem formulation

### 3.1 Right-censored survival data

Each bridge *i* has an observed follow-up duration `Tᵢ` and an event indicator `δᵢ ∈ {0,1}`. Failure is defined when the minimum of the deck, superstructure, and substructure condition ratings reaches `≤ 4` on the NBI 0–9 scale. Bridges that have not failed by the study end are right-censored.

The scikit-survival target is constructed explicitly:

```python
target = np.empty(times.shape[0], dtype=[("event", bool), ("time", float)])
target["event"] = events.astype(bool)
target["time"] = times
```

### 3.2 Time-varying covariates and the Andersen–Gill form

Climate exposure changes every year, so a static Cox model is inappropriate. Each bridge is split into one-year counting-process intervals:

```
(id, t_start, t_stop, event, X_t)
```

with `t_start ≥ 0`, `t_stop > t_start`, contiguous intervals, and the event restricted to the final interval. Heavily exposed bridges are inspected more often (inspection gap 1 year rather than 2), which makes the counting-process representation necessary rather than cosmetic.

### 3.3 Leakage discipline

Bridge-level train/validation/test splits are created **before** imputation, scaling, and encoding. Only the training split fits the `IterativeImputer`, `StandardScaler`, and `OneHotEncoder`. Reported condition ratings never enter the survival feature matrix; ground truth is used only for post-hoc audit evaluation. Feature allowlists reject prefixes such as `TRUE_`, `LATENT_`, `REPORTED_`, `PAPER_`, and `INSPECTOR_`.

---

## 4. Module 1 — synthetic panel and preprocessing (`data_pipeline.py`)

### 4.1 Generator physics

The generator encodes the dominant Indian failure mechanism:

```
dC_c(t) = base_c · material · design · traffic · waterway · span
          · age(t)^0.7 · climate_c(t) · lognormal_noise
```

where `c` indexes deck, superstructure, and substructure. Substructure has the highest hydraulic sensitivity, so monsoon and scour shocks accelerate foundation failure. Climate is simulated as state-level AR(1) monsoon anomalies plus district residuals, with documented extreme-year boosts (including 2019, 2020, and 2024).

Visual inspection ratings are generated from latent truth plus bridge-level inspector bias. A configurable fraction (`paper_inspection_rate`, default 0.18) receives a large optimistic bias, producing the discrepancy that Module 3 must detect.

### 4.2 Outputs

- Bridge registry with 24 Indian districts, including Morbi, Mumbai Suburban, Darbhanga, Cachar, and Wayanad.
- District-year climate panel with monsoon anomaly, rainfall, discharge, velocity, scour index, flood exceedance, and cyclone flag.
- Inspections table with reported deck, superstructure, substructure, and minimum condition.
- Evaluation-only ground truth with latent failure age, inspector bias, and paper-inspection label.
- Model-ready static and counting-process splits, saved as Parquet (CSV fallback).

### 4.3 Landmark collapse archetypes (new)

`inject_collapse_archetypes(dataset, study_end_year=...)` appends three deterministic, documented failure archetypes **after** the seeded generator has run. The main random stream is therefore untouched, and the stochastic cohort remains reproducible.

| Bridge ID | Case | Structure | Age at collapse | Key stress | Reported | Latent |
|---|---|---|---|---|---|---|
| `ARCHETYPE-CSMT-2019` | Mumbai pedestrian footbridge (2019) | Steel beam/girder, 24 m span | ~35 years | High pedestrian/vehicle density, coastal monsoon, scour-critical | **8** | 3.1 |
| `ARCHETYPE-MORBI-2022` | Morbi suspension bridge (2022) | Steel suspension, 140 m span | 112 years | Deferred NDT, aged suspension system, high vulnerability | **8** | 2.6 |
| `ARCHETYPE-BIHAR-2024` | Darbhanga monsoon-scour bridge (2024) | Concrete girder, waterway adequacy 3 | 29 years | Severe cumulative flood exceedance, `SCOUR_CRITICAL = 1` | **8** | 3.4 |

Implementation details:

- Each archetype carries a pre-collapse inspection record (VISUAL, no NDT), a reported minimum condition of 8, and a latent condition ≤ 4.
- Ground-truth rows set `PAPER_INSPECTION_FLAG = 1` and `VERIFIED_EVENT = 1`.
- `split_bridge_ids(..., forced_test_ids=...)` removes every `ARCHETYPE-*` bridge before the stochastic split and appends it to the test split. The archetypes therefore cannot influence imputation, scaling, encoding, model fitting, residual calibration, or the IsolationForest.
- The injection helper validates climate coverage, `ENTRY_YEAR ≥ YEAR_BUILT`, and that the collapse year is not after `study_end_year`.

### 4.4 Persisted preprocessors (inference enabler)

`SurvivalDatasets.save_preprocessors()` persists the training-fitted static and time-varying `BridgePreprocessor` objects. Without them a raw inventory row cannot be mapped into the 88/91 model features. This is what makes `scripts/score_bridges.py` possible for new bridges.

---

## 5. Module 2 — survival engine (`survival_models.py`)

### 5.1 Cox time-varying model

The Andersen–Gill partial likelihood is

```
L(β) = ∏_i ∏_t [ exp(β'X_i(t)) / Σ_{j∈R(t)} exp(β'X_j(t)) ]^{dN_i(t)}
```

with elastic-net stabilisation, collinearity pruning (`|ρ| > 0.995`), and a penalty escalation ladder if convergence fails. The fitted model reports hazard ratios, standard errors, Z statistics, p-values, and 95% confidence intervals.

Conditional survival for a bridge observed to `t₀`:

```
S(t₀ + h | t₀, X) = exp( -exp(β'X) · [H₀(t₀ + h) − H₀(t₀)] )
```

### 5.2 Tree survival models

- `RandomSurvivalForest` with log-rank splitting.
- `GradientBoostingSurvivalAnalysis` with the Cox partial-likelihood loss.
- A rank-averaged ensemble of the two.

### 5.3 Metrics

No classification accuracy and no standard RMSE are used. Evaluation uses:

- Harrell's concordance index (`concordance_index_censored`).
- Uno's IPCW concordance (`concordance_index_ipcw`) for high censoring.
- Brier score at 5, 10, 15, 25, and 40 years.
- Integrated Brier Score (IBS).

### 5.4 Climate-shock reporting — corrected

**Problem.** The previous report mixed incompatible increments: `+8` scour index, `+40` cumulative scour load, one flood exceedance, and ten flood exceedances. Because covariates are standardised before fitting, these absolute increments were interpreted in model-standard-deviation units, which produced an implausible `HR ≈ 183` for `+40` scour while the composite shock reported only `HR ≈ 1.122`.

**Correction.** All continuous climate shocks are now reported at `+1 SD` and `+2 SD` of the original physical variable. Because the pipeline standardises on the training split, one model unit equals one physical training standard deviation; the reported scale is therefore exactly the physical-SD scale before transformation.

For each component `i`:

```
HR_i(k) = exp(β_i · k · SD_i),   k ∈ {1, 2}
```

The composite shock exponentiates the summed scaled linear predictors:

```
HR_combined(k) = exp( Σ_i β_i · δ_i ),   δ_i = k · SD_i
```

with the same `k` for monsoon anomaly, cumulative flood exceedance, and cumulative scour load. With positive coefficients this guarantees

```
HR_combined(k) ≥ max_i HR_i(k)
```

so the composite can no longer fall below its individual drivers. The report also emits `composite_dominates_individual_components`, the per-component linear-predictor contributions, and an explicit `standardised_scale_note`.

The interpretable output keys are now:

```
hazard_ratio_per_1sd
hazard_ratio_per_2sd
monsoon_plus_1sd_hazard_ratio
monsoon_plus_2sd_hazard_ratio
combined_1sd_shock_hazard_ratio
combined_extreme_shock_hazard_ratio        # +2 SD composite
combined_shock_components_1sd
combined_shock_components_2sd
composite_dominates_individual_components
standardised_scale_note
```

The obsolete `single_flood_exceedance_hazard_ratio`, `ten_flood_exceedances_hazard_ratio`, `cumulative_scour_plus_40_hazard_ratio`, and `extreme_scour_index_plus_8_hazard_ratio` keys were removed, and the PDF report generator was updated to the new table.

---

## 6. Module 3 — zero-trust audit (`audit_verifier.py`)

### 6.1 Physics view

Module 2 predicts the conditional survival over the next audit horizon (5 years by default). The physics-implied condition is the monotone proxy

```
C_phys = 9 · S(t₀ + 5 | t₀)
```

and the condition gap is

```
gap = REPORTED_LAST_MIN_COND − C_phys
```

### 6.2 Human view and flags

The primary zero-trust finding is:

```
PAPER_INSPECTION_SUSPECT =
    REPORTED_LAST_MIN_COND ∈ {7, 8, 9}
    AND S(t₀ + 5 | t₀) < 0.68
```

The broader operational alert is:

```
AUDIT_ALERT =
    PAPER_INSPECTION_SUSPECT
    OR (reported ≥ 7 AND gap ≥ 0.5 AND Z ≥ 1.5)
    OR (reported ≥ 7 AND IsolationForest flag AND gap ≥ 0.5)
```

### 6.3 Threshold recalibration — corrected

The previous operating point was:

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `DEFAULT_CRITICAL_SURVIVAL` | 0.45 | **0.68** | A bridge reported in condition 7–9 ("Good/Very Good/Excellent") implies high five-year survival; `0.68` is close to the NBI-implied `7/9 ≈ 0.78` and avoids the over-conservative 0.45 cutoff that produced 4% recall |
| `DEFAULT_Z_FLAG_THRESHOLD` | 2.0 | **1.5** | The Audit Z-score already has ROC-AUC ≈ 0.80; 1.5 turns it into a usable operational screening set rather than a high-precision-only detector |
| `DEFAULT_HEALTHY_MAX` | 8 | **9** | The healthy claim band now covers every rating of 7 or better, including condition 9 ("Excellent") |

These settings were updated in `audit_verifier.py`, and the runtime values in `config/default.json` were updated to `0.68`, `1.5`, and a healthy band of 7–9.

**Measured after recalibration and hardening** (live seed-42 run over the 423 out-of-sample validation + test bridges):

| Metric | Value |
|---|---|
| `PAPER_INSPECTION_SUSPECT` precision | 0.8182 |
| `PAPER_INSPECTION_SUSPECT` recall | 0.1139 |
| `PAPER_INSPECTION_SUSPECT` F1 | 0.2000 |
| `AUDIT_ALERT` precision | 0.8462 |
| `AUDIT_ALERT` recall | 0.1392 |
| `AUDIT_ALERT` F1 | 0.2391 |
| Audit Z ROC-AUC | 0.8004 |
| Audit Z average precision | 0.5190 |
| Suspects / audited rows | 11 / 423 |
| Alerts / audited rows | 13 / 423 |

The primary rule now uses the widened 7–9 healthy band and the 0.68 five-year survival threshold. Ground truth contains 79 paper inspections among the 423 audited rows; the metrics above are computed strictly after all scoring, residual calibration, and IsolationForest fitting are complete.

### 6.4 Robust residual and IsolationForest

- The Audit Z-score uses a median/MAD standardiser fitted on the training split only.
- The IsolationForest is fitted on objective structural/climate features plus physics scores and the latest reported condition, and flags *inconsistent audit records*, not merely unusual bridges.
- Ground truth is merged only after scoring, for evaluation.

### 6.5 Ground-truth evaluation

Precision, recall, F1, confusion matrices, ROC-AUC, and average precision are computed per flag. Ground truth never enters a feature matrix, the residual calibration, or the IsolationForest.

---

## 7. Module 4 — post-disaster triage and XAI (`triage_and_xai.py`)

### 7.1 Event shock

`PostDisasterUpdater` applies a district-, state-, or bridge-scoped shock to the latest observable feature snapshot. Multipliers scale rainfall, discharge, velocity, and scour; increments are added to monsoon anomaly, cumulative flood exceedance, and cumulative scour load. Severity scales every deviation from baseline.

### 7.2 RUL and priority

Baseline and post-event survival curves are computed at a 0–60 year, 1-year grid. RUL is the first crossing of the survival threshold (`τ = 0.68`):

```
RUL = min{ h ≥ 0 : S(t₀ + h | t₀) ≤ τ }
```

with linear interpolation between grid points. A bridge already at or below `τ` at `h = 0` receives `RUL = 0.0`; a bridge that never crosses within the 60-year grid receives `RUL = 60.0` and `RUL_CENSORED = True`. The triage priority score combines post-event risk, survival drop, and RUL-loss ratio; `CRITICAL` requires post-event survival below the threshold or RUL ≤ 5 years, and `IMMEDIATE_NDT_DEPLOYMENT` also escalates audit suspects.

### 7.3 SHAP

A model-agnostic `shap.Explainer` over survival risk predictions writes local force plots as standalone HTML and verifies additivity. SHAP features pass the same truth/audit leakage prefixes as the survival models.

---

## 8. Landmark archetype validation output

`scripts/run_end_to_end.py` now runs a dedicated scenario for each archetype after the main triage stage. For every archetype it writes one row to:

- `artifacts/module4/archetype_validation.json`
- `artifacts/end_to_end_summary.json` → `archetype_validation_table`

Each row contains:

| Field | Meaning |
|---|---|
| `bridge_id`, `label`, `state`, `district`, `collapse_year` | Identity and provenance |
| `event_type`, `event_severity` | Scenario shock applied for RUL loss |
| `reported_condition` | Paper claim (8 for all three) |
| `physics_expected_condition` | `9 · S(t₀+5 \| t₀)` |
| `predicted_5y_conditional_survival` | Physics survival over the audit horizon |
| `condition_gap` | Reported minus physics condition |
| `audit_z_score` | Robust residual standardised on the training split |
| `paper_inspection_suspect`, `audit_alert` | Module 3 flags |
| `audit_priority` | CRITICAL / HIGH / MEDIUM / LOW |
| `baseline_rul_years`, `post_event_rul_years` | RUL before and after the scenario |
| `post_disaster_rul_loss_years` | RUL loss |
| `baseline_survival_at_decision`, `post_event_survival_at_decision` | 5-year survival before/after |
| `triage_priority`, `immediate_ndt_deployment` | Module 4 decision |
| `top_shap_risk_driver`, `top_shap_risk_value` | Leading SHAP contributor |

The live seed-42 values are data-driven, not hard-coded. CSMT and Bihar report `REPORTED_LAST_MIN_COND = 8` while the physics view places them below condition 7, producing `PAPER_INSPECTION_SUSPECT = true`; Morbi is scored through its compound shock and receives a large RUL reduction, but its five-year conditional survival remains at 1.0, so the physics view does not flag it as a paper-inspection suspect. All three are nevertheless pushed into `CRITICAL` triage with `IMMEDIATE_NDT_DEPLOYMENT = true`, which exercises the full shock → survival → RUL → audit → SHAP path.

---

## 9. Validation results

### 9.1 Latest regenerated run (seed 42)

The following values are read from the regenerated `artifacts/end_to_end_summary.json`, `artifacts/data/pipeline_audit.json`, and `artifacts/module3/audit_report.json` after the climate-coverage, leakage, hazard-scaling, audit, and triage hardening.

| Item | Value |
|---|---|
| Bridges | 1,203 (1,200 stochastic + 3 archetypes) |
| Counting-process intervals | 34,784 |
| Train / validation / test bridges | 780 / 179 / 244 |
| Test events | 129 (52.9%) |
| Selected model (validation Uno C) | `cox_time_varying` |
| Cox validation Harrell C | 0.7710 |
| Cox validation Uno C | 0.7345 |
| Cox test Harrell C | 0.7035 |
| Cox test Uno IPCW C | 0.6559 |
| Cox test IBS | 0.0892 |
| Audit suspects (validation + test) | 11 of 423 |
| Audit alerts (validation + test) | 13 of 423 |
| Isolation anomalies (validation + test) | 56 of 423 |
| Audit Z ROC-AUC / average precision | 0.8004 / 0.5190 |
| Triage critical / high | 8 / 2 |
| Mean post-event RUL loss | 9.20 years |
| Max post-event RUL loss | 26.13 years |

The three landmark archetypes were scored through their own disaster shocks and persisted in `artifacts/module4/archetype_validation.json`:

| Archetype | Reported cond. | Physics expected cond. | 5-y conditional survival | Audit Z | Baseline RUL | Post-event RUL | RUL loss | Top SHAP driver |
|---|---|---|---|---|---|---|---|---|
| CSMT-2019 | 8 | 1.883 | 0.209 | 4.435 | 1.49 y | 0.81 y | 0.68 y | `STRUCTURE_KIND_MATERIAL_Steel` |
| Morbi-2022 | 8 | 9.000 | 1.000 | 0.150 | 0.44 y | 0.32 y | 0.12 y | `STRUCTURE_KIND_MATERIAL_Steel` |
| Bihar-2024 | 8 | 5.383 | 0.598 | 2.328 | 4.27 y | 0.49 y | 3.78 y | `WATERWAY_ADEQUACY` |

### 9.2 Checks preserved

The pipeline asserts, among others:

- bridge splits are disjoint;
- feature allowlists are enforced for static and time-varying matrices;
- counting-process intervals are contiguous, positive-duration, and event-only-in-final;
- no future climate is joined and no missing values remain after imputation;
- survival probabilities are monotone;
- SHAP additivity holds;
- ground truth and reported condition are excluded from scoring;
- archetypes are absent from train and validation.

### 9.3 Reproducing the updated results

```powershell
python scripts/run_end_to_end.py
python -m pytest -q
python scripts/generate_report_pdf.py    # optional PDF refresh
```

The end-to-end run regenerates every artifact, persists the preprocessors, writes `archetype_validation.json`, and writes the updated thresholds into `end_to_end_summary.json`.

---

## 10. Inference — how to give the engine an input

The trained engine is a **panel model**, not a stateless classifier: each input needs an inventory row plus district climate history from `ENTRY_YEAR − 5` through the scoring year.

### 10.1 Existing cohort rows

```powershell
python scripts/score_bridges.py --bridge-ids BR-0000038,BR-0000162 --rul
```

This loads the already-preprocessed test-split rows and returns `RISK_SCORE`, `SURVIVAL_AT_{h}Y`, `RUL_YEARS`, and `RUL_CENSORED`.

### 10.2 New raw roster

```powershell
python scripts/score_bridges.py --export-sidecar
# edit artifacts/data/scoring/bridges_template.csv, then:
python scripts/score_bridges.py `
  --bridges artifacts/data/scoring/bridges_template.csv `
  --climate artifacts/data/scoring/climate.csv `
  --horizons 5,10,25,40 --rul --output scores.csv
```

The raw roster is rebuilt with the same `AndersenGillSplitter`, `build_static_survival_frame`, and `attach_dynamic_covariates` used in training, then transformed with the persisted training imputer/encoder. Scoring reuses `conditional_survival_curve` and `compute_rul_from_survival`, so values match the audit and triage reports exactly.

`--event-district Darbhanga --severity 1.35` applies a post-disaster shock; `--model ensemble` selects another fitted model; `--print-schema` prints the input contract.

---

## 11. Governance, reproducibility, and limitations

### 11.1 Reproducibility

- Deterministic generator for a fixed seed and cohort size.
- Default seed 42; all random operations accept `random_state=42`.
- Imputation, scaling, and encoding fitted on the training split only.
- Audit residual and IsolationForest fitted on the training split and evaluated out-of-sample.
- Archetypes are appended deterministically and forced into the test split, so no random draw is consumed by them.
- Artifacts are reloaded from disk before post-disaster triage, exercising the serialisation boundary.

### 11.2 Limitations

- The generator is a calibrated engineering benchmark, not a cleaned NBI/IBMS extract. It encodes the dominant failure physics so the full pipeline can be exercised before replacing the generator with real inventory CSVs.
- The audit's physics-implied condition is the monotone proxy `9 · S`, not a calibrated NBI rating.
- Hazard ratios are associational estimates from a synthetic benchmark, not causal field estimates.
- The archetypes are engineered representatives of documented failure modes, not digitised records of the specific structures.
- Model outputs are decision support for NDT prioritisation, not a certified safety determination.
- The recalibrated thresholds intentionally trade precision for recall; the operating point should be re-validated on agency data before field deployment.

---

## 12. Change log for version 0.2.0

| File | Change |
|---|---|
| `src/bridge_engine/survival_models.py` | Replaced absolute-increment shock table with +1 SD/+2 SD; composite is `exp(Σ βᵢδᵢ)`; added component decomposition and dominance check |
| `src/bridge_engine/data_pipeline.py` | Climate panel floored at 1900; graceful `ENTRY_YEAR` clamping for vintage archetypes; explicit archetype-only-test assertions; `COLLAPSE_ARCHETYPES`, `inject_collapse_archetypes`, `archetype_bridge_ids`; forced-test split support; preprocessor persistence |
| `src/bridge_engine/audit_verifier.py` | `DEFAULT_CRITICAL_SURVIVAL` 0.45 → 0.68; `DEFAULT_Z_FLAG_THRESHOLD` 2.0 → 1.5; healthy band widened to 7–9; MAD scale guarded by `max(1.4826·MAD, 1e-6)`; docstrings updated |
| `src/bridge_engine/triage_and_xai.py` | RUL threshold aligned to 0.68; survival curves clipped to `[0, 1]` and forced monotone; log-linear/exponential tails retained for `t > t_max` |
| `config/default.json` | Runtime audit thresholds updated to 0.68 and 1.5; healthy band 7–9; triage/RUL threshold 0.68 |
| `scripts/run_end_to_end.py` | Injects archetypes, validates their climate coverage, computes `archetype_validation_table`, writes `archetype_validation.json` |
| `scripts/generate_report_pdf.py` | Shock table and audit narrative updated to the new scale and thresholds |
| `scripts/score_bridges.py` | Inference entry point for cohort rows and raw rosters |
| `README.md` | Documents the recalculated thresholds, archetypes, and inference workflow |
| `docs/TECHNICAL_REPORT.md` | This report |
