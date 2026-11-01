"""Module 3: zero-trust "paper inspection" audit and anomaly verification.

The module compares two independent views of a bridge:

1. A physics/climate survival view.  A fitted Module 2 survival model predicts
   the conditional survival probability over the next audit horizon using only
   objective features: structural attributes, traffic loading, and observed
   rainfall / flood / scour history.  Reported condition ratings, NDT process
   flags, latent truth, and audit labels are never used by this view.

2. The human inspection view.  The latest reported NBI minimum condition rating
   is compared with the physics-implied condition and the conditional survival
   probability.

Two complementary anomaly detectors are then fitted:

* A robust residual standardiser turns the condition gap into a one-sided
  Audit Z-score.  The centre and scale are fitted on the calibration split only.
* An :class:`sklearn.ensemble.IsolationForest` is fitted on objective climate /
  structural features plus the physics scores and the reported condition.  The
  forest therefore flags *inconsistent* audit records, not merely unusual
  bridges.

The primary zero-trust finding is deliberately explicit: a bridge is a
``PAPER_INSPECTION_SUSPECT`` when the inspector reports a healthy condition of
7-9 while the physics model estimates a conditional survival probability below
the configured critical threshold (default 0.68).  A broader ``AUDIT_ALERT``
also incorporates the robust Z-residual and IsolationForest evidence.

Ground truth is accepted only for *post-hoc evaluation*.  It is merged after all
scores and flags are computed and is never part of a model matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import (
    FORBIDDEN_FEATURE_NAMES,
    ID_COL,
    PRE_ENTRY_CLIMATE_COLUMNS,
    STATIC_FEATURE_COLUMNS,
    SurvivalDatasets,
    assert_feature_allowlist,
    validate_bridge_frame,
    validate_climate_frame,
)
from .survival_models import (
    SurvivalEngineResult,
    _jsonable,
    load_processed_datasets,
)

LOGGER = logging.getLogger(__name__)

AUDIT_SCHEMA_VERSION: Final[int] = 1
DEFAULT_AUDIT_HORIZON_YEARS: Final[float] = 5.0
DEFAULT_CRITICAL_SURVIVAL: Final[float] = 0.68
DEFAULT_HEALTHY_MIN: Final[int] = 7
DEFAULT_HEALTHY_MAX: Final[int] = 9
DEFAULT_Z_FLAG_THRESHOLD: Final[float] = 1.5
DEFAULT_MIN_CONDITION_GAP: Final[float] = 0.5
DEFAULT_SURVIVAL_FLOOR: Final[float] = 1e-6
RANDOM_STATE: Final[int] = 42

OBJECTIVE_STATIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    STATIC_FEATURE_COLUMNS + PRE_ENTRY_CLIMATE_COLUMNS
)

AUDIT_RAW_STATIC_COLUMNS: Final[tuple[str, ...]] = (
    "ADT_TRAFFIC",
    "TRUCK_ADT_PCT",
    "MAX_SPAN_LEN",
    "WATERWAY_ADEQUACY",
    "FLOOD_PRONENESS",
)

CLIMATE_SNAPSHOT_COLUMNS: Final[tuple[str, ...]] = (
    "MONSOON_ANOMALY_AT_AUDIT",
    "SEASONAL_RAINFALL_MM_AT_AUDIT",
    "PEAK_DISCHARGE_M3S_AT_AUDIT",
    "PEAK_VELOCITY_MPS_AT_AUDIT",
    "PEAK_SCOUR_INDEX_AT_AUDIT",
    "FLOOD_EXCEEDANCE_AT_AUDIT",
    "CYCLONE_FLAG_AT_AUDIT",
    "CUMULATIVE_FLOOD_EXCEEDANCE_AT_AUDIT",
    "CUMULATIVE_SCOUR_LOAD_AT_AUDIT",
)

AUDIT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "AGE_AT_AUDIT",
    *AUDIT_RAW_STATIC_COLUMNS,
    *CLIMATE_SNAPSHOT_COLUMNS,
    "PHYSICS_EXPECTED_CONDITION",
    "EXPECTED_SURVIVAL",
    "EXPECTED_DEGRADATION_HAZARD",
    "REPORTED_LAST_MIN_COND",
    "CONDITION_GAP",
    "SURVIVAL_DEFICIT",
)

# ``REPORTED_*`` fields describe the human claim under audit.  Only the latest
# minimum rating is allowed into the audit feature matrix; every other reported
# or truth-side field is rejected.
AUDIT_ALLOWED_REPORTED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"REPORTED_LAST_MIN_COND"}
)
FORBIDDEN_AUDIT_PREFIXES: Final[tuple[str, ...]] = (
    "TRUE_",
    "LATENT_",
    "PAPER_",
    "INSPECTOR_",
    "_",
)

AuditSplit = Literal["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditVerifierConfig:
    """Configuration for the zero-trust audit verifier.

    Parameters
    ----------
    horizon_years:
        Forward window used for conditional survival.  The physics model is
        evaluated at the last observation time ``t0`` and ``t0 + horizon``.
    critical_survival_threshold:
        Conditional survival below this value is treated as high collapse risk.
        A bridge reported in condition 7-9 implies a high five-year survival
        probability, so the default (0.68) is deliberately close to
        ``(7 / 9) ~= 0.78`` rather than the old, over-conservative 0.45.
    healthy_condition_min, healthy_condition_max:
        Inclusive reported-condition band that constitutes a "healthy" claim.
        The default covers every claim of 7 or better, including condition 9.
    z_flag_threshold:
        One-sided Z-residual threshold used in the broader audit alert.
    min_condition_gap:
        Minimum reported-minus-physics condition gap required for statistical or
        IsolationForest-driven alerts.
    isolation_forest_contamination:
        Expected anomaly fraction for the IsolationForest, or ``"auto"``.
    isolation_forest_n_estimators:
        Number of isolation trees.
    isolation_forest_fit_on:
        ``"all"`` fits on every calibration bridge.  ``"ndt"`` restricts the
        forest to NDT-verified inspections when enough are available.
    model_name:
        Optional Module 2 model override.  The default is the time-varying Cox
        model because it is the leakage-safe survival view with point-in-time
        climate covariates.  Set to ``None`` to use the engine-selected model.
    calibration_split:
        Split used to fit residual and IsolationForest parameters.
    audit_splits:
        Splits included in the findings table.  Training rows are excluded by
        default to keep the audit out-of-sample.
    """

    horizon_years: float = DEFAULT_AUDIT_HORIZON_YEARS
    critical_survival_threshold: float = DEFAULT_CRITICAL_SURVIVAL
    healthy_condition_min: int = DEFAULT_HEALTHY_MIN
    healthy_condition_max: int = DEFAULT_HEALTHY_MAX
    z_flag_threshold: float = DEFAULT_Z_FLAG_THRESHOLD
    min_condition_gap: float = DEFAULT_MIN_CONDITION_GAP
    z_clip: float = 8.0
    survival_floor: float = DEFAULT_SURVIVAL_FLOOR
    isolation_forest_contamination: float | str = "auto"
    isolation_forest_n_estimators: int = 300
    isolation_forest_max_samples: int | float | str = "auto"
    isolation_forest_n_jobs: int = -1
    isolation_forest_fit_on: Literal["all", "ndt"] = "all"
    model_name: str | None = "cox_time_varying"
    calibration_split: AuditSplit = "train"
    audit_splits: tuple[str, ...] = ("validation", "test")
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive")
        if not 0.0 < self.critical_survival_threshold < 1.0:
            raise ValueError("critical_survival_threshold must be in (0, 1)")
        if not 0 <= self.healthy_condition_min <= self.healthy_condition_max <= 9:
            raise ValueError("healthy condition bounds must satisfy 0 <= min <= max <= 9")
        if self.z_flag_threshold < 0:
            raise ValueError("z_flag_threshold must be non-negative")
        if self.min_condition_gap < 0:
            raise ValueError("min_condition_gap must be non-negative")
        if self.z_clip <= 0:
            raise ValueError("z_clip must be positive")
        if not 0.0 < self.survival_floor < 1.0:
            raise ValueError("survival_floor must be in (0, 1)")
        contamination = self.isolation_forest_contamination
        if contamination != "auto":
            if not isinstance(contamination, (int, float)):
                raise ValueError("contamination must be 'auto' or a float")
            if not 0.0 < float(contamination) <= 0.5:
                raise ValueError("contamination must be in (0, 0.5]")
        if self.isolation_forest_n_estimators <= 0:
            raise ValueError("isolation_forest_n_estimators must be positive")
        if self.isolation_forest_n_jobs == 0:
            raise ValueError("isolation_forest_n_jobs must be non-zero")
        if self.isolation_forest_fit_on not in ("all", "ndt"):
            raise ValueError("isolation_forest_fit_on must be 'all' or 'ndt'")
        if self.calibration_split not in ("train", "validation", "test"):
            raise ValueError("calibration_split must be train, validation, or test")
        if not self.audit_splits:
            raise ValueError("audit_splits must not be empty")
        invalid = set(self.audit_splits).difference({"train", "validation", "test"})
        if invalid:
            raise ValueError(f"unknown audit split(s): {sorted(invalid)}")
        if len(set(self.audit_splits)) != len(self.audit_splits):
            raise ValueError("audit_splits contains duplicates")


@dataclass(frozen=True)
class AuditCalibration:
    """Robust centre and scale for the one-sided Audit Z-residual."""

    residual_center: float
    residual_scale: float
    n_rows: int
    calibration_split: str


@dataclass(frozen=True)
class AuditVerifierResult:
    """Audit findings, fitted detectors, and the serialisable report."""

    audit_table: pd.DataFrame
    report: dict[str, Any]
    metrics: dict[str, Any]
    model_name: str
    audit_feature_columns: tuple[str, ...]
    isolation_forest: IsolationForest
    feature_pipeline: Pipeline
    calibration: AuditCalibration

    def save(self, output_dir: Path) -> None:
        """Persist findings, detector state, and the audit report."""

        output_dir.mkdir(parents=True, exist_ok=True)
        _write_frame(self.audit_table, output_dir / "audit_findings.parquet")
        joblib.dump(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "isolation_forest": self.isolation_forest,
                "feature_pipeline": self.feature_pipeline,
                "calibration": asdict(self.calibration),
                "audit_feature_columns": self.audit_feature_columns,
                "model_name": self.model_name,
            },
            output_dir / "audit_models.joblib",
            compress=3,
        )
        with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(self.report), handle, indent=2)


# ---------------------------------------------------------------------------
# Feature guards
# ---------------------------------------------------------------------------


def assert_audit_feature_allowlist(feature_columns: Sequence[str]) -> None:
    """Reject truth-side, latent, and unauthorised inspection fields.

    The latest minimum condition rating is the single reported field that the
    audit is explicitly designed to challenge.  It may enter the audit feature
    matrix, but it must never enter the survival model matrix; that boundary is
    checked separately with :func:`assert_feature_allowlist`.
    """

    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("audit feature columns contain duplicates")
    for column in feature_columns:
        if column in FORBIDDEN_FEATURE_NAMES:
            raise ValueError(f"forbidden target/identifier column in audit features: {column}")
        if column.startswith(FORBIDDEN_AUDIT_PREFIXES):
            raise ValueError(f"truth-side or private column in audit features: {column}")
        if column.startswith("REPORTED_") and column not in AUDIT_ALLOWED_REPORTED_COLUMNS:
            raise ValueError(f"unauthorised inspection field in audit features: {column}")
        if column == "NDT_PERFORMED" or column == "LAST_INSPECTION_WAS_NDT":
            raise ValueError("NDT process flags must not drive the audit detector")


def _validate_engine_feature_isolation(
    engine_result: SurvivalEngineResult, model_name: str
) -> tuple[str, ...]:
    """Verify that the survival view never consumes inspection or truth fields."""

    model = engine_result.models[model_name]
    if model_name == "cox_time_varying":
        feature_names = tuple(getattr(model, "feature_names_in_", ()))
    else:
        feature_names = tuple(engine_result.static_feature_names)
    if not feature_names:
        raise ValueError("the selected survival model has no input feature names")
    assert_feature_allowlist(feature_names)
    return feature_names


# ---------------------------------------------------------------------------
# Conditional survival
# ---------------------------------------------------------------------------


def _cumulative_hazard_at(
    baseline_times: np.ndarray,
    baseline_values: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Interpolate the baseline cumulative hazard and extrapolate its tail.

    Beyond the largest observed baseline time, a log-linear tail is fitted to
    the last decile of positive baseline increments.  This is a conservative
    parametric extension, not an in-sample interpolation.  The resulting
    cumulative hazard remains non-decreasing in time.
    """

    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("times must be a finite, non-negative one-dimensional array")
    if baseline_times.size == 0 or baseline_times.size != baseline_values.size:
        raise ValueError("baseline hazard arrays must be non-empty and aligned")

    interpolation = np.interp(
        times,
        baseline_times,
        baseline_values,
        left=0.0,
        right=float(baseline_values[-1]),
    )
    upper = float(baseline_times[-1])
    beyond = times > upper
    last_value = float(baseline_values[-1])
    if beyond.any() and last_value > 0.0:
        positive = baseline_values > 0.0
        tail_times = baseline_times[positive]
        tail_values = baseline_values[positive]
        if tail_times.size >= 2 and tail_times[-1] > tail_times[0]:
            tail_size = max(5, int(np.ceil(0.10 * tail_times.size)))
            tail_times = tail_times[-tail_size:]
            tail_values = tail_values[-tail_size:]
            slope = float(
                np.polyfit(tail_times, np.log(tail_values), deg=1)[0]
            )
            slope = float(np.clip(slope, 1e-6, 1.0))
            interpolation[beyond] = last_value * np.exp(
                slope * (times[beyond] - upper)
            )
    return np.maximum(interpolation, 0.0)


def _conditional_survival_cox(
    model: Any,
    frame: pd.DataFrame,
    time_at_audit: np.ndarray,
    horizon_years: float,
    survival_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional survival for a fitted Andersen-Gill Cox model."""

    log_hazard = model.predict_log_hazard(frame)
    baseline_times, baseline_values = model.baseline_cumulative_hazard()
    cumulative_at_audit = _cumulative_hazard_at(
        baseline_times, baseline_values, time_at_audit
    )
    cumulative_at_horizon = _cumulative_hazard_at(
        baseline_times, baseline_values, time_at_audit + horizon_years
    )
    increment = np.maximum(cumulative_at_horizon - cumulative_at_audit, 0.0)
    conditional = np.exp(-np.exp(log_hazard) * increment)
    conditional = np.clip(conditional, 0.0, 1.0)
    survival_at_audit = np.clip(
        np.exp(-np.exp(log_hazard) * cumulative_at_audit), survival_floor, 1.0
    )
    survival_at_horizon = np.clip(
        survival_at_audit * conditional, survival_floor, 1.0
    )
    return survival_at_audit, survival_at_horizon


def _step_domain_max(model: Any, X: np.ndarray) -> float | None:
    """Return the largest evaluable time for a scikit-survival step model."""

    raw_model = getattr(model, "model_", None)
    if raw_model is not None and hasattr(raw_model, "predict_survival_function"):
        functions = raw_model.predict_survival_function(X[:1], return_array=False)
        if functions:
            return float(functions[0].domain[1])
    submodels = getattr(model, "models_", None)
    if isinstance(submodels, Mapping) and submodels:
        domains = [
            domain
            for domain in (_step_domain_max(submodel, X) for submodel in submodels.values())
            if domain is not None
        ]
        if domains:
            return float(min(domains))
    return None


def _tail_hazard(
    survival_low: np.ndarray,
    survival_high: np.ndarray,
    elapsed: float,
    survival_floor: float,
) -> np.ndarray:
    """Estimate an annual hazard from the last observed survival step."""

    if elapsed <= 0.0:
        return np.zeros_like(survival_low)
    low = np.clip(survival_low, survival_floor, 1.0)
    high = np.clip(survival_high, survival_floor, 1.0)
    ratio = np.clip(high / low, survival_floor, 1.0)
    hazard = -np.log(ratio) / elapsed
    return np.clip(hazard, 0.0, 5.0)


def _conditional_survival_step_model(
    model: Any,
    X: np.ndarray,
    time_at_audit: np.ndarray,
    horizon_years: float,
    survival_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional survival for a tree/ensemble step-function model.

    scikit-survival survival functions are evaluated on a common time grid per
    call.  Durations in the bridge panel are annual integers, so unique audit
    times are few; grouping by unique ``t0`` keeps the operation vectorised
    without an ``iterrows`` loop.

    Beyond the largest observed step time, a piecewise-exponential tail is used.
    The annual tail hazard is estimated from the final observed survival step.
    This avoids both an out-of-domain exception and an artificial flat tail.
    """

    n_rows = X.shape[0]
    survival_at_audit = np.empty(n_rows, dtype=np.float64)
    survival_at_horizon = np.empty(n_rows, dtype=np.float64)
    domain_max = _step_domain_max(model, X)
    unique_times, inverse = np.unique(
        time_at_audit.astype(np.float64), return_inverse=True
    )
    for group_index, t_start in enumerate(unique_times):
        mask = inverse == group_index
        group_X = X[mask]
        t0 = float(t_start)
        t1 = t0 + horizon_years

        if domain_max is None or domain_max <= 0.0:
            probabilities = model.predict_survival(group_X, [t0, t1])
            group_at_audit = np.clip(probabilities[:, 0], survival_floor, 1.0)
            group_at_horizon = np.clip(probabilities[:, 1], 0.0, 1.0)
        else:
            clipped_t0 = min(t0, domain_max)
            clipped_t1 = min(t1, domain_max)
            tail_needed = t1 > domain_max or t0 > domain_max
            if not tail_needed:
                probabilities = model.predict_survival(
                    group_X, [clipped_t0, clipped_t1]
                )
                group_at_audit = np.clip(probabilities[:, 0], survival_floor, 1.0)
                group_at_horizon = np.clip(probabilities[:, 1], 0.0, 1.0)
            else:
                delta = min(1.0, domain_max / 2.0)
                evaluation_times = np.unique(
                    np.asarray(
                        [clipped_t0, clipped_t1, domain_max - delta, domain_max],
                        dtype=np.float64,
                    )
                )
                probabilities = model.predict_survival(group_X, evaluation_times)
                column = {
                    float(time): index
                    for index, time in enumerate(evaluation_times)
                }
                audit_survival = np.clip(
                    probabilities[:, column[float(clipped_t0)]],
                    survival_floor,
                    1.0,
                )
                horizon_survival = np.clip(
                    probabilities[:, column[float(clipped_t1)]],
                    survival_floor,
                    1.0,
                )
                low_survival = probabilities[
                    :, column[float(domain_max - delta)]
                ]
                high_survival = probabilities[:, column[float(domain_max)]]
                hazard = _tail_hazard(
                    low_survival,
                    high_survival,
                    delta,
                    survival_floor,
                )
                audit_tail = np.exp(-hazard * max(0.0, t0 - domain_max))
                horizon_tail = np.exp(-hazard * max(0.0, t1 - domain_max))
                group_at_audit = np.clip(
                    audit_survival * audit_tail, survival_floor, 1.0
                )
                group_at_horizon = np.clip(
                    horizon_survival * horizon_tail, 0.0, 1.0
                )

        group_at_horizon = np.minimum(group_at_horizon, group_at_audit)
        survival_at_audit[mask] = group_at_audit
        survival_at_horizon[mask] = group_at_horizon
    return survival_at_audit, survival_at_horizon


def predict_conditional_survival(
    *,
    datasets: SurvivalDatasets,
    engine_result: SurvivalEngineResult,
    model_name: str,
    split: AuditSplit,
    horizon_years: float,
    survival_floor: float = DEFAULT_SURVIVAL_FLOOR,
) -> pd.DataFrame:
    """Predict conditional survival over ``[t0, t0 + horizon]`` per bridge.

    ``t0`` is the last observed time (inspection or censoring).  The returned
    conditional probability is ``S(t0 + h | t0, X_t0) / S(t0 | X_t0)``, not the
    unconditional survival from bridge entry.  That distinction is essential for
    auditing old bridges.
    """

    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    if not 0.0 < survival_floor < 1.0:
        raise ValueError("survival_floor must be in (0, 1)")
    if model_name not in engine_result.models:
        raise ValueError(f"unknown survival model: {model_name}")
    model = engine_result.models[model_name]

    static_frame = getattr(datasets, f"static_{split}")
    if static_frame.empty:
        raise ValueError(f"static_{split} is empty")

    if model_name == "cox_time_varying":
        counting_frame = getattr(datasets, f"counting_{split}")
        if counting_frame.empty:
            raise ValueError(f"counting_{split} is empty")
        ordered = counting_frame.sort_values(["id", "start"], kind="mergesort")
        last = ordered.groupby("id", observed=True, sort=False).tail(1).copy()
        time_at_audit = last["stop"].to_numpy(np.float64)
        survival_at_audit, survival_at_horizon = _conditional_survival_cox(
            model,
            last,
            time_at_audit,
            horizon_years,
            survival_floor,
        )
        bridge_ids = last["id"].astype(str).to_numpy()
    else:
        feature_names = tuple(engine_result.static_feature_names)
        missing = [name for name in feature_names if name not in static_frame.columns]
        if missing:
            raise ValueError(
                "static audit frame is missing fitted survival features: "
                f"{missing[:10]}"
            )
        X = static_frame.loc[:, list(feature_names)].to_numpy(np.float64)
        if not np.isfinite(X).all():
            raise ValueError("static audit design matrix contains non-finite values")
        time_at_audit = static_frame["duration"].to_numpy(np.float64)
        survival_at_audit, survival_at_horizon = _conditional_survival_step_model(
            model,
            X,
            time_at_audit,
            horizon_years,
            survival_floor,
        )
        bridge_ids = static_frame["bridge_id"].astype(str).to_numpy()

    conditional = np.clip(survival_at_horizon / survival_at_audit, 0.0, 1.0)
    hazard = -np.log(np.maximum(conditional, survival_floor)) / horizon_years
    output = pd.DataFrame(
        {
            ID_COL: bridge_ids,
            "DATA_SPLIT": split,
            "TIME_AT_AUDIT": time_at_audit.astype(np.float32),
            "SURVIVAL_AT_AUDIT": survival_at_audit.astype(np.float32),
            "SURVIVAL_AT_HORIZON": survival_at_horizon.astype(np.float32),
            "EXPECTED_SURVIVAL": conditional.astype(np.float32),
            "EXPECTED_DEGRADATION_HAZARD": hazard.astype(np.float32),
            "AUDIT_HORIZON_YEARS": np.float32(horizon_years),
        }
    )
    if output[ID_COL].duplicated().any():
        raise ValueError("conditional survival produced duplicate bridge IDs")
    if not np.isfinite(
        output[
            [
                "SURVIVAL_AT_AUDIT",
                "SURVIVAL_AT_HORIZON",
                "EXPECTED_SURVIVAL",
                "EXPECTED_DEGRADATION_HAZARD",
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("conditional survival produced non-finite values")
    return output


# ---------------------------------------------------------------------------
# Objective metadata and audit feature construction
# ---------------------------------------------------------------------------


def _climate_snapshot(climate: pd.DataFrame) -> pd.DataFrame:
    """Return district-year climate values and cumulative exposure to date."""

    validate_climate_frame(climate)
    ordered = climate.sort_values(["DISTRICT", "YEAR"], kind="mergesort").copy()
    ordered["_DISTRICT_KEY"] = ordered["DISTRICT"].astype(str)
    grouped = ordered.groupby("_DISTRICT_KEY", observed=True, sort=False)
    ordered["CUMULATIVE_FLOOD_EXCEEDANCE"] = grouped["FLOOD_EXCEEDANCE"].cumsum()
    ordered["CUMULATIVE_SCOUR_LOAD"] = grouped["PEAK_SCOUR_INDEX"].cumsum()
    snapshot = ordered[
        [
            "_DISTRICT_KEY",
            "YEAR",
            "MONSOON_ANOMALY",
            "SEASONAL_RAINFALL_MM",
            "PEAK_DISCHARGE_M3S",
            "PEAK_VELOCITY_MPS",
            "PEAK_SCOUR_INDEX",
            "FLOOD_EXCEEDANCE",
            "CYCLONE_FLAG",
            "CUMULATIVE_FLOOD_EXCEEDANCE",
            "CUMULATIVE_SCOUR_LOAD",
        ]
    ].rename(
        columns={
            "YEAR": "LAST_INSPECTION_YEAR",
            "MONSOON_ANOMALY": "MONSOON_ANOMALY_AT_AUDIT",
            "SEASONAL_RAINFALL_MM": "SEASONAL_RAINFALL_MM_AT_AUDIT",
            "PEAK_DISCHARGE_M3S": "PEAK_DISCHARGE_M3S_AT_AUDIT",
            "PEAK_VELOCITY_MPS": "PEAK_VELOCITY_MPS_AT_AUDIT",
            "PEAK_SCOUR_INDEX": "PEAK_SCOUR_INDEX_AT_AUDIT",
            "FLOOD_EXCEEDANCE": "FLOOD_EXCEEDANCE_AT_AUDIT",
            "CYCLONE_FLAG": "CYCLONE_FLAG_AT_AUDIT",
            "CUMULATIVE_FLOOD_EXCEEDANCE": (
                "CUMULATIVE_FLOOD_EXCEEDANCE_AT_AUDIT"
            ),
            "CUMULATIVE_SCOUR_LOAD": "CUMULATIVE_SCOUR_LOAD_AT_AUDIT",
        }
    )
    if snapshot.duplicated(["_DISTRICT_KEY", "LAST_INSPECTION_YEAR"]).any():
        raise ValueError("climate snapshot contains duplicate district-year rows")
    return snapshot


def build_audit_metadata(
    bridges: pd.DataFrame,
    climate: pd.DataFrame,
) -> pd.DataFrame:
    """Build objective bridge metadata at the latest inspection date."""

    validate_bridge_frame(bridges)
    required = [
        ID_COL,
        "DISTRICT",
        "LAST_INSPECTION_AGE",
        "LAST_INSPECTION_YEAR",
        "REPORTED_LAST_MIN_COND",
        *AUDIT_RAW_STATIC_COLUMNS,
    ]
    missing = [column for column in required if column not in bridges.columns]
    if missing:
        raise ValueError(f"bridge frame is missing audit metadata columns: {missing}")
    if bridges[ID_COL].duplicated().any():
        raise ValueError("bridge frame contains duplicate bridge IDs")
    if bridges[["LAST_INSPECTION_AGE", "LAST_INSPECTION_YEAR"]].isna().any().any():
        raise ValueError("latest inspection age/year contains missing values")
    if not bridges["REPORTED_LAST_MIN_COND"].between(0, 9).all():
        raise ValueError("reported minimum condition is outside the NBI 0-9 scale")

    work = bridges[required].copy()
    if "LAST_INSPECTION_WAS_NDT" in bridges.columns:
        work["NDT_PERFORMED"] = bridges["LAST_INSPECTION_WAS_NDT"].to_numpy(np.int8)
    work["_DISTRICT_KEY"] = work["DISTRICT"].astype(str)
    work["AGE_AT_AUDIT"] = work["LAST_INSPECTION_AGE"].astype(np.float32)
    snapshot = _climate_snapshot(climate)
    merged = work.merge(
        snapshot,
        on=["_DISTRICT_KEY", "LAST_INSPECTION_YEAR"],
        how="left",
        validate="many_to_one",
    )
    if merged[list(CLIMATE_SNAPSHOT_COLUMNS)].isna().any().any():
        raise ValueError("climate panel does not cover every bridge's audit year")
    merged = merged.drop(columns=["_DISTRICT_KEY", "LAST_INSPECTION_AGE"])
    merged[ID_COL] = merged[ID_COL].astype(str)
    return merged.reset_index(drop=True)


def _build_audit_table(
    *,
    physics: pd.DataFrame,
    metadata: pd.DataFrame,
    config: AuditVerifierConfig,
) -> pd.DataFrame:
    """Join physics predictions to objective metadata and the reported claim.

    The physics-implied condition is the monotone proxy ``9 * S(t0 + h | t0)``.
    It is not a calibrated NBI rating; it places survival on the same 0-9 scale
    so that the reported-minus-physics gap is interpretable to inspectors.
    """

    table = physics.merge(metadata, on=ID_COL, how="left", validate="one_to_one")
    if table.empty:
        raise ValueError("audit table is empty")
    critical_metadata = [
        "AGE_AT_AUDIT",
        "REPORTED_LAST_MIN_COND",
        *CLIMATE_SNAPSHOT_COLUMNS,
    ]
    if table[critical_metadata].isna().any().any():
        raise ValueError("audit metadata join left missing critical fields")

    expected_condition = np.clip(
        9.0 * table["EXPECTED_SURVIVAL"].to_numpy(np.float64), 0.0, 9.0
    )
    table["PHYSICS_EXPECTED_CONDITION"] = expected_condition.astype(np.float32)
    condition_gap = (
        table["REPORTED_LAST_MIN_COND"].to_numpy(np.float64) - expected_condition
    )
    table["CONDITION_GAP"] = condition_gap.astype(np.float32)
    table["SURVIVAL_DEFICIT"] = np.maximum(
        0.0,
        config.critical_survival_threshold
        - table["EXPECTED_SURVIVAL"].to_numpy(np.float64),
    ).astype(np.float32)
    reported = table["REPORTED_LAST_MIN_COND"].to_numpy(np.int16)
    table["REPORTED_HEALTHY"] = (
        (reported >= config.healthy_condition_min)
        & (reported <= config.healthy_condition_max)
    )
    table["PAPER_INSPECTION_SUSPECT"] = (
        table["REPORTED_HEALTHY"].to_numpy(bool)
        & (
            table["EXPECTED_SURVIVAL"].to_numpy(np.float64)
            < config.critical_survival_threshold
        )
    )
    missing_features = [
        column for column in AUDIT_FEATURE_COLUMNS if column not in table.columns
    ]
    if missing_features:
        raise RuntimeError(f"audit feature construction failed: {missing_features}")
    return table


# ---------------------------------------------------------------------------
# Detector fitting
# ---------------------------------------------------------------------------


def _fit_robust_residual(values: np.ndarray, split: str) -> AuditCalibration:
    """Fit a median/MAD standardiser for the one-sided condition residual."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("at least two finite residuals are required for calibration")
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    # ``max(1.4826 * MAD, 1e-6)`` is the explicit zero-dispersion guard.  A
    # homogeneous calibration subset would otherwise divide by zero and emit
    # infinite or undefined Audit Z-scores.
    scale = max(1.4826 * mad, 1e-6)
    return AuditCalibration(
        residual_center=center,
        residual_scale=scale,
        n_rows=int(finite.size),
        calibration_split=split,
    )


def _fit_isolation_forest(
    calibration: pd.DataFrame,
    *,
    config: AuditVerifierConfig,
) -> tuple[Pipeline, IsolationForest, tuple[str, ...]]:
    """Fit the audit anomaly detector on the calibration split only."""

    feature_columns = AUDIT_FEATURE_COLUMNS
    assert_audit_feature_allowlist(feature_columns)
    eligible = calibration
    if config.isolation_forest_fit_on == "ndt":
        if "NDT_PERFORMED" in calibration.columns:
            verified = calibration.loc[calibration["NDT_PERFORMED"] == 1]
            if len(verified) >= 50:
                eligible = verified
            else:
                LOGGER.warning(
                    "Only %s NDT-verified calibration rows; using all rows for "
                    "IsolationForest fitting.",
                    len(verified),
                )
        else:
            LOGGER.warning(
                "NDT_PERFORMED is unavailable; using all calibration rows for "
                "IsolationForest fitting."
            )
    X = eligible.loc[:, list(feature_columns)].to_numpy(np.float64)
    if X.shape[0] < 2:
        raise ValueError("at least two calibration rows are required")

    feature_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    transformed = feature_pipeline.fit_transform(X)
    forest = IsolationForest(
        n_estimators=config.isolation_forest_n_estimators,
        contamination=config.isolation_forest_contamination,
        max_samples=config.isolation_forest_max_samples,
        random_state=config.random_state,
        n_jobs=config.isolation_forest_n_jobs,
    )
    forest.fit(transformed)
    return feature_pipeline, forest, feature_columns


def _apply_audit_scores(
    table: pd.DataFrame,
    *,
    calibration: AuditCalibration,
    feature_pipeline: Pipeline,
    isolation_forest: IsolationForest,
    feature_columns: Sequence[str],
    config: AuditVerifierConfig,
) -> pd.DataFrame:
    """Apply residual standardisation and IsolationForest scores."""

    output = table.copy()
    residual = output["CONDITION_GAP"].to_numpy(np.float64)
    z_score = (residual - calibration.residual_center) / calibration.residual_scale
    z_score = np.clip(z_score, -config.z_clip, config.z_clip)
    output["AUDIT_Z_SCORE"] = z_score.astype(np.float32)

    X = output.loc[:, list(feature_columns)].to_numpy(np.float64)
    transformed = feature_pipeline.transform(X)
    output["ISOLATION_ANOMALY_SCORE"] = (
        -isolation_forest.decision_function(transformed)
    ).astype(np.float32)
    output["ISOLATION_FLAG"] = isolation_forest.predict(transformed) == -1

    reported = output["REPORTED_LAST_MIN_COND"].to_numpy(np.int16)
    survival = output["EXPECTED_SURVIVAL"].to_numpy(np.float64)
    gap = output["CONDITION_GAP"].to_numpy(np.float64)
    z_values = output["AUDIT_Z_SCORE"].to_numpy(np.float64)
    isolation = output["ISOLATION_FLAG"].to_numpy(bool)
    healthy = output["REPORTED_HEALTHY"].to_numpy(bool)
    safe_claim = reported >= config.healthy_condition_min

    primary = healthy & (survival < config.critical_survival_threshold)
    statistical = (
        safe_claim
        & (gap >= config.min_condition_gap)
        & (z_values >= config.z_flag_threshold)
    )
    unsupervised = safe_claim & isolation & (gap >= config.min_condition_gap)
    output["PAPER_INSPECTION_SUSPECT"] = primary
    output["AUDIT_ALERT"] = primary | statistical | unsupervised
    return output


def _prioritise(audit_table: pd.DataFrame, config: AuditVerifierConfig) -> pd.DataFrame:
    """Rank findings and assign a triage priority."""

    output = audit_table.copy()
    if output.empty:
        return output

    def percentile(values: np.ndarray) -> np.ndarray:
        return rankdata(values, method="average") / max(len(values), 1)

    z_positive = np.clip(output["AUDIT_Z_SCORE"].to_numpy(np.float64), 0.0, None)
    survival_risk = 1.0 - output["EXPECTED_SURVIVAL"].to_numpy(np.float64)
    anomaly = output["ISOLATION_ANOMALY_SCORE"].to_numpy(np.float64)
    priority_score = (
        0.45 * percentile(z_positive)
        + 0.35 * percentile(survival_risk)
        + 0.20 * percentile(anomaly)
    )
    output["AUDIT_PRIORITY_SCORE"] = priority_score.astype(np.float32)
    output["AUDIT_RANK"] = rankdata(-priority_score, method="ordinal").astype(np.int32)

    critical = output["AUDIT_ALERT"].to_numpy(bool) & (
        output["EXPECTED_SURVIVAL"].to_numpy(np.float64)
        < config.critical_survival_threshold
    )
    alert = output["AUDIT_ALERT"].to_numpy(bool)
    medium = (
        (output["AUDIT_Z_SCORE"].to_numpy(np.float64) >= config.z_flag_threshold)
        | output["ISOLATION_FLAG"].to_numpy(bool)
    )
    output["AUDIT_PRIORITY"] = np.select(
        [critical, alert, medium],
        ["CRITICAL", "HIGH", "MEDIUM"],
        default="LOW",
    )
    return output.sort_values(
        ["AUDIT_PRIORITY_SCORE", ID_COL], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ground-truth evaluation (evaluation only)
# ---------------------------------------------------------------------------


def evaluate_audit_against_ground_truth(
    audit_table: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate flags against the synthetic paper-inspection label.

    This function is evaluation-only.  It must never be called before scoring,
    and its output must not feed a model matrix.
    """

    required_audit = {ID_COL, "PAPER_INSPECTION_SUSPECT", "AUDIT_ALERT"}
    if not required_audit.issubset(audit_table.columns):
        missing = sorted(required_audit.difference(audit_table.columns))
        raise ValueError(f"audit table is missing evaluation columns: {missing}")
    required_truth = {ID_COL, "PAPER_INSPECTION_FLAG"}
    if not required_truth.issubset(ground_truth.columns):
        missing = sorted(required_truth.difference(ground_truth.columns))
        raise ValueError(f"ground truth is missing columns: {missing}")

    merged = audit_table.merge(
        ground_truth[[ID_COL, "PAPER_INSPECTION_FLAG"]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    if merged["PAPER_INSPECTION_FLAG"].isna().any():
        raise ValueError("ground truth does not cover every audited bridge")
    y_true = merged["PAPER_INSPECTION_FLAG"].to_numpy(np.int8)

    metrics: dict[str, Any] = {
        "n_evaluated": int(len(merged)),
        "n_paper_inspections": int(y_true.sum()),
        "paper_inspection_rate": float(y_true.mean()) if len(y_true) else None,
    }
    for column in (
        "PAPER_INSPECTION_SUSPECT",
        "AUDIT_ALERT",
        "ISOLATION_FLAG",
    ):
        prediction = merged[column].to_numpy(bool)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            metrics[column] = {"precision": None, "recall": None, "f1": None}
            continue
        metrics[column] = {
            "precision": float(precision_score(y_true, prediction, zero_division=0)),
            "recall": float(recall_score(y_true, prediction, zero_division=0)),
            "f1": float(f1_score(y_true, prediction, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
        }

    score = merged["AUDIT_Z_SCORE"].to_numpy(np.float64)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        metrics["audit_z_score"] = {"roc_auc": None, "average_precision": None}
    else:
        metrics["audit_z_score"] = {
            "roc_auc": float(roc_auc_score(y_true, score)),
            "average_precision": float(average_precision_score(y_true, score)),
        }
    metrics["note"] = (
        "Ground truth is used only for post-hoc evaluation. It never enters the "
        "survival features, residual calibration, or IsolationForest."
    )
    return metrics


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class AuditVerifier:
    """Fit and run the zero-trust paper-inspection audit."""

    def __init__(self, config: AuditVerifierConfig | None = None) -> None:
        self.config = config or AuditVerifierConfig()

    def fit(
        self,
        *,
        datasets: SurvivalDatasets,
        bridges: pd.DataFrame,
        climate: pd.DataFrame,
        engine_result: SurvivalEngineResult,
        ground_truth: pd.DataFrame | None = None,
    ) -> AuditVerifierResult:
        """Compute physics survival, audit residuals, and anomaly findings."""

        config = self.config
        validate_bridge_frame(bridges)
        validate_climate_frame(climate)
        model_name = config.model_name or engine_result.selected_model
        if model_name not in engine_result.models:
            raise ValueError(f"unknown survival model: {model_name}")
        survival_features = _validate_engine_feature_isolation(
            engine_result, model_name
        )

        metadata = build_audit_metadata(bridges, climate)
        split_tables: dict[str, pd.DataFrame] = {}
        for split in ("train", "validation", "test"):
            static_frame = getattr(datasets, f"static_{split}")
            if static_frame.empty:
                continue
            physics = predict_conditional_survival(
                datasets=datasets,
                engine_result=engine_result,
                model_name=model_name,
                split=split,
                horizon_years=config.horizon_years,
                survival_floor=config.survival_floor,
            )
            split_tables[split] = _build_audit_table(
                physics=physics,
                metadata=metadata,
                config=config,
            )

        calibration_split = config.calibration_split
        if calibration_split not in split_tables:
            available = [name for name in ("train", "validation", "test") if name in split_tables]
            if not available:
                raise ValueError("no split contains audit rows")
            calibration_split = available[0]
            LOGGER.warning(
                "Requested calibration split is empty; using %s instead.",
                calibration_split,
            )
        calibration_table = split_tables[calibration_split]
        calibration = _fit_robust_residual(
            calibration_table["CONDITION_GAP"].to_numpy(np.float64),
            calibration_split,
        )

        if "NDT_PERFORMED" not in calibration_table.columns:
            if config.isolation_forest_fit_on == "ndt":
                LOGGER.warning("NDT_PERFORMED is unavailable in the audit table.")
        feature_pipeline, isolation_forest, audit_features = _fit_isolation_forest(
            calibration_table,
            config=config,
        )

        for split, table in split_tables.items():
            split_tables[split] = _apply_audit_scores(
                table,
                calibration=calibration,
                feature_pipeline=feature_pipeline,
                isolation_forest=isolation_forest,
                feature_columns=audit_features,
                config=config,
            )

        audit_parts = [
            split_tables[split]
            for split in config.audit_splits
            if split in split_tables
        ]
        if not audit_parts:
            raise ValueError("none of the requested audit splits contain rows")
        audit_table = pd.concat(audit_parts, ignore_index=True)
        audit_table = _prioritise(audit_table, config)

        metrics: dict[str, Any] = {}
        if ground_truth is not None:
            metrics = evaluate_audit_against_ground_truth(audit_table, ground_truth)

        report = self._build_report(
            bridges=bridges,
            engine_result=engine_result,
            model_name=model_name,
            survival_features=survival_features,
            calibration=calibration,
            split_tables=split_tables,
            audit_table=audit_table,
            audit_features=audit_features,
            metrics=metrics,
        )
        return AuditVerifierResult(
            audit_table=audit_table,
            report=report,
            metrics=metrics,
            model_name=model_name,
            audit_feature_columns=audit_features,
            isolation_forest=isolation_forest,
            feature_pipeline=feature_pipeline,
            calibration=calibration,
        )

    def _build_report(
        self,
        *,
        bridges: pd.DataFrame,
        engine_result: SurvivalEngineResult,
        model_name: str,
        survival_features: Sequence[str],
        calibration: AuditCalibration,
        split_tables: Mapping[str, pd.DataFrame],
        audit_table: pd.DataFrame,
        audit_features: Sequence[str],
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        config = self.config
        splits = {
            name: {
                "n_rows": int(len(table)),
                "n_reported_healthy": int(table["REPORTED_HEALTHY"].sum()),
                "n_paper_inspection_suspect": int(
                    table["PAPER_INSPECTION_SUSPECT"].sum()
                ),
                "n_audit_alert": int(table["AUDIT_ALERT"].sum()),
                "n_isolation_anomalies": int(table["ISOLATION_FLAG"].sum()),
            }
            for name, table in split_tables.items()
        }
        priority_counts = {
            str(key): int(value)
            for key, value in audit_table["AUDIT_PRIORITY"].value_counts().items()
        }
        top_records = audit_table.head(20).to_dict("records")
        top_findings = [
            {
                "bridge_id": str(record[ID_COL]),
                "district": str(record.get("DISTRICT", "")),
                "age_at_audit": float(record["AGE_AT_AUDIT"]),
                "reported_condition": int(record["REPORTED_LAST_MIN_COND"]),
                "physics_expected_condition": float(
                    record["PHYSICS_EXPECTED_CONDITION"]
                ),
                "expected_survival": float(record["EXPECTED_SURVIVAL"]),
                "condition_gap": float(record["CONDITION_GAP"]),
                "audit_z_score": float(record["AUDIT_Z_SCORE"]),
                "isolation_flag": bool(record["ISOLATION_FLAG"]),
                "paper_inspection_suspect": bool(
                    record["PAPER_INSPECTION_SUSPECT"]
                ),
                "audit_alert": bool(record["AUDIT_ALERT"]),
                "priority": str(record["AUDIT_PRIORITY"]),
            }
            for record in top_records
        ]

        for column in audit_features:
            assert_audit_feature_allowlist([column])
        assert_feature_allowlist(survival_features)
        for column in audit_table.columns:
            if column in {"PAPER_INSPECTION_FLAG", "INSPECTOR_BIAS"}:
                raise AssertionError(f"truth-side column leaked into audit table: {column}")
            if column.startswith(("TRUE_", "LATENT_")):
                raise AssertionError(f"truth-side column leaked into audit table: {column}")

        survival_monotone = bool(
            (
                audit_table["SURVIVAL_AT_HORIZON"].to_numpy(np.float64)
                <= audit_table["SURVIVAL_AT_AUDIT"].to_numpy(np.float64) + 1e-6
            ).all()
        )
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "module": "audit_verifier",
            "config": _jsonable(asdict(config)),
            "physics_model": {
                "model_name": model_name,
                "engine_selected_model": engine_result.selected_model,
                "n_survival_features": len(survival_features),
                "survival_features": list(survival_features),
                "objective_input_columns": list(OBJECTIVE_STATIC_FEATURE_COLUMNS),
                "audit_horizon_years": config.horizon_years,
                "critical_survival_threshold": config.critical_survival_threshold,
            },
            "cohort": {
                "n_bridges_input": int(len(bridges)),
                "n_audited": int(len(audit_table)),
                "audit_splits": list(config.audit_splits),
                "splits": splits,
            },
            "detector": {
                "audit_feature_columns": list(audit_features),
                "isolation_forest": {
                    "n_estimators": config.isolation_forest_n_estimators,
                    "contamination": config.isolation_forest_contamination,
                    "fit_on": config.isolation_forest_fit_on,
                },
                "residual_calibration": asdict(calibration),
            },
            "findings": {
                "n_reported_healthy": int(audit_table["REPORTED_HEALTHY"].sum()),
                "n_paper_inspection_suspect": int(
                    audit_table["PAPER_INSPECTION_SUSPECT"].sum()
                ),
                "n_audit_alert": int(audit_table["AUDIT_ALERT"].sum()),
                "n_isolation_anomalies": int(audit_table["ISOLATION_FLAG"].sum()),
                "suspect_rate": float(audit_table["PAPER_INSPECTION_SUSPECT"].mean()),
                "alert_rate": float(audit_table["AUDIT_ALERT"].mean()),
                "priority_counts": priority_counts,
                "top_findings": top_findings,
            },
            "ground_truth_evaluation": _jsonable(metrics) if metrics else None,
            "checks": {
                "survival_features_objective_only": True,
                "reported_condition_excluded_from_survival": True,
                "ground_truth_excluded_from_scoring": True,
                "audit_feature_allowlist_enforced": True,
                "conditional_survival_monotone": survival_monotone,
                "audit_z_score_calibrated_on_train": (
                    calibration.calibration_split == config.calibration_split
                ),
                "no_truth_columns_in_audit_table": True,
                "isolation_forest_fitted": True,
                "holdout_audit": "train" not in config.audit_splits,
                "survival_threshold_applied": True,
            },
        }


# ---------------------------------------------------------------------------
# Persistence / CLI
# ---------------------------------------------------------------------------


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except ImportError:
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return csv_path


def _read_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"neither {path} nor {csv_path} exists")


def load_engine_result(engine_dir: Path) -> SurvivalEngineResult:
    """Load a Module 2 engine bundle from disk."""

    bundle_path = engine_dir / "survival_engine.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"survival engine bundle not found: {bundle_path}. Run Module 2 first."
        )
    bundle = joblib.load(bundle_path)
    report_path = engine_dir / "survival_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    return SurvivalEngineResult(
        models=bundle["models"],
        metrics=bundle.get("metrics", {}),
        report=report,
        selected_model=bundle["selected_model"],
        static_feature_names=tuple(bundle["static_feature_names"]),
        time_varying_feature_names=tuple(bundle["time_varying_feature_names"]),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the zero-trust paper-inspection audit against a fitted Module 2 "
            "survival engine."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--engine-dir", type=Path, default=Path("artifacts/module2"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/module3"))
    parser.add_argument("--horizon", type=float, default=DEFAULT_AUDIT_HORIZON_YEARS)
    parser.add_argument(
        "--critical-survival",
        type=float,
        default=DEFAULT_CRITICAL_SURVIVAL,
    )
    parser.add_argument("--healthy-min", type=int, default=DEFAULT_HEALTHY_MIN)
    parser.add_argument("--healthy-max", type=int, default=DEFAULT_HEALTHY_MAX)
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z_FLAG_THRESHOLD)
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_CONDITION_GAP)
    parser.add_argument("--contamination", type=str, default="auto")
    parser.add_argument(
        "--model-name",
        type=str,
        default="cox_time_varying",
        help="Survival model used for the physics view (default: cox_time_varying).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    datasets = load_processed_datasets(args.data_dir)
    engine_result = load_engine_result(args.engine_dir)
    bridges = _read_frame(args.data_dir / "bridges.parquet")
    climate = _read_frame(args.data_dir / "climate.parquet")
    ground_truth_path = args.data_dir / "EVAL_ONLY_ground_truth.parquet"
    ground_truth_available = ground_truth_path.exists() or ground_truth_path.with_suffix(
        ".csv"
    ).exists()
    ground_truth = _read_frame(ground_truth_path) if ground_truth_available else None
    contamination: float | str
    if args.contamination == "auto":
        contamination = "auto"
    else:
        contamination = float(args.contamination)
    verifier = AuditVerifier(
        AuditVerifierConfig(
            horizon_years=args.horizon,
            critical_survival_threshold=args.critical_survival,
            healthy_condition_min=args.healthy_min,
            healthy_condition_max=args.healthy_max,
            z_flag_threshold=args.z_threshold,
            min_condition_gap=args.min_gap,
            isolation_forest_contamination=contamination,
            model_name=args.model_name,
            random_state=args.seed,
        )
    )
    result = verifier.fit(
        datasets=datasets,
        bridges=bridges,
        climate=climate,
        engine_result=engine_result,
        ground_truth=ground_truth,
    )
    result.save(args.output_dir)
    LOGGER.info(
        "Audited %s bridges | %s paper-inspection suspects | %s audit alerts",
        result.report["cohort"]["n_audited"],
        result.report["findings"]["n_paper_inspection_suspect"],
        result.report["findings"]["n_audit_alert"],
    )
    LOGGER.info("Audit artifacts wrote to %s", args.output_dir)


if __name__ == "__main__":
    main()
