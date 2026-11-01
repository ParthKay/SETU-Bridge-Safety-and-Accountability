"""Module 4: post-disaster dynamic triage and survival-aware explainability.

The module closes the zero-trust pipeline:

1. :class:`PostDisasterUpdater`
   Applies a district- or bridge-scoped flood / cyclone / scour shock to the
   latest observable climate state.  For the Andersen-Gill Cox view the shock is
   written into the last counting-process interval.  For static tree models the
   shock is reflected in the pre-entry climate aggregates and any landmark
   exposure aggregates available in the fitted design.

2. :class:`TriageEngine`
   Recomputes conditional survival curves before and after the event, derives
   Remaining Useful Life (RUL) from the survival curve, and ranks bridges for
   immediate physical NDT deployment.  Optional Module 3 audit findings are
   joined only as metadata and never enter the model or SHAP matrices.

3. :class:`ShapRiskExplainer`
   Uses a model-agnostic :class:`shap.Explainer` over the selected survival
   model's risk predictions.  Local force plots explain why a bridge is
   prioritised, using only objective structural, traffic, and climate features.

Only classical survival models and SHAP are used.  No deep-learning framework
is imported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import joblib
import numpy as np
import pandas as pd
import shap

from .audit_verifier import (
    AuditCalibration,
    AuditVerifierResult,
    _conditional_survival_step_model,
    _cumulative_hazard_at,
)
from .data_pipeline import (
    ID_COL,
    BridgeDataPipeline,
    PipelineConfig,
    SurvivalDatasets,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
    assert_feature_allowlist,
    save_prepared_dataset,
)
from .survival_models import (
    SurvivalEngine,
    SurvivalEngineConfig,
    SurvivalEngineResult,
    _jsonable,
    load_processed_datasets,
)

LOGGER = logging.getLogger(__name__)

RANDOM_STATE: Final[int] = 42
TRIAGE_SCHEMA_VERSION: Final[int] = 1
DEFAULT_DECISION_HORIZON: Final[float] = 5.0
DEFAULT_SURVIVAL_THRESHOLD: Final[float] = 0.68
DEFAULT_RUL_HORIZON: Final[float] = 60.0
DEFAULT_RUL_STEP: Final[float] = 1.0
FORBIDDEN_XAI_PREFIXES: Final[tuple[str, ...]] = (
    "TRUE_",
    "LATENT_",
    "PAPER_",
    "INSPECTOR_",
    "AUDIT_",
    "ISOLATION_",
)

EventType = Literal["FLOOD", "CYCLONE", "SCOUR", "COMPOUND"]
TriageSplit = Literal["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Configuration and public containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisasterEvent:
    """A post-disaster climate shock scoped to a district, state, or bridge set.

    The multipliers scale the latest observed climate covariates.  ``severity``
    scales every deviation from the baseline, so a severity of ``1.0`` applies
    the nominal multipliers and a severity of ``1.5`` amplifies the event.
    """

    event_id: str
    event_type: EventType
    district: str | None = None
    state: str | None = None
    bridge_ids: tuple[str, ...] = ()
    severity: float = 1.0
    monsoon_anomaly_increment: float = 2.5
    rainfall_multiplier: float = 1.8
    discharge_multiplier: float = 2.0
    velocity_multiplier: float = 1.5
    scour_multiplier: float = 2.2
    flood_exceedance_increment: int = 1
    cumulative_scour_increment: float = 12.0
    event_year: int | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if self.event_type not in ("FLOOD", "CYCLONE", "SCOUR", "COMPOUND"):
            raise ValueError(f"unsupported event_type: {self.event_type!r}")
        if self.district is None and self.state is None and not self.bridge_ids:
            raise ValueError("event must target a district, a state, or explicit bridge IDs")
        normalized_ids = tuple(str(value) for value in self.bridge_ids)
        object.__setattr__(self, "bridge_ids", normalized_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("bridge_ids contains duplicates")
        if not np.isfinite(self.severity) or self.severity <= 0.0:
            raise ValueError("severity must be a positive finite value")
        if self.rainfall_multiplier < 1.0 or self.discharge_multiplier < 1.0:
            raise ValueError("rainfall and discharge multipliers must be >= 1")
        if self.velocity_multiplier < 1.0 or self.scour_multiplier < 1.0:
            raise ValueError("velocity and scour multipliers must be >= 1")
        if self.flood_exceedance_increment < 0:
            raise ValueError("flood_exceedance_increment must be non-negative")
        if self.cumulative_scour_increment < 0.0:
            raise ValueError("cumulative_scour_increment must be non-negative")
        if self.event_year is not None and not 1900 <= int(self.event_year) <= 2200:
            raise ValueError("event_year is outside a plausible range")

    @property
    def target_description(self) -> str:
        """Human-readable targeting description for reports."""

        if self.bridge_ids:
            return f"{len(self.bridge_ids)} explicit bridges"
        parts = [value for value in (self.district, self.state) if value]
        return " / ".join(parts)


@dataclass(frozen=True)
class TriageConfig:
    """Configuration for post-disaster scoring and SHAP explanations."""

    model_name: str = "cox_time_varying"
    explain_model_name: str = "ensemble"
    split: TriageSplit = "test"
    horizons: tuple[float, ...] = (5.0, 10.0, 15.0, 25.0, 40.0)
    decision_horizon_years: float = DEFAULT_DECISION_HORIZON
    survival_threshold: float = DEFAULT_SURVIVAL_THRESHOLD
    rul_threshold: float = DEFAULT_SURVIVAL_THRESHOLD
    rul_horizon_years: float = DEFAULT_RUL_HORIZON
    rul_step_years: float = DEFAULT_RUL_STEP
    crisis_rul_years: float = 5.0
    survival_floor: float = 1e-6
    top_n_explanations: int = 8
    shap_background_size: int = 100
    shap_max_evals: int = 512
    random_state: int = RANDOM_STATE
    save_force_plots: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("horizons must be a non-empty sequence of positive values")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be strictly increasing without duplicates")
        if self.decision_horizon_years not in horizons:
            raise ValueError("decision_horizon_years must be one of horizons")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if not 0.0 < self.survival_threshold < 1.0:
            raise ValueError("survival_threshold must be in (0, 1)")
        if not 0.0 < self.rul_threshold < 1.0:
            raise ValueError("rul_threshold must be in (0, 1)")
        if self.rul_horizon_years <= 0.0:
            raise ValueError("rul_horizon_years must be positive")
        if self.rul_step_years <= 0.0:
            raise ValueError("rul_step_years must be positive")
        if self.rul_horizon_years < self.decision_horizon_years:
            raise ValueError("rul_horizon_years must cover the decision horizon")
        if self.crisis_rul_years < 0.0:
            raise ValueError("crisis_rul_years must be non-negative")
        if not 0.0 < self.survival_floor < 1.0:
            raise ValueError("survival_floor must be in (0, 1)")
        if self.top_n_explanations < 0:
            raise ValueError("top_n_explanations must be non-negative")
        if self.shap_background_size < 2:
            raise ValueError("shap_background_size must be at least two")
        if self.shap_max_evals < 1:
            raise ValueError("shap_max_evals must be positive")

    @property
    def rul_times(self) -> np.ndarray:
        """Monotone RUL evaluation grid beginning at zero."""

        steps = int(np.floor(self.rul_horizon_years / self.rul_step_years)) + 1
        grid = np.arange(steps, dtype=np.float64) * self.rul_step_years
        if grid[-1] < self.rul_horizon_years:
            grid = np.append(grid, float(self.rul_horizon_years))
        return grid


@dataclass(frozen=True)
class TriageResult:
    """Post-disaster findings, SHAP contributions, and the serialisable report."""

    triage_table: pd.DataFrame
    xai_contributions: pd.DataFrame
    report: dict[str, Any]
    metrics: dict[str, Any]
    event: DisasterEvent
    model_name: str
    explain_model_name: str
    feature_names: tuple[str, ...]
    force_plot_paths: tuple[str, ...]

    def save(self, output_dir: Path) -> None:
        """Persist findings, explanations, and force plots."""

        output_dir.mkdir(parents=True, exist_ok=True)
        _write_frame(self.triage_table, output_dir / "triage_findings.parquet")
        if not self.xai_contributions.empty:
            _write_frame(
                self.xai_contributions,
                output_dir / "xai_contributions.parquet",
            )
        with (output_dir / "triage_report.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(self.report), handle, indent=2)


# ---------------------------------------------------------------------------
# Post-disaster feature updater
# ---------------------------------------------------------------------------


class PostDisasterUpdater:
    """Apply a climate shock to the latest observable feature snapshot."""

    def apply(
        self,
        frame: pd.DataFrame,
        event: DisasterEvent,
        affected_ids: Sequence[str],
        *,
        id_col: str,
    ) -> pd.DataFrame:
        """Return a shocked copy of ``frame`` for the targeted bridge rows."""

        if id_col not in frame.columns:
            raise ValueError(f"frame is missing identifier column {id_col!r}")
        target_ids = {str(value) for value in affected_ids}
        if not target_ids:
            raise ValueError("affected_ids must not be empty")
        mask = frame[id_col].astype(str).isin(target_ids)
        if not mask.any():
            raise ValueError("none of the affected bridge IDs exists in the feature frame")
        output = frame.copy()
        climate_columns = (
            "MONSOON_ANOMALY",
            "SEASONAL_RAINFALL_MM",
            "PEAK_DISCHARGE_M3S",
            "PEAK_VELOCITY_MPS",
            "PEAK_SCOUR_INDEX",
            "FLOOD_EXCEEDANCE",
            "CUMULATIVE_FLOOD_EXCEEDANCE",
            "CUMULATIVE_SCOUR_LOAD",
            "PRE_ENTRY_MONSOON_MEAN_5Y",
            "PRE_ENTRY_MONSOON_MAX_5Y",
            "PRE_ENTRY_RAINFALL_MEAN_5Y",
            "PRE_ENTRY_SCOUR_MAX_5Y",
            "PRE_ENTRY_FLOOD_EXCEEDANCES_5Y",
            "EXPOSURE_MONSOON_MEAN",
            "EXPOSURE_SCOUR_MAX",
            "EXPOSURE_FLOOD_TOTAL",
        )
        for column in climate_columns:
            if column in output.columns:
                output[column] = output[column].astype(np.float64)
        self._apply_dynamic_shock(output, event, mask)
        self._apply_static_exposure_shock(output, event, mask)
        return output

    @staticmethod
    def _scale_column(
        frame: pd.DataFrame,
        mask: pd.Series,
        column: str,
        multiplier: float,
        severity: float,
    ) -> None:
        if column not in frame.columns:
            return
        effective = 1.0 + (float(multiplier) - 1.0) * float(severity)
        values = frame.loc[mask, column].to_numpy(np.float64) * effective
        frame.loc[mask, column] = np.maximum(values, 0.0)

    @staticmethod
    def _offset_column(
        frame: pd.DataFrame,
        mask: pd.Series,
        column: str,
        increment: float,
        severity: float,
    ) -> None:
        if column not in frame.columns:
            return
        frame.loc[mask, column] = (
            frame.loc[mask, column].to_numpy(np.float64)
            + float(increment) * float(severity)
        )

    def _apply_dynamic_shock(
        self,
        frame: pd.DataFrame,
        event: DisasterEvent,
        mask: pd.Series,
    ) -> None:
        severity = event.severity
        self._offset_column(
            frame, mask, "MONSOON_ANOMALY", event.monsoon_anomaly_increment, severity
        )
        self._scale_column(
            frame, mask, "SEASONAL_RAINFALL_MM", event.rainfall_multiplier, severity
        )
        self._scale_column(
            frame, mask, "PEAK_DISCHARGE_M3S", event.discharge_multiplier, severity
        )
        self._scale_column(
            frame, mask, "PEAK_VELOCITY_MPS", event.velocity_multiplier, severity
        )
        self._scale_column(
            frame, mask, "PEAK_SCOUR_INDEX", event.scour_multiplier, severity
        )
        if "FLOOD_EXCEEDANCE" in frame.columns:
            current = frame.loc[mask, "FLOOD_EXCEEDANCE"].to_numpy(np.float64)
            frame.loc[mask, "FLOOD_EXCEEDANCE"] = np.maximum(
                current, float(event.flood_exceedance_increment)
            )
        self._offset_column(
            frame,
            mask,
            "CUMULATIVE_FLOOD_EXCEEDANCE",
            float(event.flood_exceedance_increment),
            severity,
        )
        self._offset_column(
            frame,
            mask,
            "CUMULATIVE_SCOUR_LOAD",
            event.cumulative_scour_increment,
            severity,
        )

    def _apply_static_exposure_shock(
        self,
        frame: pd.DataFrame,
        event: DisasterEvent,
        mask: pd.Series,
    ) -> None:
        severity = event.severity
        self._offset_column(
            frame,
            mask,
            "PRE_ENTRY_MONSOON_MEAN_5Y",
            0.40 * event.monsoon_anomaly_increment,
            severity,
        )
        self._offset_column(
            frame,
            mask,
            "PRE_ENTRY_MONSOON_MAX_5Y",
            event.monsoon_anomaly_increment,
            severity,
        )
        self._scale_column(
            frame,
            mask,
            "PRE_ENTRY_RAINFALL_MEAN_5Y",
            1.0 + (event.rainfall_multiplier - 1.0) * 0.50,
            severity,
        )
        self._scale_column(
            frame,
            mask,
            "PRE_ENTRY_SCOUR_MAX_5Y",
            1.0 + (event.scour_multiplier - 1.0) * 0.50,
            severity,
        )
        self._offset_column(
            frame,
            mask,
            "PRE_ENTRY_FLOOD_EXCEEDANCES_5Y",
            float(event.flood_exceedance_increment),
            severity,
        )
        self._offset_column(
            frame,
            mask,
            "EXPOSURE_MONSOON_MEAN",
            0.40 * event.monsoon_anomaly_increment,
            severity,
        )
        self._scale_column(
            frame,
            mask,
            "EXPOSURE_SCOUR_MAX",
            1.0 + (event.scour_multiplier - 1.0) * 0.50,
            severity,
        )
        self._offset_column(
            frame,
            mask,
            "EXPOSURE_FLOOD_TOTAL",
            float(event.flood_exceedance_increment),
            severity,
        )


# ---------------------------------------------------------------------------
# Survival-curve and RUL mathematics
# ---------------------------------------------------------------------------


def _cox_conditional_curve(
    model: Any,
    frame: pd.DataFrame,
    time_at_event: np.ndarray,
    horizons: np.ndarray,
    *,
    survival_floor: float,
) -> np.ndarray:
    """Conditional survival ``S(t0 + h | t0, X_t0)`` for the Cox model."""

    log_hazard = np.asarray(model.predict_log_hazard(frame), dtype=np.float64)
    baseline_times, baseline_values = model.baseline_cumulative_hazard()
    cumulative_at_event = _cumulative_hazard_at(
        baseline_times, baseline_values, time_at_event
    )
    hazard_times = time_at_event[:, None] + horizons[None, :]
    cumulative_at_horizon = _cumulative_hazard_at(
        baseline_times,
        baseline_values,
        hazard_times.reshape(-1),
    ).reshape(hazard_times.shape)
    increment = np.maximum(
        cumulative_at_horizon - cumulative_at_event[:, None], 0.0
    )
    curve = np.exp(-np.exp(log_hazard)[:, None] * increment)
    curve = np.clip(curve, 0.0, 1.0)
    curve[:, horizons <= 0.0] = 1.0
    return np.minimum.accumulate(curve, axis=1)


def _step_model_conditional_curve(
    model: Any,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    time_at_event: np.ndarray,
    horizons: np.ndarray,
    *,
    survival_floor: float,
) -> np.ndarray:
    """Conditional survival curve for a scikit-survival step-function model.

    The existing Module 3 helper is reused for each horizon so that the
    piecewise-exponential tail extrapolation and domain clipping remain
    identical to the audit view.
    """

    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"explanation frame is missing features: {missing}")
    X = frame.loc[:, list(feature_names)].to_numpy(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("step-model explanation matrix contains non-finite values")
    curve = np.ones((X.shape[0], horizons.size), dtype=np.float64)
    for index, horizon in enumerate(horizons):
        if horizon <= 0.0:
            continue
        survival_at_event, survival_at_horizon = _conditional_survival_step_model(
            model,
            X,
            time_at_event,
            float(horizon),
            survival_floor,
        )
        conditional = survival_at_horizon / np.maximum(
            survival_at_event, survival_floor
        )
        curve[:, index] = np.clip(conditional, 0.0, 1.0)
    return np.minimum.accumulate(curve, axis=1)


def conditional_survival_curve(
    *,
    model_name: str,
    model: Any,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    time_at_event: np.ndarray,
    horizons: Sequence[float],
    survival_floor: float,
) -> np.ndarray:
    """Public survival-curve dispatcher for Cox and step-function models."""

    times = np.asarray(horizons, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or (times < 0).any():
        raise ValueError("horizons must be a non-empty non-negative 1-D sequence")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("horizons must be sorted in ascending order")
    if model_name == "cox_time_varying":
        curve = _cox_conditional_curve(
            model,
            frame,
            np.asarray(time_at_event, dtype=np.float64),
            times,
            survival_floor=survival_floor,
        )
    else:
        curve = _step_model_conditional_curve(
            model,
            frame,
            feature_names,
            np.asarray(time_at_event, dtype=np.float64),
            times,
            survival_floor=survival_floor,
        )
    if not np.isfinite(curve).all():
        raise ValueError("conditional survival curve contains non-finite values")
    return np.minimum.accumulate(np.clip(curve, 0.0, 1.0), axis=1)


def compute_rul_from_survival(
    survival: np.ndarray,
    times: Sequence[float],
    *,
    threshold: float,
    censor_at: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first threshold crossing and censoring flag per row.

    ``survival`` must contain a time-zero column equal to one.  If a curve never
    falls to the threshold within the evaluation grid, the RUL is right-censored
    at the final grid time.
    """

    values = np.asarray(survival, dtype=np.float64)
    grid = np.asarray(times, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("survival must be a two-dimensional matrix")
    if grid.ndim != 1 or grid.size != values.shape[1]:
        raise ValueError("times must align with survival columns")
    if np.any(np.diff(grid) <= 0.0) or abs(float(grid[0])) > 1e-12:
        raise ValueError("times must start at zero and be strictly increasing")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if not np.isfinite(values).all():
        raise ValueError("survival contains non-finite values")
    if (values < -1e-9).any() or (values > 1.0 + 1e-9).any():
        raise ValueError("survival contains probabilities outside [0, 1]")

    below = values <= threshold
    crossed = below.any(axis=1)
    first = np.argmax(below, axis=1)
    rows = np.arange(values.shape[0])
    previous = np.maximum(first - 1, 0)
    previous_survival = values[rows, previous]
    current_survival = values[rows, first]
    denominator = previous_survival - current_survival
    fraction = np.zeros_like(denominator)
    np.divide(
        previous_survival - threshold,
        denominator,
        out=fraction,
        where=denominator > 1e-12,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    crossed_rul = grid[previous] + fraction * (grid[first] - grid[previous])
    censoring_value = float(censor_at if censor_at is not None else grid[-1])
    rul = np.where(crossed, crossed_rul, censoring_value)
    return np.maximum(rul, 0.0), ~crossed


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


class _RiskPredictor:
    """Adapt a fitted survival model to SHAP's NumPy prediction protocol.

    ``SurvivalEnsembleModel.predict_risk`` rank-averages risks within the rows
    passed to it.  That is appropriate for concordance evaluation but not for
    pointwise SHAP attribution, because each explanation batch would receive a
    different rank scale.  For SHAP, the ensemble therefore uses a fixed
    z-score average of the two component risks calibrated on the background
    sample.  The underlying fitted models are not modified.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        *,
        model_name: str,
        ensemble_stats: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.model = model
        self.feature_names = tuple(feature_names)
        self.model_name = model_name
        self.ensemble_stats = ensemble_stats

    def __call__(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("SHAP prediction matrix does not match feature names")
        if self.model_name == "cox_time_varying":
            frame = pd.DataFrame(matrix, columns=list(self.feature_names))
            return np.asarray(self.model.predict_risk(frame), dtype=np.float64)
        if self.model_name == "ensemble":
            if self.ensemble_stats is None:
                raise RuntimeError("ensemble SHAP predictor requires calibration statistics")
            rsf_mean, rsf_scale, gb_mean, gb_scale = self.ensemble_stats
            rsf_risk = np.asarray(self.model.rsf.predict_risk(matrix), dtype=np.float64)
            gb_risk = np.asarray(self.model.gbsa.predict_risk(matrix), dtype=np.float64)
            return 0.5 * (
                (rsf_risk - rsf_mean) / rsf_scale
                + (gb_risk - gb_mean) / gb_scale
            )
        return np.asarray(self.model.predict_risk(matrix), dtype=np.float64)


def _ensemble_reference(
    model: Any, background: np.ndarray
) -> tuple[float, float, float, float]:
    """Fixed centring and scaling for pointwise ensemble SHAP values."""

    rsf_risk = np.asarray(model.rsf.predict_risk(background), dtype=np.float64)
    gb_risk = np.asarray(model.gbsa.predict_risk(background), dtype=np.float64)
    rsf_scale = max(float(np.std(rsf_risk)), 1e-9)
    gb_scale = max(float(np.std(gb_risk)), 1e-9)
    return (
        float(np.mean(rsf_risk)),
        rsf_scale,
        float(np.mean(gb_risk)),
        gb_scale,
    )


@dataclass(frozen=True)
class ShapExplanation:
    """Local SHAP values and optional force-plot files."""

    contributions: pd.DataFrame
    shap_values: np.ndarray
    base_values: np.ndarray
    predictions: np.ndarray
    feature_names: tuple[str, ...]
    bridge_ids: tuple[str, ...]
    force_plot_paths: tuple[str, ...]
    additivity_ok: bool
    max_additivity_error: float


class ShapRiskExplainer:
    """Model-agnostic SHAP explainer over survival risk predictions."""

    def __init__(self, config: TriageConfig | None = None) -> None:
        self.config = config or TriageConfig()

    def explain(
        self,
        *,
        model: Any,
        model_name: str,
        feature_names: Sequence[str],
        explain_frame: pd.DataFrame,
        background_frame: pd.DataFrame,
        bridge_ids: Sequence[str],
        output_dir: Path | None = None,
    ) -> ShapExplanation:
        """Compute local contributions for the selected bridges."""

        features = tuple(feature_names)
        if not features:
            raise ValueError("feature_names must not be empty")
        assert_feature_allowlist(features)
        for feature in features:
            if feature.startswith(FORBIDDEN_XAI_PREFIXES):
                raise ValueError(f"forbidden audit/truth feature in SHAP matrix: {feature}")
        ids = tuple(str(value) for value in bridge_ids)
        if not ids:
            return ShapExplanation(
                contributions=pd.DataFrame(),
                shap_values=np.empty((0, len(features))),
                base_values=np.empty(0),
                predictions=np.empty(0),
                feature_names=features,
                bridge_ids=(),
                force_plot_paths=(),
                additivity_ok=True,
                max_additivity_error=0.0,
            )
        missing = [name for name in features if name not in explain_frame.columns]
        if missing:
            raise ValueError(f"explain frame is missing features: {missing}")
        missing_background = [
            name for name in features if name not in background_frame.columns
        ]
        if missing_background:
            raise ValueError(
                f"background frame is missing features: {missing_background}"
            )
        if "BRIDGE_ID" not in explain_frame.columns:
            raise ValueError("explain frame is missing BRIDGE_ID")

        lookup = explain_frame.copy()
        lookup["BRIDGE_ID"] = lookup["BRIDGE_ID"].astype(str)
        lookup = lookup.set_index("BRIDGE_ID", drop=False)
        if not set(ids).issubset(lookup.index):
            missing_ids = sorted(set(ids).difference(lookup.index))[:5]
            raise ValueError(f"explanation bridge IDs are absent from the frame: {missing_ids}")
        ordered = lookup.loc[list(ids)]
        X = ordered.loc[:, list(features)].to_numpy(np.float64)
        if not np.isfinite(X).all():
            raise ValueError("explain matrix contains non-finite values")

        background = background_frame.loc[:, list(features)].to_numpy(np.float64)
        background = background[np.isfinite(background).all(axis=1)]
        if background.shape[0] < 2:
            raise ValueError("SHAP background requires at least two finite rows")
        rng = np.random.default_rng(self.config.random_state)
        if background.shape[0] > self.config.shap_background_size:
            positions = rng.choice(
                background.shape[0],
                size=self.config.shap_background_size,
                replace=False,
            )
            background = background[np.sort(positions)]

        ensemble_stats = (
            _ensemble_reference(model, background)
            if model_name == "ensemble"
            else None
        )
        predictor = _RiskPredictor(
            model,
            features,
            model_name=model_name,
            ensemble_stats=ensemble_stats,
        )
        explainer = shap.Explainer(
            predictor,
            background,
            algorithm="permutation",
            seed=self.config.random_state,
            feature_names=list(features),
        )
        max_evals = max(self.config.shap_max_evals, 2 * len(features) + 1)
        explanation = explainer(X, max_evals=max_evals)
        shap_values = np.asarray(explanation.values, dtype=np.float64)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]
        if shap_values.shape != X.shape:
            raise RuntimeError(
                f"unexpected SHAP value shape {shap_values.shape} for X {X.shape}"
            )
        base_values = np.asarray(explanation.base_values, dtype=np.float64)
        if base_values.ndim > 1:
            base_values = base_values.reshape(base_values.shape[0], -1)[:, 0]
        predictions = predictor(X)
        reconstruction = base_values + shap_values.sum(axis=1)
        error = np.abs(reconstruction - predictions)
        scale = max(1.0, float(np.std(predictions)))
        max_error = float(error.max()) if error.size else 0.0
        additivity_ok = bool(max_error <= max(1e-3, 0.05 * scale))

        records: list[dict[str, Any]] = []
        for row_index, bridge_id in enumerate(ids):
            row_values = shap_values[row_index]
            order = np.argsort(-np.abs(row_values))[:10]
            for rank, feature_index in enumerate(order, start=1):
                value = float(row_values[feature_index])
                records.append(
                    {
                        "BRIDGE_ID": bridge_id,
                        "RANK": rank,
                        "FEATURE": features[feature_index],
                        "FEATURE_VALUE": float(X[row_index, feature_index]),
                        "SHAP_VALUE": value,
                        "ABS_SHAP_VALUE": abs(value),
                        "DIRECTION": "INCREASES_RISK" if value >= 0.0 else "REDUCES_RISK",
                        "BASE_VALUE": float(base_values[row_index]),
                        "MODEL_RISK_PREDICTION": float(predictions[row_index]),
                    }
                )
        contributions = pd.DataFrame.from_records(records)

        force_paths: list[str] = []
        if self.config.save_force_plots and output_dir is not None:
            plot_dir = output_dir / "force_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for row_index, bridge_id in enumerate(ids):
                try:
                    plot = shap.force_plot(
                        float(base_values[row_index]),
                        shap_values[row_index],
                        X[row_index],
                        feature_names=list(features),
                        matplotlib=False,
                    )
                    path = plot_dir / f"{_safe_filename(bridge_id)}_force.html"
                    shap.save_html(str(path), plot)
                    force_paths.append(str(path))
                except Exception as error:  # pragma: no cover - plotting backend guard
                    LOGGER.warning(
                        "Could not render SHAP force plot for %s: %s", bridge_id, error
                    )
        return ShapExplanation(
            contributions=contributions,
            shap_values=shap_values,
            base_values=base_values,
            predictions=predictions,
            feature_names=features,
            bridge_ids=ids,
            force_plot_paths=tuple(force_paths),
            additivity_ok=additivity_ok,
            max_additivity_error=max_error,
        )


# ---------------------------------------------------------------------------
# Triage engine
# ---------------------------------------------------------------------------


class TriageEngine:
    """Recompute survival after a disaster and rank bridges for NDT triage."""

    def __init__(self, config: TriageConfig | None = None) -> None:
        self.config = config or TriageConfig()
        self.updater = PostDisasterUpdater()
        self.explainer = ShapRiskExplainer(self.config)

    def run_event(
        self,
        *,
        datasets: SurvivalDatasets,
        engine_result: SurvivalEngineResult,
        bridges: pd.DataFrame,
        event: DisasterEvent,
        audit_result: AuditVerifierResult | None = None,
        output_dir: Path | None = None,
    ) -> TriageResult:
        """Score one post-disaster event and explain the highest-risk bridges."""

        started = time.perf_counter()
        split = self.config.split
        model_name = self.config.model_name
        explain_model_name = self.config.explain_model_name
        if model_name not in engine_result.models:
            raise ValueError(f"unknown triage model: {model_name}")
        if explain_model_name not in engine_result.models:
            raise ValueError(f"unknown explanation model: {explain_model_name}")

        meta = _bridge_metadata(bridges)
        affected_ids = _affected_bridge_ids(bridges, event)
        if not affected_ids:
            raise ValueError(
                f"event {event.event_id!r} matches no bridges in the cohort"
            )

        triage_frame, time_column, id_column = _latest_model_frame(
            datasets, model_name, split
        )
        triage_frame = triage_frame.copy()
        triage_frame[id_column] = triage_frame[id_column].astype(str)
        triage_frame = triage_frame[triage_frame[id_column].isin(affected_ids)].copy()
        if triage_frame.empty:
            raise ValueError(
                f"none of the affected bridges is present in the {split} split"
            )
        scored_ids = set(triage_frame[id_column].astype(str))
        shock_frame = self.updater.apply(
            triage_frame,
            event,
            affected_ids=scored_ids,
            id_col=id_column,
        )

        feature_names = _model_feature_names(engine_result, model_name)
        assert_feature_allowlist(feature_names)
        time_at_event = triage_frame[time_column].to_numpy(np.float64)
        curve_times = self.config.rul_times
        horizon_times = np.asarray(self.config.horizons, dtype=np.float64)

        baseline_curve = conditional_survival_curve(
            model_name=model_name,
            model=engine_result.models[model_name],
            frame=triage_frame,
            feature_names=feature_names,
            time_at_event=time_at_event,
            horizons=curve_times,
            survival_floor=self.config.survival_floor,
        )
        post_curve = conditional_survival_curve(
            model_name=model_name,
            model=engine_result.models[model_name],
            frame=shock_frame,
            feature_names=feature_names,
            time_at_event=time_at_event,
            horizons=curve_times,
            survival_floor=self.config.survival_floor,
        )
        _validate_curves(baseline_curve, curve_times, label="baseline")
        _validate_curves(post_curve, curve_times, label="post-event")

        baseline_rul, baseline_censored = compute_rul_from_survival(
            baseline_curve,
            curve_times,
            threshold=self.config.rul_threshold,
            censor_at=self.config.rul_horizon_years,
        )
        post_rul, post_censored = compute_rul_from_survival(
            post_curve,
            curve_times,
            threshold=self.config.rul_threshold,
            censor_at=self.config.rul_horizon_years,
        )

        baseline_horizon = conditional_survival_curve(
            model_name=model_name,
            model=engine_result.models[model_name],
            frame=triage_frame,
            feature_names=feature_names,
            time_at_event=time_at_event,
            horizons=horizon_times,
            survival_floor=self.config.survival_floor,
        )
        post_horizon = conditional_survival_curve(
            model_name=model_name,
            model=engine_result.models[model_name],
            frame=shock_frame,
            feature_names=feature_names,
            time_at_event=time_at_event,
            horizons=horizon_times,
            survival_floor=self.config.survival_floor,
        )
        decision_index = self.config.horizons.index(self.config.decision_horizon_years)

        table = pd.DataFrame(
            {
                "BRIDGE_ID": triage_frame[id_column].astype(str).to_numpy(),
                "DATA_SPLIT": split,
                "TIME_AT_EVENT_YEARS": time_at_event.astype(np.float32),
                "EVENT_ID": event.event_id,
                "EVENT_TYPE": event.event_type,
                "EVENT_SEVERITY": np.float32(event.severity),
                "BASELINE_SURVIVAL_AT_DECISION": baseline_horizon[
                    :, decision_index
                ].astype(np.float32),
                "POST_EVENT_SURVIVAL_AT_DECISION": post_horizon[
                    :, decision_index
                ].astype(np.float32),
                "SURVIVAL_DROP_AT_DECISION": (
                    baseline_horizon[:, decision_index]
                    - post_horizon[:, decision_index]
                ).astype(np.float32),
                "RUL_BASELINE_YEARS": baseline_rul.astype(np.float32),
                "RUL_POST_EVENT_YEARS": post_rul.astype(np.float32),
                "RUL_LOSS_YEARS": np.maximum(
                    baseline_rul - post_rul, 0.0
                ).astype(np.float32),
                "RUL_BASELINE_CENSORED": baseline_censored,
                "RUL_POST_EVENT_CENSORED": post_censored,
            }
        )
        base_survival = table["BASELINE_SURVIVAL_AT_DECISION"].to_numpy(np.float64)
        post_survival = table["POST_EVENT_SURVIVAL_AT_DECISION"].to_numpy(np.float64)
        base_rul = table["RUL_BASELINE_YEARS"].to_numpy(np.float64)
        rul_loss = table["RUL_LOSS_YEARS"].to_numpy(np.float64)
        risk_post = np.clip(1.0 - post_survival, 0.0, 1.0)
        risk_increase = np.clip(base_survival - post_survival, 0.0, 1.0)
        rul_loss_ratio = np.clip(
            rul_loss / np.maximum(base_rul, self.config.rul_step_years), 0.0, 1.0
        )
        priority_score = (
            0.50 * risk_post + 0.30 * risk_increase + 0.20 * rul_loss_ratio
        )
        table["TRIAGE_PRIORITY_SCORE"] = priority_score.astype(np.float32)

        table = table.merge(meta, on="BRIDGE_ID", how="left", validate="one_to_one")
        if table[["STATE", "DISTRICT"]].isna().any().any():
            raise ValueError("bridge metadata join left missing state/district fields")
        table = _attach_audit_findings(table, audit_result)

        audit_suspect = table.get(
            "PAPER_INSPECTION_SUSPECT", pd.Series(False, index=table.index)
        ).to_numpy(bool)
        audit_alert = table.get(
            "AUDIT_ALERT", pd.Series(False, index=table.index)
        ).to_numpy(bool)
        critical = (post_survival < self.config.survival_threshold) | (
            table["RUL_POST_EVENT_YEARS"].to_numpy(np.float64)
            <= self.config.crisis_rul_years
        )
        high = (
            ~critical
            & (
                (table["RUL_LOSS_YEARS"].to_numpy(np.float64) >= self.config.crisis_rul_years)
                | audit_suspect
                | (risk_increase >= 0.10)
            )
        )
        medium = ~critical & ~high & (
            (risk_increase > 0.02) | audit_alert
        )
        table["TRIAGE_PRIORITY"] = np.select(
            [critical, high, medium],
            ["CRITICAL", "HIGH", "MEDIUM"],
            default="LOW",
        )
        table["IMMEDIATE_NDT_DEPLOYMENT"] = critical | high | audit_suspect
        table["AUDIT_ESCALATION"] = audit_suspect
        score_order = np.argsort(
            -table["TRIAGE_PRIORITY_SCORE"].to_numpy(np.float64), kind="mergesort"
        )
        ordinal_rank = np.empty(len(table), dtype=np.int32)
        ordinal_rank[score_order] = np.arange(1, len(table) + 1, dtype=np.int32)
        table["RANK"] = ordinal_rank
        table = table.sort_values(
            ["TRIAGE_PRIORITY_SCORE", "BRIDGE_ID"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        explanation_ids = tuple(
            table["BRIDGE_ID"].head(self.config.top_n_explanations).astype(str)
        )
        explanation = self._explain(
            datasets=datasets,
            engine_result=engine_result,
            event=event,
            explain_model_name=explain_model_name,
            bridge_ids=explanation_ids,
            output_dir=output_dir,
        )
        if not explanation.contributions.empty:
            table = table.merge(
                explanation.contributions.groupby("BRIDGE_ID", as_index=False).agg(
                    TOP_RISK_DRIVER=("FEATURE", "first"),
                    TOP_RISK_DRIVER_SHAP=("SHAP_VALUE", "first"),
                ),
                on="BRIDGE_ID",
                how="left",
            )

        report = self._build_report(
            event=event,
            engine_result=engine_result,
            model_name=model_name,
            explain_model_name=explain_model_name,
            feature_names=feature_names,
            affected_ids=affected_ids,
            scored_ids=scored_ids,
            table=table,
            explanation=explanation,
            elapsed_seconds=time.perf_counter() - started,
        )
        metrics = _triage_metrics(table, explanation)
        return TriageResult(
            triage_table=table,
            xai_contributions=explanation.contributions,
            report=report,
            metrics=metrics,
            event=event,
            model_name=model_name,
            explain_model_name=explain_model_name,
            feature_names=explanation.feature_names,
            force_plot_paths=explanation.force_plot_paths,
        )

    def _explain(
        self,
        *,
        datasets: SurvivalDatasets,
        engine_result: SurvivalEngineResult,
        event: DisasterEvent,
        explain_model_name: str,
        bridge_ids: Sequence[str],
        output_dir: Path | None,
    ) -> ShapExplanation:
        if not bridge_ids:
            return ShapExplanation(
                contributions=pd.DataFrame(),
                shap_values=np.empty((0, 0)),
                base_values=np.empty(0),
                predictions=np.empty(0),
                feature_names=(),
                bridge_ids=(),
                force_plot_paths=(),
                additivity_ok=True,
                max_additivity_error=0.0,
            )
        features = _model_feature_names(engine_result, explain_model_name)
        explain_frame, _, id_column = _latest_model_frame(
            datasets, explain_model_name, self.config.split
        )
        background_frame, _, _ = _latest_model_frame(
            datasets, explain_model_name, "train"
        )
        explain_frame = explain_frame.copy()
        explain_frame["BRIDGE_ID"] = explain_frame[id_column].astype(str)
        background_frame = background_frame.copy()
        background_frame["BRIDGE_ID"] = background_frame[id_column].astype(str)
        explain_frame = self.updater.apply(
            explain_frame,
            event,
            affected_ids=bridge_ids,
            id_col=id_column,
        )
        return self.explainer.explain(
            model=engine_result.models[explain_model_name],
            model_name=explain_model_name,
            feature_names=features,
            explain_frame=explain_frame,
            background_frame=background_frame,
            bridge_ids=bridge_ids,
            output_dir=output_dir,
        )

    def _build_report(
        self,
        *,
        event: DisasterEvent,
        engine_result: SurvivalEngineResult,
        model_name: str,
        explain_model_name: str,
        feature_names: Sequence[str],
        affected_ids: set[str],
        scored_ids: set[str],
        table: pd.DataFrame,
        explanation: ShapExplanation,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        priority_counts = {
            str(key): int(value)
            for key, value in table["TRIAGE_PRIORITY"].value_counts().items()
        }
        top_records = table.head(20).to_dict("records")
        top_findings = [
            {
                "bridge_id": str(record["BRIDGE_ID"]),
                "district": str(record.get("DISTRICT", "")),
                "baseline_survival": float(record["BASELINE_SURVIVAL_AT_DECISION"]),
                "post_event_survival": float(
                    record["POST_EVENT_SURVIVAL_AT_DECISION"]
                ),
                "rul_baseline_years": float(record["RUL_BASELINE_YEARS"]),
                "rul_post_event_years": float(record["RUL_POST_EVENT_YEARS"]),
                "rul_loss_years": float(record["RUL_LOSS_YEARS"]),
                "priority": str(record["TRIAGE_PRIORITY"]),
                "immediate_ndt": bool(record["IMMEDIATE_NDT_DEPLOYMENT"]),
                "audit_suspect": bool(record.get("PAPER_INSPECTION_SUSPECT", False)),
            }
            for record in top_records
        ]
        monotone_ok = bool(
            (
                table["POST_EVENT_SURVIVAL_AT_DECISION"].to_numpy(np.float64)
                <= table["BASELINE_SURVIVAL_AT_DECISION"].to_numpy(np.float64)
                + 1e-6
            ).all()
        )
        return {
            "schema_version": TRIAGE_SCHEMA_VERSION,
            "module": "triage_and_xai",
            "config": _jsonable(asdict(self.config)),
            "event": _jsonable(asdict(event)),
            "cohort": {
                "n_targeted": int(len(affected_ids)),
                "n_scored": int(len(scored_ids)),
                "n_rows": int(len(table)),
                "split": self.config.split,
                "model_name": model_name,
                "engine_selected_model": engine_result.selected_model,
                "explain_model_name": explain_model_name,
                "n_xai_features": len(feature_names),
                "xai_features": list(feature_names),
            },
            "findings": {
                "n_immediate_ndt": int(table["IMMEDIATE_NDT_DEPLOYMENT"].sum()),
                "n_critical": int((table["TRIAGE_PRIORITY"] == "CRITICAL").sum()),
                "n_high": int((table["TRIAGE_PRIORITY"] == "HIGH").sum()),
                "n_medium": int((table["TRIAGE_PRIORITY"] == "MEDIUM").sum()),
                "n_low": int((table["TRIAGE_PRIORITY"] == "LOW").sum()),
                "priority_counts": priority_counts,
                "mean_survival_drop": float(
                    table["SURVIVAL_DROP_AT_DECISION"].mean()
                ),
                "max_survival_drop": float(
                    table["SURVIVAL_DROP_AT_DECISION"].max()
                ),
                "mean_rul_loss_years": float(table["RUL_LOSS_YEARS"].mean()),
                "max_rul_loss_years": float(table["RUL_LOSS_YEARS"].max()),
                "n_audit_escalations": int(table["AUDIT_ESCALATION"].sum()),
                "top_findings": top_findings,
            },
            "xai": {
                "n_explained": len(explanation.bridge_ids),
                "n_contributions": int(len(explanation.contributions)),
                "additivity_ok": bool(explanation.additivity_ok),
                "max_additivity_error": float(explanation.max_additivity_error),
                "force_plot_files": list(explanation.force_plot_paths),
            },
            "checks": {
                "event_targets_known_bridges": True,
                "objective_features_only": True,
                "survival_curves_monotone": True,
                "rul_non_negative": bool(
                    (table["RUL_POST_EVENT_YEARS"].to_numpy(np.float64) >= 0.0).all()
                ),
                "rul_loss_non_negative": bool(
                    (table["RUL_LOSS_YEARS"].to_numpy(np.float64) >= 0.0).all()
                ),
                "post_event_survival_not_above_baseline": monotone_ok,
                "shap_additivity_ok": bool(explanation.additivity_ok),
                "force_plots_written": (
                    not self.config.save_force_plots
                    or not explanation.bridge_ids
                    or bool(explanation.force_plot_paths)
                ),
            },
            "runtime_seconds": float(elapsed_seconds),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bridge_metadata(bridges: pd.DataFrame) -> pd.DataFrame:
    required = [ID_COL, "STATE", "DISTRICT"]
    missing = [column for column in required if column not in bridges.columns]
    if missing:
        raise ValueError(f"bridge table is missing metadata columns: {missing}")
    columns = [
        column
        for column in (
            ID_COL,
            "STATE",
            "DISTRICT",
            "YEAR_BUILT",
            "STRUCTURE_KIND_MATERIAL",
            "STRUCTURE_TYPE_DESIGN",
            "MAX_SPAN_LEN",
            "WATERWAY_ADEQUACY",
            "LAST_INSPECTION_AGE",
            "REPORTED_LAST_MIN_COND",
        )
        if column in bridges.columns
    ]
    meta = bridges.loc[:, columns].copy()
    meta[ID_COL] = meta[ID_COL].astype(str)
    if meta[ID_COL].duplicated().any():
        raise ValueError("bridge metadata contains duplicate bridge IDs")
    rename = {ID_COL: "BRIDGE_ID"}
    if "LAST_INSPECTION_AGE" in meta.columns:
        rename["LAST_INSPECTION_AGE"] = "AGE_AT_AUDIT"
    return meta.rename(columns=rename).reset_index(drop=True)


def _affected_bridge_ids(bridges: pd.DataFrame, event: DisasterEvent) -> set[str]:
    ids = bridges[ID_COL].astype(str)
    if event.bridge_ids:
        return set(ids[ids.isin(set(event.bridge_ids))])
    mask = pd.Series(True, index=bridges.index)
    if event.district is not None:
        if "DISTRICT" not in bridges.columns:
            raise ValueError("bridge table has no DISTRICT column")
        mask &= bridges["DISTRICT"].astype(str).eq(str(event.district))
    if event.state is not None:
        if "STATE" not in bridges.columns:
            raise ValueError("bridge table has no STATE column")
        mask &= bridges["STATE"].astype(str).eq(str(event.state))
    return set(ids[mask])


def _model_feature_names(
    engine_result: SurvivalEngineResult, model_name: str
) -> tuple[str, ...]:
    if model_name == "cox_time_varying":
        return tuple(engine_result.time_varying_feature_names)
    return tuple(engine_result.static_feature_names)


def _latest_model_frame(
    datasets: SurvivalDatasets,
    model_name: str,
    split: TriageSplit,
) -> tuple[pd.DataFrame, str, str]:
    if model_name == "cox_time_varying":
        frame = getattr(datasets, f"counting_{split}")
        if frame.empty:
            raise ValueError(f"counting_{split} is empty")
        ordered = frame.sort_values(["id", "start"], kind="mergesort")
        latest = ordered.groupby("id", observed=True, sort=False).tail(1).copy()
        if latest["id"].duplicated().any():
            raise ValueError("latest counting-process frame contains duplicate bridges")
        return latest.reset_index(drop=True), "stop", "id"
    frame = getattr(datasets, f"static_{split}")
    if frame.empty:
        raise ValueError(f"static_{split} is empty")
    if frame["bridge_id"].duplicated().any():
        raise ValueError("static frame contains duplicate bridges")
    return frame.copy(), "duration", "bridge_id"


def _attach_audit_findings(
    table: pd.DataFrame,
    audit_result: AuditVerifierResult | None,
) -> pd.DataFrame:
    output = table.copy()
    default_columns = {
        "PAPER_INSPECTION_SUSPECT": False,
        "AUDIT_ALERT": False,
        "AUDIT_PRIORITY": "NOT_AUDITED",
        "AUDIT_Z_SCORE": np.nan,
        "ISOLATION_FLAG": False,
        "EXPECTED_SURVIVAL": np.nan,
    }
    for column, value in default_columns.items():
        if column not in output.columns:
            output[column] = value
    if audit_result is None:
        return output
    available = [
        column
        for column in (
            "BRIDGE_ID",
            "PAPER_INSPECTION_SUSPECT",
            "AUDIT_ALERT",
            "AUDIT_PRIORITY",
            "AUDIT_Z_SCORE",
            "ISOLATION_FLAG",
            "EXPECTED_SURVIVAL",
        )
        if column in audit_result.audit_table.columns
    ]
    if "BRIDGE_ID" not in available:
        return output
    audit = audit_result.audit_table.loc[:, available].copy()
    audit["BRIDGE_ID"] = audit["BRIDGE_ID"].astype(str)
    if audit["BRIDGE_ID"].duplicated().any():
        raise ValueError("audit table contains duplicate bridge IDs")
    output = output.drop(
        columns=[
            column
            for column in default_columns
            if column in output.columns and column != "BRIDGE_ID"
        ],
        errors="ignore",
    )
    output = output.merge(
        audit,
        on="BRIDGE_ID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_AUDIT"),
    )
    for column, value in default_columns.items():
        if column in output.columns:
            output[column] = output[column].fillna(value)
        else:
            output[column] = value
    output["PAPER_INSPECTION_SUSPECT"] = output[
        "PAPER_INSPECTION_SUSPECT"
    ].astype(bool)
    output["AUDIT_ALERT"] = output["AUDIT_ALERT"].astype(bool)
    output["ISOLATION_FLAG"] = output["ISOLATION_FLAG"].astype(bool)
    return output


def _validate_curves(
    curve: np.ndarray, times: Sequence[float], *, label: str
) -> None:
    if curve.ndim != 2:
        raise ValueError(f"{label} survival curve must be two-dimensional")
    if curve.shape[1] != len(times):
        raise ValueError(f"{label} survival curve does not align with times")
    if not np.isfinite(curve).all():
        raise ValueError(f"{label} survival curve contains non-finite values")
    if (curve < -1e-9).any() or (curve > 1.0 + 1e-9).any():
        raise ValueError(f"{label} survival curve contains invalid probabilities")
    if (np.diff(curve, axis=1) > 1e-9).any():
        raise ValueError(f"{label} survival curve is not non-increasing")


def _triage_metrics(
    table: pd.DataFrame, explanation: ShapExplanation
) -> dict[str, Any]:
    return {
        "n_scored": int(len(table)),
        "n_immediate_ndt": int(table["IMMEDIATE_NDT_DEPLOYMENT"].sum()),
        "mean_survival_drop": float(table["SURVIVAL_DROP_AT_DECISION"].mean()),
        "max_survival_drop": float(table["SURVIVAL_DROP_AT_DECISION"].max()),
        "mean_rul_loss_years": float(table["RUL_LOSS_YEARS"].mean()),
        "max_rul_loss_years": float(table["RUL_LOSS_YEARS"].max()),
        "priority_counts": {
            str(key): int(value)
            for key, value in table["TRIAGE_PRIORITY"].value_counts().items()
        },
        "n_audit_escalations": int(table["AUDIT_ESCALATION"].sum()),
        "shap": {
            "n_explained": len(explanation.bridge_ids),
            "additivity_ok": bool(explanation.additivity_ok),
            "max_additivity_error": float(explanation.max_additivity_error),
        },
    }


def _safe_filename(value: str) -> str:
    allowed = "-_."
    return "".join(
        character if character.isalnum() or character in allowed else "_"
        for character in value
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError, OSError):
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
    """Load a Module 2 fitted engine result from disk."""

    bundle = joblib.load(engine_dir / "survival_engine.joblib")
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


def load_audit_result(audit_dir: Path) -> AuditVerifierResult:
    """Load a Module 3 audit result from disk."""

    bundle = joblib.load(audit_dir / "audit_models.joblib")
    calibration_data = bundle["calibration"]
    report_path = audit_dir / "audit_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    metrics = report.get("ground_truth_evaluation") or {}
    return AuditVerifierResult(
        audit_table=_read_frame(audit_dir / "audit_findings.parquet"),
        report=report,
        metrics=metrics,
        model_name=str(bundle["model_name"]),
        audit_feature_columns=tuple(bundle["audit_feature_columns"]),
        isolation_forest=bundle["isolation_forest"],
        feature_pipeline=bundle["feature_pipeline"],
        calibration=AuditCalibration(**calibration_data),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-disaster bridge triage and SHAP survival explanations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Module 1 artifact directory. Generated only when --engine-dir is absent.",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("artifacts/module2"),
        help="Fitted Module 2 artifact directory.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Optional Module 3 artifact directory.",
    )
    parser.add_argument("--n-bridges", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--district", type=str, default="Darbhanga")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--event-type", type=str, default="COMPOUND")
    parser.add_argument("--severity", type=float, default=1.35)
    parser.add_argument("--horizons", type=str, default="5,10,15,25,40")
    parser.add_argument("--decision-horizon", type=float, default=5.0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model", type=str, default="cox_time_varying")
    parser.add_argument("--explain-model", type=str, default="ensemble")
    parser.add_argument("--top-explanations", type=int, default=8)
    parser.add_argument("--no-force-plots", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/module4")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    horizons = tuple(float(value) for value in args.horizons.split(",") if value.strip())

    engine_path = args.engine_dir / "survival_engine.joblib"
    if engine_path.exists():
        if args.data_dir is None:
            raise ValueError("--data-dir is required when loading a fitted engine")
        datasets = load_processed_datasets(args.data_dir)
        engine_result = load_engine_result(args.engine_dir)
        bridges = _read_frame(args.data_dir / "bridges.parquet")
    else:
        dataset = SyntheticBridgeDataGenerator(
            SyntheticDataConfig(n_bridges=args.n_bridges, random_state=args.seed)
        ).generate()
        prepared = BridgeDataPipeline(
            PipelineConfig(random_state=args.seed, max_fit_rows=20_000)
        ).prepare(dataset)
        if args.data_dir is not None:
            save_prepared_dataset(dataset, prepared, args.data_dir)
        engine_result = SurvivalEngine(
            SurvivalEngineConfig(
                horizons=horizons,
                rsf_n_estimators=80,
                gbsa_n_estimators=80,
                n_jobs=1,
                random_state=args.seed,
            )
        ).fit(prepared)
        engine_result.save(args.engine_dir)
        datasets = prepared
        bridges = dataset.bridges

    audit_result = None
    if args.audit_dir is not None:
        audit_path = args.audit_dir / "audit_models.joblib"
        if audit_path.exists():
            audit_result = load_audit_result(args.audit_dir)
        else:
            LOGGER.warning("No audit artifacts found in %s", args.audit_dir)

    district = args.district
    if args.state is None and district is not None:
        observed = set(bridges["DISTRICT"].astype(str))
        if district not in observed:
            district = str(bridges["DISTRICT"].astype(str).value_counts().index[0])
            LOGGER.warning("Requested district absent; using %s", district)
    event = DisasterEvent(
        event_id=f"{args.event_type.lower()}-{args.seed}",
        event_type=args.event_type,  # type: ignore[arg-type]
        district=district,
        state=args.state,
        severity=args.severity,
    )
    config = TriageConfig(
        model_name=args.model,
        explain_model_name=args.explain_model,
        split=args.split,  # type: ignore[arg-type]
        horizons=horizons,
        decision_horizon_years=args.decision_horizon,
        top_n_explanations=args.top_explanations,
        save_force_plots=not args.no_force_plots,
        random_state=args.seed,
    )
    result = TriageEngine(config).run_event(
        datasets=datasets,
        engine_result=engine_result,
        bridges=bridges,
        event=event,
        audit_result=audit_result,
        output_dir=args.output_dir,
    )
    result.save(args.output_dir)
    LOGGER.info(
        "Triaged %s bridges affected by %s | immediate NDT=%s | mean RUL loss=%.2f y",
        result.metrics["n_scored"],
        event.event_id,
        result.metrics["n_immediate_ndt"],
        result.metrics["mean_rul_loss_years"],
    )
    LOGGER.info("Wrote triage report and explanations to %s", args.output_dir)


if __name__ == "__main__":
    main()
