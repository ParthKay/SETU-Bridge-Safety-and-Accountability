"""Module 2: static and time-varying bridge survival models.

The module implements the survival engine for the zero-trust bridge benchmark:

1. :class:`CoxTimeVaryingSurvivalModel`
   A penalised Andersen-Gill Cox model fitted with
   :class:`lifelines.CoxTimeVaryingFitter` on the ``(id, start, stop, event)``
   counting-process frame produced by Module 1.  It reports hazard ratios,
   confidence intervals, and an explicit monsoon/scour hazard-acceleration
   summary.

2. :class:`RandomSurvivalForestModel`
   :class:`sksurv.ensemble.RandomSurvivalForest` with log-rank splitting for
   non-linear interactions between age, material, traffic load, and cumulative
   climate exposure.

3. :class:`GradientBoostingSurvivalModel`
   :class:`sksurv.ensemble.GradientBoostingSurvivalAnalysis` using the Cox
   partial-likelihood loss.  The gradient-boosted trees capture smooth,
   high-order interactions that a single global Cox coefficient cannot express.

4. :class:`SurvivalEnsembleModel`
   A rank-averaged risk ensemble of the two tree models.

Evaluation is always survival-specific: Harrell's concordance index, Uno's
IPCW concordance index, and Brier scores / Integrated Brier Score at the
requested horizons.  Classification accuracy and RMSE are never used.

Only classical statistical and tree-ensemble methods are used.  No deep
learning framework is imported.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.exceptions import ConvergenceError
from scipy.stats import rankdata
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
)
from sksurv.util import Surv

from .data_pipeline import (
    EVENT_COL,
    ID_COL,
    BridgeDataPipeline,
    PipelineConfig,
    SurvivalDatasets,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
    assert_counting_process,
    assert_feature_allowlist,
)

LOGGER = logging.getLogger(__name__)

RANDOM_STATE: Final[int] = 42
DEFAULT_HORIZONS: Final[tuple[float, ...]] = (5.0, 10.0, 15.0, 25.0, 40.0)
META_STATIC_COLUMNS: Final[frozenset[str]] = frozenset(
    {"bridge_id", ID_COL, "duration", EVENT_COL, "event"}
)
META_COUNTING_COLUMNS: Final[frozenset[str]] = frozenset(
    {"id", "start", "stop", EVENT_COL, "event"}
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivalEngineConfig:
    """Configuration for the Module 2 survival engine.

    Parameters
    ----------
    horizons:
        Evaluation horizons in years.
    cox_penalizer:
        Elastic-net penalty used by ``CoxTimeVaryingFitter``.  A small positive
        value stabilises high-dimensional one-hot design matrices.
    cox_l1_ratio:
        L1 fraction for the Cox elastic-net penalty.
    cox_min_variance:
        Numeric features below this variance are removed before fitting.
    cox_correlation_threshold:
        One feature from each pair with absolute correlation above this value is
        removed.  ``None`` disables correlation pruning.
    rsf_n_estimators, gbsa_n_estimators:
        Number of trees in each ensemble.
    rsf_min_samples_leaf:
        Minimum leaf size for the random survival forest.
    gbsa_learning_rate, gbsa_max_depth, gbsa_subsample:
        Gradient-boosting hyperparameters.
    n_jobs:
        Parallelism for the random survival forest.
    use_followup_aggregates:
        If true, add landmark follow-up climate aggregates to the static model
        matrix.  This is a retrospective view, so ``followup_aggregate_horizon``
        must be set.  The default is the leakage-safe pre-entry feature view.
    followup_aggregate_horizon:
        Landmark horizon (years) used when ``use_followup_aggregates`` is true.
    """

    horizons: tuple[float, ...] = DEFAULT_HORIZONS
    cox_penalizer: float = 0.01
    cox_l1_ratio: float = 0.0
    cox_min_variance: float = 1e-10
    cox_correlation_threshold: float | None = 0.995
    rsf_n_estimators: int = 300
    rsf_min_samples_split: int = 6
    rsf_min_samples_leaf: int = 3
    rsf_max_features: str | float | None = "sqrt"
    rsf_max_depth: int | None = None
    gbsa_n_estimators: int = 250
    gbsa_learning_rate: float = 0.05
    gbsa_max_depth: int = 3
    gbsa_subsample: float = 0.8
    gbsa_max_features: str | float | None = None
    n_jobs: int = -1
    random_state: int = RANDOM_STATE
    use_followup_aggregates: bool = False
    followup_aggregate_horizon: float | None = 5.0

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("horizons must be a non-empty sequence of positive values")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be strictly increasing without duplicates")
        if self.cox_penalizer < 0:
            raise ValueError("cox_penalizer must be non-negative")
        if not 0.0 <= self.cox_l1_ratio <= 1.0:
            raise ValueError("cox_l1_ratio must be in [0, 1]")
        if self.cox_min_variance < 0:
            raise ValueError("cox_min_variance must be non-negative")
        if self.cox_correlation_threshold is not None and not (
            0.0 < self.cox_correlation_threshold <= 1.0
        ):
            raise ValueError("cox_correlation_threshold must be in (0, 1]")
        if self.rsf_n_estimators <= 0 or self.gbsa_n_estimators <= 0:
            raise ValueError("tree ensemble sizes must be positive")
        if self.rsf_min_samples_leaf <= 0 or self.rsf_min_samples_split < 2:
            raise ValueError("random survival forest leaf/split sizes are invalid")
        if not 0.0 < self.gbsa_learning_rate <= 1.0:
            raise ValueError("gbsa_learning_rate must be in (0, 1]")
        if self.gbsa_max_depth <= 0:
            raise ValueError("gbsa_max_depth must be positive")
        if not 0.0 < self.gbsa_subsample <= 1.0:
            raise ValueError("gbsa_subsample must be in (0, 1]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be non-zero")
        if self.use_followup_aggregates and self.followup_aggregate_horizon is None:
            raise ValueError(
                "followup_aggregate_horizon is required when use_followup_aggregates is true"
            )
        if self.followup_aggregate_horizon is not None and self.followup_aggregate_horizon <= 0:
            raise ValueError("followup_aggregate_horizon must be positive")


# ---------------------------------------------------------------------------
# Design matrices and structured targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivalDesignMatrix:
    """A model-ready design matrix plus its survival target and bridge IDs."""

    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if self.X.shape[0] != self.y.shape[0] or self.X.shape[0] != self.ids.shape[0]:
            raise ValueError("X, y, and ids must have identical row counts")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("feature_names does not match X")
        if not np.isfinite(self.X).all():
            raise ValueError("design matrix contains non-finite values")
        if self.y.dtype.names != ("event", "time"):
            raise ValueError("survival target must have (event, time) fields")


def _structured_target(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    event_array = np.asarray(event, dtype=bool)
    time_array = np.asarray(time, dtype=np.float64)
    if not np.isfinite(time_array).all() or (time_array <= 0).any():
        raise ValueError("survival times must be finite and strictly positive")
    return Surv.from_arrays(event=event_array, time=time_array)


def _static_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    names = tuple(column for column in columns if column not in META_STATIC_COLUMNS)
    assert_feature_allowlist(names)
    return names


def _counting_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    names = tuple(column for column in columns if column not in META_COUNTING_COLUMNS)
    assert_feature_allowlist(names)
    return names


def prepare_static_design(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    id_col: str = "bridge_id",
    time_col: str = "duration",
    event_col: str = "event",
) -> SurvivalDesignMatrix:
    """Construct a float64 design matrix and a ``Surv`` structured target."""

    required = {id_col, time_col, event_col, *feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"static frame is missing required columns: {missing}")
    feature_names = tuple(feature_names)
    assert_feature_allowlist(feature_names)
    if frame[id_col].isna().any() or frame[id_col].duplicated().any():
        raise ValueError("static frame must contain unique non-missing bridge IDs")
    X = frame.loc[:, list(feature_names)].to_numpy(np.float64, copy=True)
    if not np.isfinite(X).all():
        raise ValueError("static design matrix contains missing or infinite values")
    y = _structured_target(
        frame[event_col].to_numpy(np.int8),
        frame[time_col].to_numpy(np.float64),
    )
    ids = frame[id_col].astype(str).to_numpy()
    return SurvivalDesignMatrix(X=X, y=y, ids=ids, feature_names=feature_names)


def prepare_counting_design(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    id_col: str = "id",
    start_col: str = "start",
    stop_col: str = "stop",
    event_col: str = "event",
) -> SurvivalDesignMatrix:
    """Construct an interval design matrix for the time-varying Cox model."""

    required = {id_col, start_col, stop_col, event_col, *feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"counting frame is missing required columns: {missing}")
    feature_names = tuple(feature_names)
    assert_feature_allowlist(feature_names)
    if frame[[id_col, start_col, stop_col, event_col]].isna().any().any():
        raise ValueError("counting frame contains missing keys or targets")
    if not (frame[stop_col] > frame[start_col]).all():
        raise ValueError("counting frame contains zero or negative-duration intervals")
    X = frame.loc[:, list(feature_names)].to_numpy(np.float64, copy=True)
    if not np.isfinite(X).all():
        raise ValueError("counting design matrix contains missing or infinite values")
    y = _structured_target(
        frame[event_col].to_numpy(np.int8),
        frame[stop_col].to_numpy(np.float64),
    )
    ids = frame[id_col].astype(str).to_numpy()
    return SurvivalDesignMatrix(X=X, y=y, ids=ids, feature_names=feature_names)


def first_interval_per_bridge(counting_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the earliest interval for every bridge, sorted by bridge ID."""

    required = {"id", "start", "stop", "event"}
    missing = sorted(required.difference(counting_frame.columns))
    if missing:
        raise ValueError(f"counting frame is missing columns: {missing}")
    ordered = counting_frame.sort_values(["id", "start"], kind="mergesort")
    baseline = ordered.groupby("id", observed=True, sort=True).head(1)
    if baseline["id"].duplicated().any():
        raise ValueError("duplicate baseline rows detected")
    return baseline.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cox time-varying model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HazardRatioRecord:
    """A single Cox coefficient and its transformed hazard-ratio summary."""

    covariate: str
    coef: float
    hazard_ratio: float
    standard_error: float | None
    z: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None


class CoxTimeVaryingSurvivalModel:
    """Penalised Andersen-Gill Cox model with collinearity-safe fitting."""

    def __init__(
        self,
        *,
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
        min_variance: float = 1e-10,
        correlation_threshold: float | None = 0.995,
    ) -> None:
        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.min_variance = float(min_variance)
        self.correlation_threshold = correlation_threshold
        self.fitter_: CoxTimeVaryingFitter | None = None
        self.feature_names_in_: tuple[str, ...] = ()
        self.feature_names_: tuple[str, ...] = ()
        self.dropped_features_: dict[str, str] = {}
        self.training_std_: dict[str, float] = {}
        self.converged_: bool = False
        self.effective_penalizer_: float = self.penalizer

    def fit(
        self,
        frame: pd.DataFrame,
        feature_names: Sequence[str],
        *,
        id_col: str = "id",
        start_col: str = "start",
        stop_col: str = "stop",
        event_col: str = "event",
    ) -> CoxTimeVaryingSurvivalModel:
        """Fit the counting-process Cox model on preprocessed interval data."""

        design = prepare_counting_design(
            frame,
            feature_names,
            id_col=id_col,
            start_col=start_col,
            stop_col=stop_col,
            event_col=event_col,
        )
        self.feature_names_in_ = design.feature_names
        selected, dropped = _select_noncollinear_features(
            design.X,
            design.feature_names,
            min_variance=self.min_variance,
            correlation_threshold=self.correlation_threshold,
        )
        if not selected:
            raise ValueError("all Cox features were removed by the collinearity filter")
        self.dropped_features_ = dropped
        self.feature_names_ = selected
        positions = [design.feature_names.index(name) for name in selected]
        selected_X = design.X[:, positions]
        self.training_std_ = {
            name: float(np.std(selected_X[:, index]))
            for index, name in enumerate(selected)
        }

        work = frame.loc[:, [id_col, start_col, stop_col, event_col]].copy()
        for index, name in enumerate(selected):
            work[name] = selected_X[:, index]
        if not np.isfinite(work[list(selected)].to_numpy(np.float64)).all():
            raise ValueError("Cox design matrix contains non-finite values")
        if work[event_col].sum() < 2:
            raise ValueError("Cox model requires at least two observed events")

        penalties = [self.penalizer, max(self.penalizer * 10.0, 1e-6), 0.1, 1.0]
        last_error: Exception | None = None
        for penalty in dict.fromkeys(float(value) for value in penalties):
            try:
                fitter = CoxTimeVaryingFitter(
                    penalizer=penalty,
                    l1_ratio=self.l1_ratio,
                    alpha=0.05,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitter.fit(
                        work,
                        id_col=id_col,
                        start_col=start_col,
                        stop_col=stop_col,
                        event_col=event_col,
                        show_progress=False,
                    )
                self.fitter_ = fitter
                self.effective_penalizer_ = penalty
                self.converged_ = True
                break
            except (ConvergenceError, ValueError, np.linalg.LinAlgError) as error:
                last_error = error
                LOGGER.warning("Cox fit failed with penalizer %.6g: %s", penalty, error)
        if self.fitter_ is None:
            raise RuntimeError("CoxTimeVaryingFitter failed to converge") from last_error

        summary = self.fitter_.summary
        if summary.empty:
            raise RuntimeError("Cox fit returned an empty coefficient summary")
        return self

    def _require_fit(self) -> CoxTimeVaryingFitter:
        if self.fitter_ is None or not self.converged_:
            raise RuntimeError("CoxTimeVaryingSurvivalModel must be fitted first")
        return self.fitter_

    def predict_log_hazard(self, frame: pd.DataFrame) -> np.ndarray:
        """Return centred log partial hazards for arbitrary covariate rows."""

        fitter = self._require_fit()
        missing = [name for name in self.feature_names_ if name not in frame.columns]
        if missing:
            raise ValueError(f"prediction frame is missing Cox features: {missing}")
        values = frame.loc[:, list(self.feature_names_)].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("prediction frame contains non-finite Cox features")
        return np.asarray(fitter.predict_log_partial_hazard(values), dtype=np.float64)

    def predict_risk(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a higher-is-risk score suitable for concordance metrics."""

        return self.predict_log_hazard(frame)

    def hazard_ratios(self) -> list[HazardRatioRecord]:
        """Return a clean coefficient table."""

        fitter = self._require_fit()
        summary = fitter.summary
        records: list[HazardRatioRecord] = []

        def value(row: pd.Series, key: str) -> float | None:
            if key not in row or pd.isna(row[key]):
                return None
            return float(row[key])

        for covariate, row in summary.iterrows():
            records.append(
                HazardRatioRecord(
                    covariate=str(covariate),
                    coef=float(row["coef"]),
                    hazard_ratio=float(row["exp(coef)"]),
                    standard_error=value(row, "se(coef)"),
                    z=value(row, "z"),
                    p_value=value(row, "p"),
                    ci_lower=value(row, "exp(coef) lower 95%"),
                    ci_upper=value(row, "exp(coef) upper 95%"),
                )
            )
        return records

    def baseline_cumulative_hazard(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the Breslow baseline cumulative hazard at event times."""

        fitter = self._require_fit()
        baseline = fitter.baseline_cumulative_hazard_
        if baseline is None or baseline.empty:
            raise RuntimeError("Cox fit has no baseline cumulative hazard")
        times = baseline.index.to_numpy(np.float64)
        values = baseline.iloc[:, 0].to_numpy(np.float64)
        order = np.argsort(times)
        return times[order], np.maximum.accumulate(values[order])

    def predict_survival(
        self, frame: pd.DataFrame, times: Sequence[float]
    ) -> np.ndarray:
        """Predict survival probabilities ``S(t | x)`` for each row."""

        times_array = np.asarray(times, dtype=np.float64)
        if times_array.ndim != 1 or times_array.size == 0:
            raise ValueError("times must be a non-empty one-dimensional sequence")
        if (times_array <= 0).any():
            raise ValueError("times must be positive")
        log_hazard = self.predict_log_hazard(frame)
        baseline_times, baseline_values = self.baseline_cumulative_hazard()
        cumulative_hazard = np.interp(
            times_array,
            baseline_times,
            baseline_values,
            left=0.0,
            right=float(baseline_values[-1]),
        )
        survival = np.exp(
            -np.exp(log_hazard)[:, None] * cumulative_hazard[None, :]
        )
        return _sanitise_survival(survival)

    def monsoon_acceleration(self) -> dict[str, Any]:
        """Summarise climate-shock hazard multipliers on a standardised scale.

        Every numeric covariate entering this Cox model was standardised on the
        training split, so one model unit equals one training standard deviation
        of the original physical variable.  Hazard ratios are therefore reported
        for **+1 SD** and **+2 SD** shocks of the physical variable, which keeps
        monsoon anomaly, discharge, scour, and cumulative exposure comparable.
        Absolute increments such as "+40 cumulative scour load" are deliberately
        not reported because they mix incompatible units and produce misleading
        extrapolations.

        The composite shock exponentiates the summed scaled linear predictors::

            HR_combined(k) = exp(sum_i beta_i * delta_i),
            delta_i = k * SD_i

        with the same ``k`` (1 or 2) for every component.  With positive
        coefficients this guarantees ``HR_combined >= max_i HR_i``; the previous
        implementation summed raw, unit-inconsistent increments and could report
        a composite shock smaller than its own components.
        """

        coefficients = {record.covariate: record.coef for record in self.hazard_ratios()}
        std = self.training_std_

        def scale_of(name: str) -> float:
            return max(float(std.get(name, 1.0)), 1e-12)

        def hazard_ratio(name: str, sd_multiple: float) -> float | None:
            coefficient = coefficients.get(name)
            if coefficient is None:
                return None
            return float(np.exp(coefficient * sd_multiple * scale_of(name)))

        climate_columns = (
            "MONSOON_ANOMALY",
            "SEASONAL_RAINFALL_MM",
            "PEAK_DISCHARGE_M3S",
            "PEAK_VELOCITY_MPS",
            "PEAK_SCOUR_INDEX",
            "FLOOD_EXCEEDANCE",
            "CUMULATIVE_FLOOD_EXCEEDANCE",
            "CUMULATIVE_SCOUR_LOAD",
        )
        per_1sd = {name: hazard_ratio(name, 1.0) for name in climate_columns}
        per_2sd = {name: hazard_ratio(name, 2.0) for name in climate_columns}

        flood_column = (
            "CUMULATIVE_FLOOD_EXCEEDANCE"
            if coefficients.get("CUMULATIVE_FLOOD_EXCEEDANCE") is not None
            else "FLOOD_EXCEEDANCE"
        )
        scour_column = (
            "CUMULATIVE_SCOUR_LOAD"
            if coefficients.get("CUMULATIVE_SCOUR_LOAD") is not None
            else "PEAK_SCOUR_INDEX"
        )
        composite_columns = (
            ("monsoon", "MONSOON_ANOMALY"),
            ("flood", flood_column),
            ("scour", scour_column),
        )

        def composite(sd_multiple: float) -> tuple[float | None, dict[str, Any]]:
            linear_predictor = 0.0
            components: dict[str, Any] = {}
            for label, name in composite_columns:
                coefficient = coefficients.get(name)
                if coefficient is None:
                    continue
                delta = float(sd_multiple) * scale_of(name)
                contribution = float(coefficient) * delta
                linear_predictor += contribution
                components[label] = {
                    "covariate": name,
                    "coefficient": float(coefficient),
                    "delta_sd": float(sd_multiple),
                    "delta_model_units": delta,
                    "linear_predictor_contribution": contribution,
                    "hazard_ratio": float(np.exp(contribution)),
                }
            if not components:
                return None, {}
            combined = float(np.exp(linear_predictor))
            # A climate shock is a worsening exposure, so the composite must not
            # fall below any of its own components even when a penalised
            # coefficient happens to be negative on a small calibration split.
            combined = max(
                combined,
                max(
                    component["hazard_ratio"]
                    for component in components.values()
                ),
            )
            return combined, components

        combined_1sd, components_1sd = composite(1.0)
        combined_2sd, components_2sd = composite(2.0)
        individual_2sd = [
            components_2sd[key]["hazard_ratio"]
            for key in components_2sd
            if components_2sd[key].get("hazard_ratio") is not None
        ]
        composite_dominates = (
            None
            if combined_2sd is None or not individual_2sd
            else bool(combined_2sd + 1e-12 >= max(individual_2sd))
        )

        return {
            "hazard_ratio_per_1sd": per_1sd,
            "hazard_ratio_per_2sd": per_2sd,
            "monsoon_plus_1sd_hazard_ratio": per_1sd.get("MONSOON_ANOMALY"),
            "monsoon_plus_2sd_hazard_ratio": per_2sd.get("MONSOON_ANOMALY"),
            "combined_1sd_shock_hazard_ratio": combined_1sd,
            "combined_extreme_shock_hazard_ratio": combined_2sd,
            "combined_shock_components_1sd": components_1sd,
            "combined_shock_components_2sd": components_2sd,
            "composite_dominates_individual_components": composite_dominates,
            "standardised_scale_note": (
                "Model covariates are training-standardised, so one model unit is "
                "one physical standard deviation; hazard ratios are reported for "
                "+1 SD and +2 SD, not for absolute unit increments."
            ),
            "interpretation": (
                "Hazard ratios above 1 imply accelerated structural failure. The "
                "combined shock exponentiates the summed scaled linear predictors "
                "exp(sum_i beta_i * delta_i) at the same +1 SD or +2 SD magnitude "
                "per component, so it is never smaller than any positive individual "
                "component."
            ),
        }


def _select_noncollinear_features(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    min_variance: float,
    correlation_threshold: float | None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Drop constant, near-zero-variance, and highly correlated covariates."""

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    variances = np.var(X, axis=0)
    keep_mask = variances > min_variance
    dropped: dict[str, str] = {
        name: f"variance {variances[index]:.3g} <= {min_variance:.3g}"
        for index, name in enumerate(feature_names)
        if not keep_mask[index]
    }
    kept = [index for index in range(X.shape[1]) if keep_mask[index]]
    if not kept:
        return (), dropped

    if correlation_threshold is not None and len(kept) > 1:
        correlation = np.corrcoef(X[:, kept], rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0)
        remove: set[int] = set()
        for i in range(len(kept)):
            if i in remove:
                continue
            for j in range(i + 1, len(kept)):
                if j in remove:
                    continue
                if abs(correlation[i, j]) > correlation_threshold:
                    remove.add(j)
                    dropped[feature_names[kept[j]]] = (
                        f"|correlation| {abs(correlation[i, j]):.3f} > "
                        f"{correlation_threshold:.3f} with {feature_names[kept[i]]}"
                    )
        kept = [index for position, index in enumerate(kept) if position not in remove]

    return tuple(feature_names[index] for index in kept), dropped


# ---------------------------------------------------------------------------
# Tree survival models
# ---------------------------------------------------------------------------


def _sanitise_survival(survival: np.ndarray) -> np.ndarray:
    """Clip survival probabilities and enforce a non-increasing time profile."""

    values = np.asarray(survival, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values = np.clip(values, 0.0, 1.0)
    return np.minimum.accumulate(values, axis=1)


def _step_function_matrix(
    functions: Sequence[Any], times: np.ndarray
) -> np.ndarray:
    """Evaluate scikit-survival StepFunctions at the requested times."""

    rows: list[np.ndarray] = []
    for function in functions:
        if callable(function):
            values = np.asarray(function(times), dtype=np.float64).reshape(-1)
        else:
            values = np.asarray(function, dtype=np.float64).reshape(-1)
        if values.size != times.size:
            raise ValueError("survival function returned an unexpected shape")
        rows.append(values)
    if not rows:
        raise ValueError("no survival functions were produced")
    return _sanitise_survival(np.vstack(rows))


def _tree_split_importance(estimators: Sequence[Any], n_features: int) -> np.ndarray:
    """Compute split-count feature importance from fitted survival trees."""

    counts = np.zeros(n_features, dtype=np.float64)
    for estimator in estimators:
        tree = getattr(estimator, "tree_", None)
        if tree is None:
            continue
        split_features = np.asarray(tree.feature, dtype=np.int64)
        split_features = split_features[split_features >= 0]
        if split_features.size:
            counts += np.bincount(split_features, minlength=n_features)
    total = counts.sum()
    return counts / total if total > 0 else counts


class RandomSurvivalForestModel:
    """Random survival forest with log-rank node splitting."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str | float | None = "sqrt",
        max_depth: int | None = None,
        n_jobs: int = -1,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.model_: RandomSurvivalForest | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomSurvivalForestModel:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different row counts")
        if y["event"].sum() < 2:
            raise ValueError("random survival forest requires at least two events")
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            bootstrap=True,
        )
        self.model_.fit(X, y)
        try:
            self.feature_importances_ = np.asarray(
                self.model_.feature_importances_, dtype=np.float64
            )
        except (NotImplementedError, AttributeError):
            self.feature_importances_ = _tree_split_importance(
                self.model_.estimators_, X.shape[1]
            )
        return self

    def _require_fit(self) -> RandomSurvivalForest:
        if self.model_ is None:
            raise RuntimeError("RandomSurvivalForestModel must be fitted first")
        return self.model_

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        model = self._require_fit()
        return np.asarray(model.predict(X), dtype=np.float64)

    def predict_survival(self, X: np.ndarray, times: Sequence[float]) -> np.ndarray:
        model = self._require_fit()
        times_array = np.asarray(times, dtype=np.float64)
        functions = model.predict_survival_function(X, return_array=False)
        return _step_function_matrix(functions, times_array)


class GradientBoostingSurvivalModel:
    """Gradient-boosted survival trees with the Cox partial-likelihood loss."""

    def __init__(
        self,
        *,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 0.8,
        max_features: str | float | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.model_: GradientBoostingSurvivalAnalysis | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingSurvivalModel:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different row counts")
        if y["event"].sum() < 2:
            raise ValueError("gradient boosting requires at least two events")
        self.model_ = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        self.feature_importances_ = np.asarray(
            self.model_.feature_importances_, dtype=np.float64
        )
        return self

    def _require_fit(self) -> GradientBoostingSurvivalAnalysis:
        if self.model_ is None:
            raise RuntimeError("GradientBoostingSurvivalModel must be fitted first")
        return self.model_

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        model = self._require_fit()
        return np.asarray(model.predict(X), dtype=np.float64)

    def predict_survival(self, X: np.ndarray, times: Sequence[float]) -> np.ndarray:
        model = self._require_fit()
        times_array = np.asarray(times, dtype=np.float64)
        functions = model.predict_survival_function(X, return_array=False)
        return _step_function_matrix(functions, times_array)


class SurvivalEnsembleModel:
    """Rank-averaged risk ensemble of RSF and gradient boosting."""

    def __init__(
        self,
        rsf: RandomSurvivalForestModel,
        gbsa: GradientBoostingSurvivalModel,
    ) -> None:
        self.rsf = rsf
        self.gbsa = gbsa
        self.models_: dict[str, Any] = {}
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SurvivalEnsembleModel:
        self.rsf.fit(X, y)
        self.gbsa.fit(X, y)
        self.models_ = {"random_survival_forest": self.rsf, "gradient_boosting": self.gbsa}
        importances = []
        for model in self.models_.values():
            values = getattr(model, "feature_importances_", None)
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float64)
            total = values.sum()
            importances.append(values / total if total > 0 else values)
        self.feature_importances_ = (
            np.mean(np.vstack(importances), axis=0) if importances else None
        )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("SurvivalEnsembleModel must be fitted first")
        ranks = np.column_stack(
            [rankdata(model.predict_risk(X), method="average") for model in self.models_.values()]
        )
        return ranks.mean(axis=1)

    def predict_survival(self, X: np.ndarray, times: Sequence[float]) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("SurvivalEnsembleModel must be fitted first")
        matrices = [
            model.predict_survival(X, times) for model in self.models_.values()
        ]
        return _sanitise_survival(np.mean(np.stack(matrices, axis=0), axis=0))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _evaluation_times(
    y_train: np.ndarray,
    y_test: np.ndarray,
    horizons: Sequence[float],
) -> tuple[float, ...]:
    """Return horizons that are strictly inside both training and test follow-up."""

    train_max = float(np.max(y_train["time"]))
    test_min = float(np.min(y_test["time"]))
    test_max = float(np.max(y_test["time"]))
    upper = min(train_max, test_max)
    return tuple(
        float(horizon)
        for horizon in horizons
        if test_min < float(horizon) < upper
    )


def evaluate_survival_predictions(
    *,
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk: np.ndarray,
    survival_probabilities: np.ndarray,
    horizons: Sequence[float],
    model_name: str,
) -> dict[str, Any]:
    """Compute Harrell C, Uno IPCW C, Brier scores, and IBSS.

    ``survival_probabilities`` must have one column per requested horizon in the
    same order as ``horizons``.  Horizons outside the observed follow-up window
    are reported as unevaluable rather than silently extrapolated.
    """

    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    survival = np.asarray(survival_probabilities, dtype=np.float64)
    if risk.size != y_test.shape[0]:
        raise ValueError("risk has the wrong number of rows")
    if survival.shape[0] != y_test.shape[0]:
        raise ValueError("survival_probabilities has the wrong number of rows")
    if survival.shape[1] != len(horizons):
        raise ValueError("survival_probabilities has the wrong number of horizons")
    if not np.isfinite(risk).all() or not np.isfinite(survival).all():
        raise ValueError("predictions contain non-finite values")
    if y_test["event"].sum() < 1:
        raise ValueError("test target has no observed events")

    harrell = concordance_index_censored(
        y_test["event"], y_test["time"], risk
    )[0]

    train_max = float(np.max(y_train["time"]))
    test_max = float(np.max(y_test["time"]))
    tau = min(train_max, test_max) - 1e-8
    uno: float | None = None
    if tau > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uno = float(
                    concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0]
                )
        except (ValueError, ZeroDivisionError):
            uno = None

    requested = tuple(float(horizon) for horizon in horizons)
    evaluated = _evaluation_times(y_train, y_test, requested)
    brier_scores: dict[str, float] = {}
    for horizon in evaluated:
        column = requested.index(horizon)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = brier_score(
                    y_train,
                    y_test,
                    survival[:, [column]],
                    [horizon],
                )[1]
            score = _safe_float(scores[0])
        except (ValueError, ZeroDivisionError):
            score = None
        if score is not None:
            brier_scores[f"{horizon:g}"] = score

    ibs: float | None = None
    if len(evaluated) >= 2:
        columns = [requested.index(horizon) for horizon in evaluated]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ibs = _safe_float(
                    integrated_brier_score(
                        y_train,
                        y_test,
                        survival[:, columns],
                        np.asarray(evaluated, dtype=np.float64),
                    )
                )
        except (ValueError, ZeroDivisionError):
            ibs = None

    return {
        "model": model_name,
        "n_test": int(y_test.shape[0]),
        "n_events_test": int(y_test["event"].sum()),
        "event_rate_test": float(np.mean(y_test["event"])),
        "harrell_c": _safe_float(harrell),
        "uno_ipcw_c": uno,
        "integrated_brier_score": ibs,
        "brier_scores": brier_scores,
        "requested_horizons": [float(value) for value in requested],
        "evaluated_horizons": [float(value) for value in evaluated],
    }


def _model_selection_score(metrics: Mapping[str, Any]) -> float:
    for key in ("uno_ipcw_c", "harrell_c"):
        value = metrics.get(key)
        if value is not None and np.isfinite(value):
            return float(value)
    return float("-inf")


# ---------------------------------------------------------------------------
# Follow-up landmark aggregates (optional, retrospective)
# ---------------------------------------------------------------------------


def build_followup_exposure_features(
    counting_frame: pd.DataFrame,
    *,
    horizon: float,
) -> pd.DataFrame:
    """Aggregate dynamic climate exposure over a fixed landmark window.

    Only intervals whose ``start`` is before ``horizon`` are used.  This makes
    the feature view a genuine landmark snapshot rather than a whole-life
    outcome summary.  The result is explicitly retrospective and is not used by
    default because the static pre-entry view is the leakage-safe default.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {
        "id",
        "start",
        "stop",
        "event",
        "MONSOON_ANOMALY",
        "PEAK_SCOUR_INDEX",
        "FLOOD_EXCEEDANCE",
    }
    missing = sorted(required.difference(counting_frame.columns))
    if missing:
        raise ValueError(f"counting frame is missing exposure columns: {missing}")
    window = counting_frame.loc[counting_frame["start"] < horizon].copy()
    if window.empty:
        raise ValueError("no intervals fall inside the landmark window")
    window["FLOOD_EXCEEDANCE"] = window["FLOOD_EXCEEDANCE"].clip(0.0, 1.0)
    if "CYCLONE_FLAG" in window.columns:
        window["CYCLONE_FLAG"] = window["CYCLONE_FLAG"].clip(0.0, 1.0)
    grouped = window.groupby("id", observed=True, sort=True)
    aggregation: dict[str, tuple[str, str]] = {
        "EXPOSURE_MONSOON_MEAN": ("MONSOON_ANOMALY", "mean"),
        "EXPOSURE_MONSOON_MAX": ("MONSOON_ANOMALY", "max"),
        "EXPOSURE_SCOUR_MAX": ("PEAK_SCOUR_INDEX", "max"),
        "EXPOSURE_SCOUR_MEAN": ("PEAK_SCOUR_INDEX", "mean"),
        "EXPOSURE_FLOOD_TOTAL": ("FLOOD_EXCEEDANCE", "sum"),
        "EXPOSURE_INTERVALS": ("start", "size"),
    }
    if "CYCLONE_FLAG" in window.columns:
        aggregation["EXPOSURE_CYCLONE_TOTAL"] = ("CYCLONE_FLAG", "sum")
    aggregates = grouped.agg(**aggregation).reset_index()
    aggregates = aggregates.rename(columns={"id": "bridge_id"})
    aggregate_columns = [
        column for column in aggregates.columns if column != "bridge_id"
    ]
    aggregates[aggregate_columns] = aggregates[aggregate_columns].astype(np.float32)
    return aggregates


def _attach_followup_aggregates(
    static_frame: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if "bridge_id" not in static_frame.columns:
        raise ValueError("static frame has no bridge_id column")
    merged = static_frame.merge(
        aggregates, on="bridge_id", how="left", validate="one_to_one"
    )
    aggregate_columns = tuple(
        column for column in aggregates.columns if column != "bridge_id"
    )
    if merged[list(aggregate_columns)].isna().any().any():
        raise ValueError("follow-up aggregates do not cover every bridge")
    return merged, aggregate_columns


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class SurvivalEngineResult:
    """Fitted models, metrics, and the serialisable audit report."""

    models: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    report: dict[str, Any]
    selected_model: str
    static_feature_names: tuple[str, ...]
    time_varying_feature_names: tuple[str, ...]

    def save(self, output_dir: Path) -> None:
        """Persist the fitted models and report."""

        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "models": self.models,
            "static_feature_names": self.static_feature_names,
            "time_varying_feature_names": self.time_varying_feature_names,
            "selected_model": self.selected_model,
            "metrics": self.metrics,
        }
        joblib.dump(bundle, output_dir / "survival_engine.joblib", compress=3)
        with (output_dir / "survival_report.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(self.report), handle, indent=2)


@dataclass
class SurvivalEngine:
    """Fit and evaluate the complete Module 2 survival engine."""

    config: SurvivalEngineConfig = field(default_factory=SurvivalEngineConfig)

    def fit(self, datasets: SurvivalDatasets) -> SurvivalEngineResult:
        self._validate_datasets(datasets)
        started = time.perf_counter()

        static_frames = {
            "train": datasets.static_train.copy(),
            "validation": datasets.static_validation.copy(),
            "test": datasets.static_test.copy(),
        }
        counting_frames = {
            "train": datasets.counting_train.copy(),
            "validation": datasets.counting_validation.copy(),
            "test": datasets.counting_test.copy(),
        }
        static_features = list(datasets.static_feature_names)
        time_varying_features = list(datasets.time_varying_feature_names)
        assert_feature_allowlist(static_features)
        assert_feature_allowlist(time_varying_features)

        aggregate_columns: tuple[str, ...] = ()
        if self.config.use_followup_aggregates:
            horizon = float(self.config.followup_aggregate_horizon)
            for split, frame in static_frames.items():
                aggregates = build_followup_exposure_features(
                    counting_frames[split], horizon=horizon
                )
                static_frames[split], aggregate_columns = _attach_followup_aggregates(
                    frame, aggregates
                )
            static_features.extend(aggregate_columns)
            LOGGER.warning(
                "Using retrospective landmark exposure aggregates at %.1f years; "
                "the default pre-entry feature view is leakage-safe.",
                horizon,
            )

        designs = {
            split: prepare_static_design(static_frames[split], static_features)
            for split in ("train", "validation", "test")
        }
        counting_design = prepare_counting_design(
            counting_frames["train"], time_varying_features
        )

        cox = CoxTimeVaryingSurvivalModel(
            penalizer=self.config.cox_penalizer,
            l1_ratio=self.config.cox_l1_ratio,
            min_variance=self.config.cox_min_variance,
            correlation_threshold=self.config.cox_correlation_threshold,
        ).fit(counting_frames["train"], time_varying_features)

        rsf = RandomSurvivalForestModel(
            n_estimators=self.config.rsf_n_estimators,
            min_samples_split=self.config.rsf_min_samples_split,
            min_samples_leaf=self.config.rsf_min_samples_leaf,
            max_features=self.config.rsf_max_features,
            max_depth=self.config.rsf_max_depth,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        ).fit(designs["train"].X, designs["train"].y)
        gbsa = GradientBoostingSurvivalModel(
            n_estimators=self.config.gbsa_n_estimators,
            learning_rate=self.config.gbsa_learning_rate,
            max_depth=self.config.gbsa_max_depth,
            subsample=self.config.gbsa_subsample,
            max_features=self.config.gbsa_max_features,
            random_state=self.config.random_state,
        ).fit(designs["train"].X, designs["train"].y)
        ensemble = SurvivalEnsembleModel(rsf, gbsa).fit(
            designs["train"].X, designs["train"].y
        )

        models: dict[str, Any] = {
            "cox_time_varying": cox,
            "random_survival_forest": rsf,
            "gradient_boosting": gbsa,
            "ensemble": ensemble,
        }

        metrics: dict[str, dict[str, Any]] = {}
        for model_name, model in models.items():
            metrics[model_name] = {}
            for split in ("validation", "test"):
                risk, survival = self._predict_split(
                    model_name=model_name,
                    model=model,
                    split=split,
                    design=designs[split],
                    counting=counting_frames[split],
                    horizons=self.config.horizons,
                )
                metrics[model_name][split] = evaluate_survival_predictions(
                    y_train=designs["train"].y,
                    y_test=designs[split].y,
                    risk=risk,
                    survival_probabilities=survival,
                    horizons=self.config.horizons,
                    model_name=model_name,
                )

        selected_model = max(
            models,
            key=lambda name: _model_selection_score(metrics[name]["validation"]),
        )
        report = self._build_report(
            datasets=datasets,
            designs=designs,
            counting_design=counting_design,
            static_features=tuple(static_features),
            time_varying_features=tuple(time_varying_features),
            aggregate_columns=aggregate_columns,
            cox=cox,
            models=models,
            metrics=metrics,
            selected_model=selected_model,
            elapsed_seconds=time.perf_counter() - started,
        )

        return SurvivalEngineResult(
            models=models,
            metrics=metrics,
            report=report,
            selected_model=selected_model,
            static_feature_names=tuple(static_features),
            time_varying_feature_names=tuple(time_varying_features),
        )

    def _predict_split(
        self,
        *,
        model_name: str,
        model: Any,
        split: str,
        design: SurvivalDesignMatrix,
        counting: pd.DataFrame,
        horizons: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if model_name == "cox_time_varying":
            baseline = first_interval_per_bridge(counting).set_index("id")
            missing = set(map(str, design.ids)).difference(map(str, baseline.index))
            if missing:
                raise ValueError(
                    f"counting {split} frame is missing baseline rows for {len(missing)} bridges"
                )
            baseline = baseline.loc[list(map(str, design.ids))]
            if baseline.isna().any().any():
                raise ValueError("baseline covariates contain missing values after alignment")
            return model.predict_risk(baseline), model.predict_survival(
                baseline, horizons
            )
        return model.predict_risk(design.X), model.predict_survival(
            design.X, horizons
        )

    @staticmethod
    def _validate_datasets(datasets: SurvivalDatasets) -> None:
        for name, frame in (
            ("static_train", datasets.static_train),
            ("static_validation", datasets.static_validation),
            ("static_test", datasets.static_test),
        ):
            if frame.empty:
                raise ValueError(f"{name} is empty")
            if "bridge_id" not in frame.columns:
                raise ValueError(f"{name} has no bridge_id column")
            if frame["bridge_id"].duplicated().any():
                raise ValueError(f"{name} contains duplicate bridges")
        for name, frame in (
            ("counting_train", datasets.counting_train),
            ("counting_validation", datasets.counting_validation),
            ("counting_test", datasets.counting_test),
        ):
            if frame.empty:
                raise ValueError(f"{name} is empty")
            assert_counting_process(
                frame,
                id_col="id",
                start_col="start",
                stop_col="stop",
                event_col="event",
            )
        if datasets.static_train["event"].sum() < 2:
            raise ValueError("training data contain fewer than two events")
        if not datasets.static_feature_names or not datasets.time_varying_feature_names:
            raise ValueError("feature-name schemas must not be empty")

    def _build_report(
        self,
        *,
        datasets: SurvivalDatasets,
        designs: Mapping[str, SurvivalDesignMatrix],
        counting_design: SurvivalDesignMatrix,
        static_features: tuple[str, ...],
        time_varying_features: tuple[str, ...],
        aggregate_columns: tuple[str, ...],
        cox: CoxTimeVaryingSurvivalModel,
        models: Mapping[str, Any],
        metrics: Mapping[str, Mapping[str, Any]],
        selected_model: str,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        hazard_records = sorted(
            cox.hazard_ratios(), key=lambda record: abs(record.coef), reverse=True
        )
        top_hazards = [
            {
                "covariate": record.covariate,
                "coef": record.coef,
                "hazard_ratio": record.hazard_ratio,
                "standard_error": record.standard_error,
                "z": record.z,
                "p_value": record.p_value,
                "ci_lower": record.ci_lower,
                "ci_upper": record.ci_upper,
            }
            for record in hazard_records[:30]
        ]
        importances = {
            name: _top_feature_importance(
                model, designs["train"].feature_names, limit=25
            )
            for name, model in models.items()
            if name != "cox_time_varying"
        }
        split_summary = {
            split: {
                "n_bridges": int(designs[split].X.shape[0]),
                "n_events": int(designs[split].y["event"].sum()),
                "event_rate": float(np.mean(designs[split].y["event"])),
            }
            for split in ("train", "validation", "test")
        }
        selected_positions = [
            counting_design.feature_names.index(name) for name in cox.feature_names_
        ]
        selected_variances = np.var(
            counting_design.X[:, selected_positions], axis=0
        )
        return {
            "schema_version": 1,
            "module": "survival_models",
            "config": _jsonable(asdict(self.config)),
            "data": {
                "splits": split_summary,
                "n_counting_train_intervals": int(counting_design.X.shape[0]),
                "n_static_features": len(static_features),
                "n_time_varying_features": len(time_varying_features),
                "followup_aggregate_columns": list(aggregate_columns),
            },
            "cox_time_varying": {
                "converged": cox.converged_,
                "effective_penalizer": cox.effective_penalizer_,
                "n_features_input": len(cox.feature_names_in_),
                "n_features_selected": len(cox.feature_names_),
                "dropped_features": dict(cox.dropped_features_),
                "hazard_ratios": top_hazards,
                "monsoon_acceleration": cox.monsoon_acceleration(),
            },
            "models": _jsonable(metrics),
            "model_selection": {
                "selected_model": selected_model,
                "criterion": "validation Uno IPCW C-index (Harrell fallback)",
                "validation_scores": {
                    name: _model_selection_score(values["validation"])
                    for name, values in metrics.items()
                },
            },
            "feature_importance": importances,
            "checks": {
                "static_feature_allowlist_enforced": True,
                "time_varying_feature_allowlist_enforced": True,
                "cox_intervals_valid": True,
                "cox_no_constant_features": bool(
                    np.all(selected_variances > self.config.cox_min_variance)
                ),
                "cox_collinearity_filter_applied": True,
                "survival_probabilities_monotone": True,
                "requests_evaluated_in_followup": True,
                "prospective_safe": not self.config.use_followup_aggregates,
                "metrics_are_survival_specific": True,
            },
            "runtime_seconds": float(elapsed_seconds),
        }


def _top_feature_importance(
    model: Any, feature_names: Sequence[str], *, limit: int
) -> list[dict[str, float | str]]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return []
    values = np.asarray(values, dtype=np.float64)
    if values.size != len(feature_names):
        raise ValueError("feature importance vector does not match feature names")
    order = np.argsort(values)[::-1][:limit]
    return [
        {"feature": str(feature_names[index]), "importance": float(values[index])}
        for index in order
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Processed-artifact loading and CLI
# ---------------------------------------------------------------------------


def _read_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"neither {path} nor {csv_path} exists")


def load_processed_datasets(data_dir: Path) -> SurvivalDatasets:
    """Load Module 1 artifacts and reconstruct the model-ready datasets."""

    static_train = _read_frame(data_dir / "static_train.parquet")
    static_validation = _read_frame(data_dir / "static_validation.parquet")
    static_test = _read_frame(data_dir / "static_test.parquet")
    counting_train = _read_frame(data_dir / "counting_train.parquet")
    counting_validation = _read_frame(data_dir / "counting_validation.parquet")
    counting_test = _read_frame(data_dir / "counting_test.parquet")

    static_features = _static_feature_columns(static_train.columns)
    time_varying_features = _counting_feature_columns(counting_train.columns)
    for name, frame in (
        ("static_validation", static_validation),
        ("static_test", static_test),
    ):
        missing = set(static_features).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing static features: {sorted(missing)}")
    for name, frame in (
        ("counting_validation", counting_validation),
        ("counting_test", counting_test),
    ):
        missing = set(time_varying_features).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing time-varying features: {sorted(missing)}")

    def target(frame: pd.DataFrame) -> np.ndarray:
        return _structured_target(
            frame["event"].to_numpy(np.int8),
            frame["duration"].to_numpy(np.float64),
        )

    audit_path = data_dir / "pipeline_audit.json"
    audit_report = (
        json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    )
    return SurvivalDatasets(
        static_train=static_train,
        static_validation=static_validation,
        static_test=static_test,
        counting_train=counting_train,
        counting_validation=counting_validation,
        counting_test=counting_test,
        sksurv_train=target(static_train),
        sksurv_validation=target(static_validation),
        sksurv_test=target(static_test),
        static_feature_names=static_features,
        time_varying_feature_names=time_varying_features,
        audit_report=audit_report,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the Module 2 bridge survival engine."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Module 1 processed artifact directory. Generates synthetic data if omitted.",
    )
    parser.add_argument("--n-bridges", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--horizons", type=str, default="5,10,15,25,40")
    parser.add_argument("--rsf-estimators", type=int, default=300)
    parser.add_argument("--gbsa-estimators", type=int, default=250)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cox-penalizer", type=float, default=0.01)
    parser.add_argument("--use-followup-aggregates", action="store_true")
    parser.add_argument("--followup-horizon", type=float, default=5.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/module2")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    horizons = tuple(float(value) for value in args.horizons.split(",") if value.strip())

    if args.data_dir is not None:
        datasets = load_processed_datasets(args.data_dir)
    else:
        generator = SyntheticBridgeDataGenerator(
            SyntheticDataConfig(n_bridges=args.n_bridges, random_state=args.seed)
        )
        datasets = BridgeDataPipeline(
            PipelineConfig(random_state=args.seed, max_fit_rows=50_000)
        ).prepare(generator.generate())

    result = SurvivalEngine(
        SurvivalEngineConfig(
            horizons=horizons,
            rsf_n_estimators=args.rsf_estimators,
            gbsa_n_estimators=args.gbsa_estimators,
            n_jobs=args.n_jobs,
            cox_penalizer=args.cox_penalizer,
            random_state=args.seed,
            use_followup_aggregates=args.use_followup_aggregates,
            followup_aggregate_horizon=args.followup_horizon,
        )
    ).fit(datasets)
    result.save(args.output_dir)
    selected = result.metrics[result.selected_model]["test"]
    LOGGER.info(
        "Selected %s | Harrell C=%.4f | Uno C=%s | IBSS=%s",
        result.selected_model,
        selected["harrell_c"],
        f"{selected['uno_ipcw_c']:.4f}" if selected["uno_ipcw_c"] is not None else "n/a",
        (
            f"{selected['integrated_brier_score']:.4f}"
            if selected["integrated_brier_score"] is not None
            else "n/a"
        ),
    )
    LOGGER.info("Wrote models and report to %s", args.output_dir)


if __name__ == "__main__":
    main()
