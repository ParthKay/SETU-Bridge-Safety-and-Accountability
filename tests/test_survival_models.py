"""Module 2 verification tests: Cox TV, tree ensembles, metrics, and leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bridge_engine.data_pipeline import (
    BridgeDataPipeline,
    PipelineConfig,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
)
from bridge_engine.survival_models import (
    CoxTimeVaryingSurvivalModel,
    SurvivalEngine,
    SurvivalEngineConfig,
    build_followup_exposure_features,
    evaluate_survival_predictions,
    first_interval_per_bridge,
)

N_BRIDGES = 220
SEED = 2027


@pytest.fixture(scope="module")
def datasets():
    generator = SyntheticBridgeDataGenerator(
        SyntheticDataConfig(n_bridges=N_BRIDGES, random_state=SEED)
    )
    pipeline = BridgeDataPipeline(
        PipelineConfig(random_state=SEED, max_fit_rows=20_000)
    )
    return pipeline.prepare(generator.generate())


@pytest.fixture(scope="module")
def engine_result(datasets):
    return SurvivalEngine(
        SurvivalEngineConfig(
            horizons=(5.0, 10.0, 20.0),
            cox_penalizer=0.1,
            rsf_n_estimators=15,
            gbsa_n_estimators=15,
            n_jobs=1,
            random_state=SEED,
        )
    ).fit(datasets)


def test_engine_produces_four_models_and_survival_metrics(engine_result) -> None:
    assert set(engine_result.models) == {
        "cox_time_varying",
        "random_survival_forest",
        "gradient_boosting",
        "ensemble",
    }
    assert engine_result.selected_model in engine_result.models

    for model_name, split_metrics in engine_result.metrics.items():
        assert set(split_metrics) == {"validation", "test"}
        for _split, metrics in split_metrics.items():
            assert metrics["model"] == model_name
            assert 0.0 <= metrics["harrell_c"] <= 1.0
            if metrics["uno_ipcw_c"] is not None:
                assert 0.0 <= metrics["uno_ipcw_c"] <= 1.0
            if metrics["integrated_brier_score"] is not None:
                assert metrics["integrated_brier_score"] >= 0.0
            assert all(value >= 0.0 for value in metrics["brier_scores"].values())
            assert metrics["n_test"] > 0
            assert metrics["n_events_test"] > 0


def test_survival_predictions_are_probabilities_and_monotone(
    datasets, engine_result
) -> None:
    horizons = [5.0, 10.0, 20.0]
    X = datasets.static_test.loc[
        :, list(datasets.static_feature_names)
    ].to_numpy(np.float64)
    for model_name in ("random_survival_forest", "gradient_boosting", "ensemble"):
        model = engine_result.models[model_name]
        survival = model.predict_survival(X, horizons)
        assert survival.shape == (len(X), len(horizons))
        assert np.all(survival >= 0.0) and np.all(survival <= 1.0)
        assert np.all(np.diff(survival, axis=1) <= 1e-12)

    cox = engine_result.models["cox_time_varying"]
    baseline = first_interval_per_bridge(datasets.counting_test)
    cox_survival = cox.predict_survival(baseline, horizons)
    assert cox_survival.shape == (len(baseline), len(horizons))
    assert np.all(cox_survival >= 0.0) and np.all(cox_survival <= 1.0)
    assert np.all(np.diff(cox_survival, axis=1) <= 1e-12)


def test_cox_hazard_ratios_and_monsoon_acceleration(engine_result) -> None:
    report = engine_result.report["cox_time_varying"]
    assert report["converged"] is True
    assert report["n_features_selected"] > 0
    assert report["hazard_ratios"]
    first = report["hazard_ratios"][0]
    assert {"covariate", "coef", "hazard_ratio", "ci_lower", "ci_upper"}.issubset(first)
    assert first["hazard_ratio"] == pytest.approx(np.exp(first["coef"]))

    acceleration = report["monsoon_acceleration"]
    assert "monsoon_plus_2sd_hazard_ratio" in acceleration
    assert "combined_extreme_shock_hazard_ratio" in acceleration
    assert set(acceleration["hazard_ratio_per_1sd"]) >= {
        "MONSOON_ANOMALY",
        "PEAK_SCOUR_INDEX",
    }


def test_report_is_json_serialisable_and_leakage_safe(engine_result) -> None:
    payload = json.dumps(engine_result.report)
    assert len(payload) > 100
    checks = engine_result.report["checks"]
    assert checks["static_feature_allowlist_enforced"] is True
    assert checks["time_varying_feature_allowlist_enforced"] is True
    assert checks["prospective_safe"] is True
    assert engine_result.report["data"]["followup_aggregate_columns"] == []


def test_engine_artifacts_are_persisted(engine_result, tmp_path: Path) -> None:
    engine_result.save(tmp_path)
    assert (tmp_path / "survival_engine.joblib").exists()
    assert (tmp_path / "survival_report.json").exists()
    report = json.loads((tmp_path / "survival_report.json").read_text(encoding="utf-8"))
    assert report["model_selection"]["selected_model"] == engine_result.selected_model


def test_cox_time_varying_model_drops_constant_and_collinear_features(datasets) -> None:
    frame = datasets.counting_train.copy()
    feature_names = list(datasets.time_varying_feature_names)
    frame["CONSTANT_FEATURE"] = 7.0
    frame["DUPLICATE_FEATURE"] = frame[feature_names[0]].to_numpy()
    model = CoxTimeVaryingSurvivalModel(penalizer=0.1)
    model.fit(frame, [*feature_names, "CONSTANT_FEATURE", "DUPLICATE_FEATURE"])

    assert "CONSTANT_FEATURE" in model.dropped_features_
    assert "DUPLICATE_FEATURE" in model.dropped_features_
    assert "CONSTANT_FEATURE" not in model.feature_names_
    assert "DUPLICATE_FEATURE" not in model.feature_names_

    baseline = first_interval_per_bridge(datasets.counting_test)
    survival = model.predict_survival(baseline, [5.0, 10.0, 20.0])
    assert survival.shape == (len(baseline), 3)
    assert np.all(survival >= 0.0) and np.all(survival <= 1.0)
    assert np.all(np.diff(survival, axis=1) <= 1e-12)


def test_followup_aggregates_use_only_landmark_window(datasets) -> None:
    aggregates = build_followup_exposure_features(
        datasets.counting_train, horizon=5.0
    )
    assert aggregates["bridge_id"].is_unique
    assert "event" not in aggregates.columns
    assert "duration" not in aggregates.columns
    assert {
        "EXPOSURE_MONSOON_MEAN",
        "EXPOSURE_SCOUR_MAX",
        "EXPOSURE_FLOOD_TOTAL",
        "EXPOSURE_INTERVALS",
    }.issubset(aggregates.columns)
    assert (aggregates["EXPOSURE_INTERVALS"] <= 5).all()
    assert (aggregates["EXPOSURE_FLOOD_TOTAL"] >= 0.0).all()


def test_followup_aggregates_change_report_safety_flag(datasets) -> None:
    result = SurvivalEngine(
        SurvivalEngineConfig(
            horizons=(5.0, 10.0),
            cox_penalizer=0.1,
            rsf_n_estimators=8,
            gbsa_n_estimators=8,
            n_jobs=1,
            random_state=SEED,
            use_followup_aggregates=True,
            followup_aggregate_horizon=5.0,
        )
    ).fit(datasets)
    assert result.report["checks"]["prospective_safe"] is False
    assert result.report["data"]["followup_aggregate_columns"]


def test_engine_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        SurvivalEngineConfig(horizons=())
    with pytest.raises(ValueError):
        SurvivalEngineConfig(horizons=(10.0, 5.0))
    with pytest.raises(ValueError):
        SurvivalEngineConfig(use_followup_aggregates=True, followup_aggregate_horizon=None)
    with pytest.raises(ValueError):
        SurvivalEngineConfig(rsf_n_estimators=0)


def test_evaluation_rejects_malformed_predictions(datasets) -> None:
    y_train = datasets.sksurv_train
    y_test = datasets.sksurv_test
    with pytest.raises(ValueError):
        evaluate_survival_predictions(
            y_train=y_train,
            y_test=y_test,
            risk=np.zeros(len(y_test) - 1),
            survival_probabilities=np.ones((len(y_test), 2)),
            horizons=(5.0, 10.0),
            model_name="broken",
        )
    with pytest.raises(ValueError):
        evaluate_survival_predictions(
            y_train=y_train,
            y_test=y_test,
            risk=np.zeros(len(y_test)),
            survival_probabilities=np.ones((len(y_test), 3)),
            horizons=(5.0, 10.0),
            model_name="broken",
        )


def test_risk_scores_are_aligned_with_bridge_ids(datasets, engine_result) -> None:
    model = engine_result.models["random_survival_forest"]
    X = datasets.static_test.loc[
        :, list(datasets.static_feature_names)
    ].to_numpy(np.float64)
    risk = model.predict_risk(X)
    assert risk.shape == (len(datasets.static_test),)
    assert np.isfinite(risk).all()
    assert len(np.unique(risk)) > 1
