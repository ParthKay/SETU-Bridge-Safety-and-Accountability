"""Module 3 verification tests: audit rules, leakage guards, and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bridge_engine.audit_verifier import (
    AUDIT_FEATURE_COLUMNS,
    AuditVerifier,
    AuditVerifierConfig,
    _fit_robust_residual,
    assert_audit_feature_allowlist,
    evaluate_audit_against_ground_truth,
    predict_conditional_survival,
)
from bridge_engine.data_pipeline import (
    BridgeDataPipeline,
    PipelineConfig,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
)
from bridge_engine.survival_models import (
    SurvivalEngine,
    SurvivalEngineConfig,
)

N_BRIDGES = 480
SEED = 2028


@pytest.fixture(scope="module")
def dataset():
    generator = SyntheticBridgeDataGenerator(
        SyntheticDataConfig(n_bridges=N_BRIDGES, random_state=SEED)
    )
    return generator.generate()


@pytest.fixture(scope="module")
def datasets(dataset):
    return BridgeDataPipeline(
        PipelineConfig(random_state=SEED, max_fit_rows=20_000)
    ).prepare(dataset)


@pytest.fixture(scope="module")
def engine_result(datasets):
    return SurvivalEngine(
        SurvivalEngineConfig(
            horizons=(5.0, 10.0),
            cox_penalizer=0.1,
            rsf_n_estimators=25,
            gbsa_n_estimators=25,
            n_jobs=1,
            random_state=SEED,
        )
    ).fit(datasets)


@pytest.fixture(scope="module")
def audit_result(datasets, dataset, engine_result):
    return AuditVerifier(
        AuditVerifierConfig(
            model_name="cox_time_varying",
            random_state=SEED,
        )
    ).fit(
        datasets=datasets,
        bridges=dataset.bridges,
        climate=dataset.climate,
        engine_result=engine_result,
        ground_truth=dataset.ground_truth,
    )


def test_config_and_feature_guards() -> None:
    with pytest.raises(ValueError):
        AuditVerifierConfig(horizon_years=0.0)
    with pytest.raises(ValueError):
        AuditVerifierConfig(critical_survival_threshold=0.0)
    with pytest.raises(ValueError):
        AuditVerifierConfig(healthy_condition_min=8, healthy_condition_max=7)
    with pytest.raises(ValueError):
        AuditVerifierConfig(isolation_forest_contamination=0.9)
    with pytest.raises(ValueError):
        AuditVerifierConfig(audit_splits=())
    with pytest.raises(ValueError):
        AuditVerifierConfig(calibration_split="holdout")

    assert_audit_feature_allowlist(AUDIT_FEATURE_COLUMNS)
    for forbidden in (
        "PAPER_INSPECTION_FLAG",
        "TRUE_FAILURE_AGE",
        "LATENT_FRAILTY",
        "INSPECTOR_BIAS",
        "NDT_PERFORMED",
        "REPORTED_DECK_COND",
        "EVENT",
    ):
        with pytest.raises(ValueError):
            assert_audit_feature_allowlist([*AUDIT_FEATURE_COLUMNS, forbidden])


def test_mad_scale_guards_zero_dispersion() -> None:
    calibration = _fit_robust_residual(np.full(12, 3.5), "train")
    assert calibration.residual_center == pytest.approx(3.5)
    assert calibration.residual_scale == pytest.approx(1e-6)
    assert np.isfinite(calibration.residual_scale)


def test_conditional_survival_is_forward_looking(
    datasets, engine_result
) -> None:
    for model_name in (
        "cox_time_varying",
        "random_survival_forest",
        "gradient_boosting",
        "ensemble",
    ):
        physics = predict_conditional_survival(
            datasets=datasets,
            engine_result=engine_result,
            model_name=model_name,
            split="test",
            horizon_years=5.0,
        )
        assert physics["BRIDGE_ID"].is_unique
        assert set(physics["DATA_SPLIT"]) == {"test"}
        assert physics["EXPECTED_SURVIVAL"].between(0.0, 1.0).all()
        assert (
            physics["SURVIVAL_AT_HORIZON"]
            <= physics["SURVIVAL_AT_AUDIT"] + 1e-6
        ).all()
        assert np.isfinite(physics["EXPECTED_DEGRADATION_HAZARD"]).all()
        assert (physics["EXPECTED_DEGRADATION_HAZARD"] >= 0.0).all()


def test_audit_table_respects_primary_rule_and_holdout(audit_result, datasets) -> None:
    table = audit_result.audit_table
    assert not table.empty
    assert set(table["DATA_SPLIT"]).issubset({"validation", "test"})
    assert table["BRIDGE_ID"].is_unique
    assert set(table["BRIDGE_ID"]).issubset(
        set(datasets.audit_report["validation_bridge_ids"])
        | set(datasets.audit_report["test_bridge_ids"])
    )

    reported = table["REPORTED_LAST_MIN_COND"].to_numpy(int)
    survival = table["EXPECTED_SURVIVAL"].to_numpy(float)
    expected_flag = (
        (reported >= 7)
        & (reported <= 9)
        & (survival < audit_result.report["config"]["critical_survival_threshold"])
    )
    assert np.array_equal(table["PAPER_INSPECTION_SUSPECT"].to_numpy(bool), expected_flag)
    assert table["AUDIT_ALERT"].dtype == bool
    assert table["ISOLATION_FLAG"].dtype == bool
    assert np.isfinite(
        table[
            [
                "CONDITION_GAP",
                "AUDIT_Z_SCORE",
                "ISOLATION_ANOMALY_SCORE",
                "AUDIT_PRIORITY_SCORE",
            ]
        ].to_numpy(float)
    ).all()
    assert table["AUDIT_PRIORITY"].isin(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).all()


def test_audit_table_has_no_truth_or_latent_columns(audit_result) -> None:
    table = audit_result.audit_table
    forbidden_prefixes = ("TRUE_", "LATENT_", "PAPER_INSPECTION_FLAG", "INSPECTOR_")
    assert not any(column.startswith(forbidden_prefixes) for column in table.columns)
    assert "PAPER_INSPECTION_SUSPECT" in table.columns
    assert audit_result.report["checks"]["ground_truth_excluded_from_scoring"] is True
    assert audit_result.report["checks"]["reported_condition_excluded_from_survival"] is True
    assert audit_result.report["checks"]["no_truth_columns_in_audit_table"] is True
    assert all(audit_result.report["checks"].values())


def test_residual_calibration_and_isolation_forest(audit_result) -> None:
    calibration = audit_result.calibration
    assert calibration.residual_scale > 0.0
    assert calibration.calibration_split == "train"
    assert calibration.n_rows > 0
    assert audit_result.isolation_forest is not None
    assert hasattr(audit_result.isolation_forest, "decision_function")
    assert audit_result.feature_pipeline is not None

    z_scores = audit_result.audit_table["AUDIT_Z_SCORE"].to_numpy(float)
    assert np.isfinite(z_scores).all()
    assert np.abs(z_scores).max() <= audit_result.report["config"]["z_clip"] + 1e-6


def test_ground_truth_separates_paper_inspection_signal(audit_result) -> None:
    metrics = audit_result.metrics
    assert metrics["n_evaluated"] == len(audit_result.audit_table)
    assert metrics["n_paper_inspections"] > 0
    assert metrics["audit_z_score"]["roc_auc"] is not None
    assert metrics["audit_z_score"]["roc_auc"] > 0.65
    assert metrics["audit_z_score"]["average_precision"] > 0.30
    assert audit_result.report["findings"]["n_audit_alert"] > 0


def test_longer_horizon_produces_explicit_suspects(
    datasets, dataset, engine_result
) -> None:
    result = AuditVerifier(
        AuditVerifierConfig(
            model_name="cox_time_varying",
            horizon_years=10.0,
            random_state=SEED,
        )
    ).fit(
        datasets=datasets,
        bridges=dataset.bridges,
        climate=dataset.climate,
        engine_result=engine_result,
        ground_truth=dataset.ground_truth,
    )
    table = result.audit_table
    assert result.report["findings"]["n_paper_inspection_suspect"] > 0
    primary = table["PAPER_INSPECTION_SUSPECT"].to_numpy(bool)
    assert primary.sum() > 0
    assert (
        table.loc[primary, "REPORTED_LAST_MIN_COND"].between(7, 9).all()
    )
    assert (
        table.loc[primary, "EXPECTED_SURVIVAL"]
        < result.report["config"]["critical_survival_threshold"]
    ).all()


def test_audit_result_persists_json_and_binary_artifacts(
    audit_result, tmp_path: Path
) -> None:
    audit_result.save(tmp_path)
    findings_written = (tmp_path / "audit_findings.parquet").exists() or (
        tmp_path / "audit_findings.csv"
    ).exists()
    assert findings_written
    assert (tmp_path / "audit_models.joblib").exists()
    assert (tmp_path / "audit_report.json").exists()
    report = json.loads((tmp_path / "audit_report.json").read_text(encoding="utf-8"))
    assert report["module"] == "audit_verifier"
    assert report["cohort"]["n_audited"] == len(audit_result.audit_table)


def test_ground_truth_evaluation_rejects_incomplete_truth(audit_result) -> None:
    truth = pd.DataFrame(
        {
            "BRIDGE_ID": [audit_result.audit_table["BRIDGE_ID"].iloc[0]],
            "PAPER_INSPECTION_FLAG": [0],
        }
    )
    with pytest.raises(ValueError):
        evaluate_audit_against_ground_truth(audit_result.audit_table, truth)


def test_invalid_engine_model_is_rejected(datasets, dataset, engine_result) -> None:
    verifier = AuditVerifier(AuditVerifierConfig(model_name="not_a_model"))
    with pytest.raises(ValueError):
        verifier.fit(
            datasets=datasets,
            bridges=dataset.bridges,
            climate=dataset.climate,
            engine_result=engine_result,
        )
