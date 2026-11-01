"""Module 4 verification tests: disaster update, RUL, SHAP, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bridge_engine.audit_verifier import AuditVerifier, AuditVerifierConfig
from bridge_engine.data_pipeline import (
    ID_COL,
    BridgeDataPipeline,
    PipelineConfig,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
)
from bridge_engine.survival_models import SurvivalEngine, SurvivalEngineConfig
from bridge_engine.triage_and_xai import (
    DisasterEvent,
    PostDisasterUpdater,
    TriageConfig,
    TriageEngine,
    compute_rul_from_survival,
)

N_BRIDGES = 360
SEED = 2030


@pytest.fixture(scope="module")
def dataset():
    return SyntheticBridgeDataGenerator(
        SyntheticDataConfig(n_bridges=N_BRIDGES, random_state=SEED)
    ).generate()


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
            rsf_n_estimators=15,
            gbsa_n_estimators=15,
            n_jobs=1,
            random_state=SEED,
        )
    ).fit(datasets)


@pytest.fixture(scope="module")
def audit_result(datasets, dataset, engine_result):
    return AuditVerifier(
        AuditVerifierConfig(model_name="cox_time_varying", random_state=SEED)
    ).fit(
        datasets=datasets,
        bridges=dataset.bridges,
        climate=dataset.climate,
        engine_result=engine_result,
        ground_truth=dataset.ground_truth,
    )


@pytest.fixture(scope="module")
def event(dataset, datasets) -> DisasterEvent:
    test_ids = set(map(str, datasets.audit_report["test_bridge_ids"]))
    cohort = dataset.bridges[dataset.bridges[ID_COL].astype(str).isin(test_ids)]
    district = str(cohort["DISTRICT"].astype(str).value_counts().index[0])
    return DisasterEvent(
        event_id="TEST-BIHAR-2024",
        event_type="COMPOUND",
        district=district,
        severity=1.35,
    )


@pytest.fixture(scope="module")
def triage_result(datasets, dataset, engine_result, audit_result, event, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("module4")
    config = TriageConfig(
        horizons=(5.0, 10.0),
        decision_horizon_years=5.0,
        rul_horizon_years=20.0,
        rul_step_years=2.0,
        top_n_explanations=3,
        shap_background_size=24,
        shap_max_evals=65,
        random_state=SEED,
        save_force_plots=True,
    )
    result = TriageEngine(config).run_event(
        datasets=datasets,
        engine_result=engine_result,
        bridges=dataset.bridges,
        event=event,
        audit_result=audit_result,
        output_dir=output_dir,
    )
    return result


def test_event_and_config_validation() -> None:
    with pytest.raises(ValueError):
        DisasterEvent(event_id="x", event_type="FLOOD")
    with pytest.raises(ValueError):
        DisasterEvent(event_id="x", event_type="FLOOD", district="Darbhanga", severity=0.0)
    with pytest.raises(ValueError):
        DisasterEvent(event_id="x", event_type="TSUNAMI", district="Darbhanga")
    with pytest.raises(ValueError):
        TriageConfig(horizons=(5.0, 10.0), decision_horizon_years=7.0)
    with pytest.raises(ValueError):
        TriageConfig(horizons=(10.0, 5.0))
    with pytest.raises(ValueError):
        TriageConfig(top_n_explanations=-1)


def test_post_disaster_updater_is_scoped_and_increases_exposure(
    datasets, dataset, event
) -> None:
    frame = datasets.counting_test.sort_values(["id", "start"]).groupby("id").tail(1)
    affected = set(
        frame.loc[
            frame["id"].astype(str).map(
                dataset.bridges.set_index(dataset.bridges[ID_COL].astype(str))[
                    "DISTRICT"
                ].astype(str)
            )
            == event.district,
            "id",
        ].astype(str)
    )
    assert affected
    before = frame.copy()
    shocked = PostDisasterUpdater().apply(
        frame, event, affected_ids=affected, id_col="id"
    )
    affected_mask = shocked["id"].astype(str).isin(affected)
    for column in (
        "MONSOON_ANOMALY",
        "SEASONAL_RAINFALL_MM",
        "PEAK_DISCHARGE_M3S",
        "PEAK_SCOUR_INDEX",
        "CUMULATIVE_FLOOD_EXCEEDANCE",
        "CUMULATIVE_SCOUR_LOAD",
    ):
        assert (
            shocked.loc[affected_mask, column].to_numpy(float)
            >= before.loc[affected_mask, column].to_numpy(float) - 1e-9
        ).all()
        assert np.allclose(
            shocked.loc[~affected_mask, column].to_numpy(float),
            before.loc[~affected_mask, column].to_numpy(float),
        )
    assert (
        shocked.loc[affected_mask, "PEAK_SCOUR_INDEX"].to_numpy(float)
        > before.loc[affected_mask, "PEAK_SCOUR_INDEX"].to_numpy(float)
    ).all()


def test_compute_rul_interpolates_threshold_crossing() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    survival = np.array(
        [
            [1.0, 0.8, 0.6, 0.3, 0.2],
            [1.0, 0.9, 0.8, 0.7, 0.6],
        ]
    )
    rul, censored = compute_rul_from_survival(
        survival, times, threshold=0.45, censor_at=4.0
    )
    assert rul[0] == pytest.approx(2.5)
    assert not censored[0]
    assert rul[1] == pytest.approx(4.0)
    assert censored[1]


def test_compute_rul_boundary_cases() -> None:
    times = np.array([0.0, 1.0, 60.0])
    survival = np.array(
        [
            [0.60, 0.50, 0.40],
            [1.00, 0.90, 0.80],
        ]
    )
    rul, censored = compute_rul_from_survival(
        survival, times, threshold=0.68, censor_at=60.0
    )
    assert rul[0] == 0.0
    assert not censored[0]
    assert rul[1] == 60.0
    assert censored[1]


def test_triage_table_and_priorities(triage_result, datasets, event) -> None:
    table = triage_result.triage_table
    assert not table.empty
    assert table["BRIDGE_ID"].is_unique
    assert set(table["BRIDGE_ID"]).issubset(
        set(map(str, datasets.audit_report["test_bridge_ids"]))
    )
    assert (table["EVENT_ID"] == event.event_id).all()
    assert table["TRIAGE_PRIORITY"].isin(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ).all()
    for column in (
        "BASELINE_SURVIVAL_AT_DECISION",
        "POST_EVENT_SURVIVAL_AT_DECISION",
        "SURVIVAL_DROP_AT_DECISION",
        "RUL_BASELINE_YEARS",
        "RUL_POST_EVENT_YEARS",
        "RUL_LOSS_YEARS",
        "TRIAGE_PRIORITY_SCORE",
    ):
        assert np.isfinite(table[column].to_numpy(float)).all()
    assert (table["RUL_LOSS_YEARS"] >= 0.0).all()
    assert (table["TRIAGE_PRIORITY_SCORE"] >= 0.0).all()
    assert (table["TRIAGE_PRIORITY_SCORE"] <= 1.0 + 1e-9).all()
    assert table["AUDIT_PRIORITY"].notna().all()
    assert table["IMMEDIATE_NDT_DEPLOYMENT"].dtype == bool


def test_shap_explanations_are_objective_and_additive(triage_result) -> None:
    contributions = triage_result.xai_contributions
    assert not contributions.empty
    assert {
        "BRIDGE_ID",
        "FEATURE",
        "FEATURE_VALUE",
        "SHAP_VALUE",
        "ABS_SHAP_VALUE",
        "DIRECTION",
    }.issubset(contributions.columns)
    forbidden = ("TRUE_", "LATENT_", "PAPER_", "INSPECTOR_", "AUDIT_", "ISOLATION_")
    assert not contributions["FEATURE"].str.startswith(forbidden).any()
    assert np.isfinite(contributions["SHAP_VALUE"].to_numpy(float)).all()
    assert triage_result.report["checks"]["shap_additivity_ok"] is True
    assert triage_result.report["xai"]["n_explained"] > 0
    assert triage_result.force_plot_paths
    for path in triage_result.force_plot_paths:
        assert Path(path).exists()


def test_triage_report_checks_and_metrics(triage_result) -> None:
    report = triage_result.report
    assert report["module"] == "triage_and_xai"
    assert report["event"]["event_type"] == "COMPOUND"
    assert report["cohort"]["n_scored"] == len(triage_result.triage_table)
    assert report["findings"]["n_immediate_ndt"] >= 0
    assert all(report["checks"].values())
    assert triage_result.metrics["n_scored"] == len(triage_result.triage_table)
    assert triage_result.metrics["mean_rul_loss_years"] >= 0.0


def test_tree_model_triage_path(datasets, dataset, engine_result, event) -> None:
    result = TriageEngine(
        TriageConfig(
            model_name="random_survival_forest",
            explain_model_name="ensemble",
            horizons=(5.0, 10.0),
            decision_horizon_years=5.0,
            rul_horizon_years=10.0,
            rul_step_years=5.0,
            top_n_explanations=1,
            shap_background_size=16,
            shap_max_evals=35,
            random_state=SEED,
            save_force_plots=False,
        )
    ).run_event(
        datasets=datasets,
        engine_result=engine_result,
        bridges=dataset.bridges,
        event=event,
        audit_result=None,
    )
    assert not result.triage_table.empty
    assert (result.triage_table["RUL_POST_EVENT_YEARS"] >= 0.0).all()
    assert result.report["cohort"]["model_name"] == "random_survival_forest"
    assert result.report["cohort"]["explain_model_name"] == "ensemble"


def test_triage_result_persists_artifacts(triage_result, tmp_path: Path) -> None:
    triage_result.save(tmp_path)
    assert (tmp_path / "triage_report.json").exists()
    findings = (tmp_path / "triage_findings.parquet").exists() or (
        tmp_path / "triage_findings.csv"
    ).exists()
    contributions = (tmp_path / "xai_contributions.parquet").exists() or (
        tmp_path / "xai_contributions.csv"
    ).exists()
    assert findings
    assert contributions
    report = json.loads((tmp_path / "triage_report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["cohort"]["n_scored"] == len(triage_result.triage_table)
    assert len(report["xai"]["force_plot_files"]) == len(triage_result.force_plot_paths)


def test_missing_district_is_rejected(datasets, dataset, engine_result) -> None:
    event = DisasterEvent(
        event_id="NO-MATCH",
        event_type="FLOOD",
        district="Atlantis",
        severity=1.0,
    )
    with pytest.raises(ValueError):
        TriageEngine(
            TriageConfig(
                horizons=(5.0, 10.0),
                top_n_explanations=0,
                save_force_plots=False,
            )
        ).run_event(
            datasets=datasets,
            engine_result=engine_result,
            bridges=dataset.bridges,
            event=event,
        )
