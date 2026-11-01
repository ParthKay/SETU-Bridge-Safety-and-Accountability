"""Run the complete four-module zero-trust bridge engine end to end.

The script:

1. generates the synthetic NBI / Indian climate benchmark;
2. preprocesses it into leakage-safe static and counting-process datasets;
3. fits the Cox time-varying, random survival forest, gradient boosting, and
   ensemble survival models;
4. fits the paper-inspection audit verifier;
5. reloads the persisted artifacts, applies a post-disaster shock, and writes
   RUL triage and SHAP explanations.

Run from the project root::

    python scripts/run_end_to_end.py --quick
    python scripts/run_end_to_end.py --config config/default.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bridge_engine.audit_verifier import AuditVerifier, AuditVerifierConfig  # noqa: E402
from bridge_engine.data_pipeline import (  # noqa: E402
    COLLAPSE_ARCHETYPES,
    ID_COL,
    BridgeDataPipeline,
    PipelineConfig,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
    inject_collapse_archetypes,
    save_prepared_dataset,
)
from bridge_engine.survival_models import (  # noqa: E402
    SurvivalEngine,
    SurvivalEngineConfig,
)
from bridge_engine.triage_and_xai import (  # noqa: E402
    DisasterEvent,
    TriageConfig,
    TriageEngine,
    load_audit_result,
    load_engine_result,
)

LOGGER = logging.getLogger("end_to_end")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be an object")
    return dict(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _archetype_validation_table(
    *,
    dataset: Any,
    prepared: Any,
    engine_result: Any,
    audit_result: Any,
    triage_config: dict[str, Any],
    triage_dir: Path,
) -> list[dict[str, Any]]:
    """Score the three landmark collapse archetypes through audit and triage."""

    audit_table = audit_result.audit_table.set_index(ID_COL)
    archetype_config = dict(triage_config)
    archetype_config.update(
        {
            "save_force_plots": False,
            "top_n_explanations": 1,
            "shap_background_size": 48,
            "shap_max_evals": 192,
        }
    )
    engine = TriageEngine(TriageConfig(**archetype_config))
    rows: list[dict[str, Any]] = []

    for archetype in COLLAPSE_ARCHETYPES:
        bridge_id = str(archetype["bridge_id"])
        if bridge_id not in audit_table.index:
            raise ValueError(f"archetype {bridge_id} is absent from the audit table")
        audited = audit_table.loc[bridge_id]
        event = DisasterEvent(
            event_id=f"{bridge_id}-SCENARIO",
            event_type=str(archetype["event_type"]),
            bridge_ids=(bridge_id,),
            severity=float(archetype["event_severity"]),
        )
        result = engine.run_event(
            datasets=prepared,
            engine_result=engine_result,
            bridges=dataset.bridges,
            event=event,
            audit_result=audit_result,
            output_dir=None,
        )
        finding = result.triage_table.iloc[0]
        rows.append(
            {
                "bridge_id": bridge_id,
                "label": str(archetype["label"]),
                "state": str(archetype["state"]),
                "district": str(archetype["district"]),
                "collapse_year": int(archetype["collapse_year"]),
                "event_type": event.event_type,
                "event_severity": float(event.severity),
                "reported_condition": int(audited["REPORTED_LAST_MIN_COND"]),
                "physics_expected_condition": float(
                    audited["PHYSICS_EXPECTED_CONDITION"]
                ),
                "predicted_5y_conditional_survival": float(
                    audited["EXPECTED_SURVIVAL"]
                ),
                "condition_gap": float(audited["CONDITION_GAP"]),
                "audit_z_score": float(audited["AUDIT_Z_SCORE"]),
                "paper_inspection_suspect": bool(
                    audited["PAPER_INSPECTION_SUSPECT"]
                ),
                "audit_alert": bool(audited["AUDIT_ALERT"]),
                "audit_priority": str(audited["AUDIT_PRIORITY"]),
                "baseline_rul_years": float(finding["RUL_BASELINE_YEARS"]),
                "post_event_rul_years": float(finding["RUL_POST_EVENT_YEARS"]),
                "post_disaster_rul_loss_years": float(finding["RUL_LOSS_YEARS"]),
                "baseline_survival_at_decision": float(
                    finding["BASELINE_SURVIVAL_AT_DECISION"]
                ),
                "post_event_survival_at_decision": float(
                    finding["POST_EVENT_SURVIVAL_AT_DECISION"]
                ),
                "triage_priority": str(finding["TRIAGE_PRIORITY"]),
                "immediate_ndt_deployment": bool(
                    finding["IMMEDIATE_NDT_DEPLOYMENT"]
                ),
                "top_shap_risk_driver": (
                    str(finding["TOP_RISK_DRIVER"])
                    if "TOP_RISK_DRIVER" in finding
                    else None
                ),
                "top_shap_risk_value": (
                    float(finding["TOP_RISK_DRIVER_SHAP"])
                    if "TOP_RISK_DRIVER_SHAP" in finding
                    else None
                ),
            }
        )

    path = triage_dir / "archetype_validation.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(rows), handle, indent=2)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full zero-trust bridge survival and audit engine."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "default.json"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-bridges", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--district", type=str, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller cohort and shallower ensembles for a fast smoke run.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)
    config = _load_config(args.config)
    started = time.perf_counter()

    seed = int(args.seed or config["synthetic"]["seed"])
    n_bridges = int(args.n_bridges or config["synthetic"]["n_bridges"])
    if args.quick:
        n_bridges = min(n_bridges, 420)
    output_root = args.output_root or (ROOT / config["output"]["root"])
    data_dir = output_root / "data"
    engine_dir = output_root / "module2"
    audit_dir = output_root / "module3"
    triage_dir = output_root / "module4"
    for directory in (data_dir, engine_dir, audit_dir, triage_dir):
        directory.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Stage 1/4 | generating %s synthetic bridges", n_bridges)
    synthetic_config = SyntheticDataConfig(
        n_bridges=n_bridges,
        study_end_year=int(config["synthetic"]["study_end_year"]),
        min_build_year=int(config["synthetic"]["min_build_year"]),
        max_build_year=int(config["synthetic"]["max_build_year"]),
        paper_inspection_rate=float(config["synthetic"]["paper_inspection_rate"]),
        ndt_rate=float(config["synthetic"]["ndt_rate"]),
        random_state=seed,
    )
    dataset = SyntheticBridgeDataGenerator(synthetic_config).generate()
    dataset = inject_collapse_archetypes(
        dataset, study_end_year=int(config["synthetic"]["study_end_year"])
    )
    LOGGER.info(
        "Injected %s landmark collapse archetypes into the held-out test split",
        len(COLLAPSE_ARCHETYPES),
    )
    pipeline_config = PipelineConfig(
        test_size=float(config["pipeline"]["test_size"]),
        validation_size=float(config["pipeline"]["validation_size"]),
        interval_width=float(config["pipeline"]["interval_width"]),
        max_fit_rows=int(config["pipeline"]["max_fit_rows"]),
        imputer_max_iter=int(config["pipeline"]["imputer_max_iter"]),
        random_state=seed,
    )
    prepared = BridgeDataPipeline(pipeline_config).prepare(dataset)
    save_prepared_dataset(dataset, prepared, data_dir)
    LOGGER.info(
        "Prepared %s bridges and %s counting-process intervals",
        prepared.audit_report["n_bridges"],
        sum(part["n_intervals"] for part in prepared.audit_report["split"].values()),
    )

    LOGGER.info("Stage 2/4 | fitting survival engine")
    survival_config = config["survival"]
    horizons = tuple(float(value) for value in survival_config["horizons"])
    engine = SurvivalEngine(
        SurvivalEngineConfig(
            horizons=horizons,
            cox_penalizer=float(survival_config["cox_penalizer"]),
            rsf_n_estimators=int(survival_config["rsf_n_estimators"]),
            gbsa_n_estimators=int(survival_config["gbsa_n_estimators"]),
            n_jobs=int(survival_config["n_jobs"]),
            random_state=seed,
        )
    ).fit(prepared)
    engine.save(engine_dir)
    selected_test = engine.metrics[engine.selected_model]["test"]
    LOGGER.info(
        "Selected %s | Harrell C=%.4f | Uno C=%s | IBS=%s",
        engine.selected_model,
        selected_test["harrell_c"],
        (
            f"{selected_test['uno_ipcw_c']:.4f}"
            if selected_test["uno_ipcw_c"] is not None
            else "n/a"
        ),
        (
            f"{selected_test['integrated_brier_score']:.4f}"
            if selected_test["integrated_brier_score"] is not None
            else "n/a"
        ),
    )

    LOGGER.info("Stage 3/4 | fitting paper-inspection audit verifier")
    audit_config = config["audit"]
    audit = AuditVerifier(
        AuditVerifierConfig(
            horizon_years=float(audit_config["horizon_years"]),
            critical_survival_threshold=float(
                audit_config["critical_survival_threshold"]
            ),
            healthy_condition_min=int(audit_config["healthy_condition_min"]),
            healthy_condition_max=int(audit_config["healthy_condition_max"]),
            z_flag_threshold=float(audit_config["z_flag_threshold"]),
            model_name=str(audit_config["model_name"]),
            random_state=seed,
        )
    ).fit(
        datasets=prepared,
        bridges=dataset.bridges,
        climate=dataset.climate,
        engine_result=engine,
        ground_truth=dataset.ground_truth,
    )
    audit.save(audit_dir)
    LOGGER.info(
        "Audit findings: %s suspects, %s alerts",
        audit.report["findings"]["n_paper_inspection_suspect"],
        audit.report["findings"]["n_audit_alert"],
    )

    LOGGER.info("Stage 4/4 | post-disaster triage and SHAP explanations")
    loaded_engine = load_engine_result(engine_dir)
    loaded_audit = load_audit_result(audit_dir)
    event_config = dict(config["triage"]["event"])
    if args.district is not None:
        event_config["district"] = args.district
    observed_districts = set(dataset.bridges["DISTRICT"].astype(str))
    if event_config.get("district") not in observed_districts:
        fallback = str(
            dataset.bridges["DISTRICT"].astype(str).value_counts().index[0]
        )
        LOGGER.warning(
            "Configured district %s is absent; using %s",
            event_config.get("district"),
            fallback,
        )
        event_config["district"] = fallback
    event = DisasterEvent(**event_config)
    triage_config = dict(config["triage"]["config"])
    triage_config["horizons"] = tuple(float(value) for value in triage_config["horizons"])
    if args.quick:
        triage_config["shap_background_size"] = 32
        triage_config["shap_max_evals"] = 128
        triage_config["top_n_explanations"] = 4
    triage_config["random_state"] = seed
    triage = TriageEngine(TriageConfig(**triage_config)).run_event(
        datasets=prepared,
        engine_result=loaded_engine,
        bridges=dataset.bridges,
        event=event,
        audit_result=loaded_audit,
        output_dir=triage_dir,
    )
    triage.save(triage_dir)
    LOGGER.info(
        "Triage: %s critical, %s high, %s immediate NDT, mean RUL loss %.2f y",
        triage.metrics["priority_counts"].get("CRITICAL", 0),
        triage.metrics["priority_counts"].get("HIGH", 0),
        triage.metrics["n_immediate_ndt"],
        triage.metrics["mean_rul_loss_years"],
    )

    LOGGER.info("Archetype validation | scoring landmark collapse cases")
    archetype_validation = _archetype_validation_table(
        dataset=dataset,
        prepared=prepared,
        engine_result=loaded_engine,
        audit_result=loaded_audit,
        triage_config=triage_config,
        triage_dir=triage_dir,
    )

    summary = {
        "schema_version": 1,
        "project": "Zero-Trust Bridge Survival & Audit Verification Engine",
        "status": "complete",
        "seed": seed,
        "n_bridges": int(len(dataset.bridges)),
        "n_generated_bridges": n_bridges,
        "n_archetype_bridges": len(COLLAPSE_ARCHETYPES),
        "runtime_seconds": float(time.perf_counter() - started),
        "artifacts": {
            "data": str(data_dir),
            "survival_engine": str(engine_dir),
            "audit_verifier": str(audit_dir),
            "triage_xai": str(triage_dir),
        },
        "archetype_validation_table": archetype_validation,
        "module1": {
            "n_bridges": prepared.audit_report["n_bridges"],
            "n_intervals": sum(
                part["n_intervals"]
                for part in prepared.audit_report["split"].values()
            ),
            "checks": prepared.audit_report["checks"],
        },
        "module2": {
            "selected_model": engine.selected_model,
            "test_metrics": selected_test,
        },
        "module3": {
            "metrics": audit.metrics,
            "findings": {
                key: value
                for key, value in audit.report["findings"].items()
                if key != "top_findings"
            },
            "top_findings": audit.report["findings"]["top_findings"][:5],
        },
        "module4": {
            "event": triage.report["event"],
            "metrics": triage.metrics,
            "findings": {
                key: value
                for key, value in triage.report["findings"].items()
                if key != "top_findings"
            },
            "top_findings": triage.report["findings"]["top_findings"][:5],
            "checks": triage.report["checks"],
        },
    }
    summary_path = output_root / "end_to_end_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2)
    LOGGER.info("End-to-end pipeline complete in %.1f s", summary["runtime_seconds"])
    LOGGER.info("Summary: %s", summary_path)
    return summary


if __name__ == "__main__":
    main()
