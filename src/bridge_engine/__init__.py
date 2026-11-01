"""Zero-trust bridge survival and audit verification engine.

Public objects are imported lazily so that loading one fitted artifact does not
pay the import cost of every analysis module.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ARCHETYPE_ID_PREFIX",
    "AUDIT_FEATURE_COLUMNS",
    "AndersenGillSplitter",
    "AuditCalibration",
    "AuditVerifier",
    "AuditVerifierConfig",
    "AuditVerifierResult",
    "BridgeDataPipeline",
    "BridgePreprocessor",
    "COLLAPSE_ARCHETYPES",
    "CoxTimeVaryingSurvivalModel",
    "DisasterEvent",
    "GradientBoostingSurvivalModel",
    "PipelineConfig",
    "PostDisasterUpdater",
    "RandomSurvivalForestModel",
    "ShapExplanation",
    "ShapRiskExplainer",
    "SurvivalDatasets",
    "SurvivalEngine",
    "SurvivalEngineConfig",
    "SurvivalEngineResult",
    "SurvivalEnsembleModel",
    "SyntheticBridgeDataGenerator",
    "SyntheticDataConfig",
    "SyntheticDataset",
    "TriageConfig",
    "TriageEngine",
    "TriageResult",
    "archetype_bridge_ids",
    "assert_audit_feature_allowlist",
    "build_audit_metadata",
    "build_followup_exposure_features",
    "build_sksurv_target",
    "compute_rul_from_survival",
    "conditional_survival_curve",
    "evaluate_audit_against_ground_truth",
    "evaluate_survival_predictions",
    "inject_collapse_archetypes",
    "load_audit_result",
    "load_engine_result",
    "prepare_counting_design",
    "prepare_static_design",
    "predict_conditional_survival",
]

_LAZY_IMPORTS: dict[str, str] = {
    "AUDIT_FEATURE_COLUMNS": ".audit_verifier",
    "AuditCalibration": ".audit_verifier",
    "AuditVerifier": ".audit_verifier",
    "AuditVerifierConfig": ".audit_verifier",
    "AuditVerifierResult": ".audit_verifier",
    "assert_audit_feature_allowlist": ".audit_verifier",
    "build_audit_metadata": ".audit_verifier",
    "evaluate_audit_against_ground_truth": ".audit_verifier",
    "predict_conditional_survival": ".audit_verifier",
    "ARCHETYPE_ID_PREFIX": ".data_pipeline",
    "COLLAPSE_ARCHETYPES": ".data_pipeline",
    "AndersenGillSplitter": ".data_pipeline",
    "BridgeDataPipeline": ".data_pipeline",
    "BridgePreprocessor": ".data_pipeline",
    "PipelineConfig": ".data_pipeline",
    "SurvivalDatasets": ".data_pipeline",
    "SyntheticBridgeDataGenerator": ".data_pipeline",
    "SyntheticDataConfig": ".data_pipeline",
    "SyntheticDataset": ".data_pipeline",
    "archetype_bridge_ids": ".data_pipeline",
    "build_sksurv_target": ".data_pipeline",
    "inject_collapse_archetypes": ".data_pipeline",
    "CoxTimeVaryingSurvivalModel": ".survival_models",
    "GradientBoostingSurvivalModel": ".survival_models",
    "RandomSurvivalForestModel": ".survival_models",
    "SurvivalEngine": ".survival_models",
    "SurvivalEngineConfig": ".survival_models",
    "SurvivalEngineResult": ".survival_models",
    "SurvivalEnsembleModel": ".survival_models",
    "build_followup_exposure_features": ".survival_models",
    "evaluate_survival_predictions": ".survival_models",
    "prepare_counting_design": ".survival_models",
    "prepare_static_design": ".survival_models",
    "DisasterEvent": ".triage_and_xai",
    "PostDisasterUpdater": ".triage_and_xai",
    "ShapExplanation": ".triage_and_xai",
    "ShapRiskExplainer": ".triage_and_xai",
    "TriageConfig": ".triage_and_xai",
    "TriageEngine": ".triage_and_xai",
    "TriageResult": ".triage_and_xai",
    "compute_rul_from_survival": ".triage_and_xai",
    "conditional_survival_curve": ".triage_and_xai",
    "load_audit_result": ".triage_and_xai",
    "load_engine_result": ".triage_and_xai",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
