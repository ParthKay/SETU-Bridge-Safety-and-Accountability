"""Score bridges with the persisted survival engine.

Two ways to give the engine an input:

1. **Existing cohort rows** (no preprocessing needed)::

       python scripts/score_bridges.py --bridge-ids BR-0000038,BR-0000162 --rul

   The bridge's already-preprocessed test-split row is loaded from
   ``artifacts/data`` and scored.

2. **A new bridge roster** (raw inventory columns)::

       python scripts/score_bridges.py --export-sidecar
       python scripts/score_bridges.py --bridges artifacts/data/scoring/bridges_template.csv \
           --climate artifacts/data/scoring/climate.csv --rul

   The roster is rebuilt into the same counting-process and static frames used in
   training, then transformed with the *training-fitted* imputer/encoder saved in
   ``artifacts/data/*_preprocessor.joblib`` before prediction.

Survival curves and RUL reuse the project's own Module 3/4 functions
(``conditional_survival_curve``, ``compute_rul_from_survival``) so the numbers
match the audit and triage reports exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bridge_engine.data_pipeline import (  # noqa: E402
    CATEGORICAL_FEATURE_COLUMNS,
    DURATION_COL,
    DYNAMIC_FEATURE_COLUMNS,
    EVENT_COL,
    ID_COL,
    PRE_ENTRY_CLIMATE_COLUMNS,
    STATIC_FEATURE_COLUMNS,
    STATIC_MODEL_FEATURE_COLUMNS,
    TIME_VARYING_FEATURE_COLUMNS,
    AndersenGillSplitter,
    BridgePreprocessor,
    attach_dynamic_covariates,
    build_counting_process_frame,
    build_static_survival_frame,
    validate_bridge_frame,
    validate_climate_frame,
)
from bridge_engine.triage_and_xai import (  # noqa: E402
    DisasterEvent,
    PostDisasterUpdater,
    compute_rul_from_survival,
    conditional_survival_curve,
)

SURVIVAL_FLOOR = 1e-6
SIDECAR_BRIDGE_COLUMNS: tuple[str, ...] = (
    ID_COL,
    "STATE",
    "DISTRICT",
    "ENTRY_YEAR",
    *STATIC_FEATURE_COLUMNS,
)
RAW_INPUT_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys([*SIDECAR_BRIDGE_COLUMNS, DURATION_COL, EVENT_COL])
)


def inventory_schema() -> dict[str, Any]:
    """Describe exactly what the raw-roster input needs."""

    return {
        "bridge_roster_required_columns": list(RAW_INPUT_COLUMNS),
        "categorical_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "numeric_static_columns": [
            column
            for column in STATIC_FEATURE_COLUMNS
            if column not in set(CATEGORICAL_FEATURE_COLUMNS)
        ],
        "automatically_derived": {
            "static_model_features": list(STATIC_MODEL_FEATURE_COLUMNS),
            "time_varying_model_features": list(TIME_VARYING_FEATURE_COLUMNS),
            "dynamic_features": list(DYNAMIC_FEATURE_COLUMNS),
            "pre_entry_climate_features": list(PRE_ENTRY_CLIMATE_COLUMNS),
        },
        "climate_panel_columns": [
            "STATE",
            "DISTRICT",
            "YEAR",
            "MONSOON_ANOMALY",
            "SEASONAL_RAINFALL_MM",
            "PEAK_DISCHARGE_M3S",
            "PEAK_VELOCITY_MPS",
            "PEAK_SCOUR_INDEX",
            "FLOOD_EXCEEDANCE",
            "CYCLONE_FLAG",
        ],
        "notes": (
            "EVENT is the historical failure label and is never a model input; use 0 "
            "for a bridge you only want to score. DURATION_YEARS is the exposure "
            "window since ENTRY_YEAR; for a current-condition score use "
            "as-of-year - ENTRY_YEAR (the template builder does this). The climate "
            "panel must cover every district from ENTRY_YEAR - 5 through "
            "ENTRY_YEAR + DURATION_YEARS."
        ),
    }


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input format for {path}; use CSV or Parquet")


def load_engine(engine_dir: Path) -> dict[str, Any]:
    path = engine_dir / "survival_engine.joblib"
    if not path.exists():
        raise FileNotFoundError(f"trained engine not found: {path}")
    bundle = joblib.load(path)
    missing = {"models", "static_feature_names", "time_varying_feature_names"} - set(bundle)
    if missing:
        raise ValueError(f"engine bundle is missing keys: {sorted(missing)}")
    return bundle


def load_preprocessors(data_dir: Path) -> tuple[BridgePreprocessor, BridgePreprocessor]:
    paths = {
        "static": data_dir / "static_preprocessor.joblib",
        "time_varying": data_dir / "time_varying_preprocessor.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing training-fitted preprocessor(s): "
            f"{missing}. Re-run `python scripts/run_end_to_end.py` once so the "
            "pipeline persists them into artifacts/data."
        )
    return joblib.load(paths["static"]), joblib.load(paths["time_varying"])


def _feature_names(bundle: dict[str, Any], model_name: str) -> tuple[str, ...]:
    key = (
        "time_varying_feature_names"
        if model_name == "cox_time_varying"
        else "static_feature_names"
    )
    return tuple(bundle[key])


def _curve_and_rul(
    *,
    model_name: str,
    model: Any,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    time_at_score: np.ndarray,
    horizons: np.ndarray,
    rul_threshold: float,
    rul_horizon: float,
    rul_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    survival = conditional_survival_curve(
        model_name=model_name,
        model=model,
        frame=frame,
        feature_names=feature_names,
        time_at_event=time_at_score,
        horizons=horizons,
        survival_floor=SURVIVAL_FLOOR,
    )
    rul_grid = np.arange(0.0, float(rul_horizon) + float(rul_step) / 2.0, float(rul_step))
    rul_curve = conditional_survival_curve(
        model_name=model_name,
        model=model,
        frame=frame,
        feature_names=feature_names,
        time_at_event=time_at_score,
        horizons=rul_grid,
        survival_floor=SURVIVAL_FLOOR,
    )
    rul, censored = compute_rul_from_survival(
        rul_curve, rul_grid, threshold=rul_threshold, censor_at=rul_horizon
    )
    return survival, rul, censored


def score_existing_bridges(
    *,
    bridge_ids: Sequence[str],
    bundle: dict[str, Any],
    data_dir: Path,
    model_name: str,
    split: str,
    horizons: np.ndarray,
    rul_threshold: float,
    rul_horizon: float,
    rul_step: float,
    event: DisasterEvent | None,
    event_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score rows that already exist in a prepared split (no preprocessing)."""

    model = bundle["models"][model_name]
    feature_names = _feature_names(bundle, model_name)
    ids = {str(value) for value in bridge_ids}

    if model_name == "cox_time_varying":
        frame = _read_table(data_dir / f"counting_{split}.parquet")
        frame = frame.loc[frame["id"].astype(str).isin(ids)].copy()
        if frame.empty:
            raise ValueError(f"no requested bridges found in counting_{split}")
        frame = (
            frame.sort_values(["id", "start"], kind="mergesort")
            .groupby("id", observed=True, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        id_col, time_col = "id", "stop"
    else:
        frame = _read_table(data_dir / f"static_{split}.parquet")
        frame = frame.loc[frame["bridge_id"].astype(str).isin(ids)].copy()
        if frame.empty:
            raise ValueError(f"no requested bridges found in static_{split}")
        frame = frame.reset_index(drop=True)
        id_col, time_col = "bridge_id", "duration"

    if event is not None:
        target = frame[id_col].astype(str)
        if event_ids is not None:
            target = target.loc[target.isin({str(value) for value in event_ids})]
        if target.empty:
            raise ValueError(f"event targets no rows in the {split} split")
        frame = PostDisasterUpdater().apply(
            frame, event, affected_ids=target, id_col=id_col
        )

    time_at_score = frame[time_col].to_numpy(np.float64)
    survival, rul, censored = _curve_and_rul(
        model_name=model_name,
        model=model,
        frame=frame,
        feature_names=feature_names,
        time_at_score=time_at_score,
        horizons=horizons,
        rul_threshold=rul_threshold,
        rul_horizon=rul_horizon,
        rul_step=rul_step,
    )
    risk = np.asarray(model.predict_risk(frame), dtype=np.float64)

    output = pd.DataFrame(
        {
            "BRIDGE_ID": frame[id_col].astype(str).to_numpy(),
            "MODEL": model_name,
            "DATA_SPLIT": split,
            "RISK_SCORE": risk,
            "TIME_AT_SCORE_YEARS": time_at_score.astype(np.float32),
        }
    )
    for index, horizon in enumerate(horizons):
        output[f"SURVIVAL_AT_{horizon:g}Y"] = survival[:, index].astype(np.float32)
    output["RUL_YEARS"] = rul.astype(np.float32)
    output["RUL_CENSORED"] = censored
    return output


def build_raw_frames(
    bridges: pd.DataFrame,
    climate: pd.DataFrame,
    *,
    as_of_year: int,
    interval_width: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild raw static and counting frames from an inventory roster."""

    missing = [column for column in RAW_INPUT_COLUMNS if column not in bridges.columns]
    if missing:
        raise ValueError(
            f"bridge roster is missing required columns: {missing}. "
            "Run --print-schema for the input contract."
        )
    roster = bridges.copy()
    roster[ID_COL] = roster[ID_COL].astype(str)
    roster["ENTRY_YEAR"] = roster["ENTRY_YEAR"].astype(np.int64)
    roster[DURATION_COL] = np.maximum(as_of_year - roster["ENTRY_YEAR"], 1).astype(np.float64)
    roster[EVENT_COL] = roster[EVENT_COL].fillna(0).astype(np.int8)

    unknown = sorted(set(roster["DISTRICT"].astype(str)) - set(climate["DISTRICT"].astype(str)))
    if unknown:
        raise ValueError(f"climate panel does not cover districts: {unknown}")
    validate_bridge_frame(roster)
    validate_climate_frame(climate)

    static_raw = build_static_survival_frame(roster, climate)
    intervals = AndersenGillSplitter(interval_width).split(roster)
    intervals = attach_dynamic_covariates(intervals, climate, interval_width=interval_width)
    counting_raw = build_counting_process_frame(intervals)
    return static_raw, counting_raw


def _transform(
    frame: pd.DataFrame,
    preprocessor: BridgePreprocessor,
    *,
    id_col: str,
    keep: Sequence[str] = (),
) -> pd.DataFrame:
    features = preprocessor.transform(frame)
    columns = [id_col, *keep]
    meta = frame.loc[:, columns].reset_index(drop=True)
    return pd.concat([meta, features.reset_index(drop=True)], axis=1)


def score_raw_roster(
    *,
    bridges: pd.DataFrame,
    climate: pd.DataFrame,
    bundle: dict[str, Any],
    data_dir: Path,
    model_name: str,
    as_of_year: int,
    horizons: np.ndarray,
    rul_threshold: float,
    rul_horizon: float,
    rul_step: float,
    event: DisasterEvent | None,
    event_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Score an arbitrary inventory roster through the training transform."""

    static_preprocessor, tv_preprocessor = load_preprocessors(data_dir)
    model = bundle["models"][model_name]
    feature_names = _feature_names(bundle, model_name)

    static_raw, counting_raw = build_raw_frames(
        bridges, climate, as_of_year=as_of_year
    )

    if model_name == "cox_time_varying":
        latest = (
            counting_raw.sort_values(["id", "start"], kind="mergesort")
            .groupby("id", observed=True, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        frame = _transform(latest, tv_preprocessor, id_col="id", keep=("stop",))
        id_col, time_col = "id", "stop"
    else:
        static_frame = static_raw.rename(columns={ID_COL: "bridge_id"})
        frame = _transform(
            static_frame, static_preprocessor, id_col="bridge_id", keep=("duration",)
        )
        id_col, time_col = "bridge_id", "duration"

    if event is not None:
        target = frame[id_col].astype(str)
        if event_ids is not None:
            target = target.loc[target.isin({str(value) for value in event_ids})]
        if target.empty:
            raise ValueError("event targets none of the supplied bridges")
        frame = PostDisasterUpdater().apply(
            frame, event, affected_ids=target, id_col=id_col
        )

    time_at_score = frame[time_col].to_numpy(np.float64)
    survival, rul, censored = _curve_and_rul(
        model_name=model_name,
        model=model,
        frame=frame,
        feature_names=feature_names,
        time_at_score=time_at_score,
        horizons=horizons,
        rul_threshold=rul_threshold,
        rul_horizon=rul_horizon,
        rul_step=rul_step,
    )
    risk = np.asarray(model.predict_risk(frame), dtype=np.float64)

    output = pd.DataFrame(
        {
            "BRIDGE_ID": frame[id_col].astype(str).to_numpy(),
            "MODEL": model_name,
            "AS_OF_YEAR": as_of_year,
            "RISK_SCORE": risk,
            "TIME_AT_SCORE_YEARS": time_at_score.astype(np.float32),
        }
    )
    for index, horizon in enumerate(horizons):
        output[f"SURVIVAL_AT_{horizon:g}Y"] = survival[:, index].astype(np.float32)
    output["RUL_YEARS"] = rul.astype(np.float32)
    output["RUL_CENSORED"] = censored
    return output


def export_sidecar(
    data_dir: Path, output_dir: Path, *, as_of_year: int = 2024
) -> tuple[Path, Path]:
    """Write a roster template and climate panel from the shipped cohort."""

    bridges = _read_table(data_dir / "bridges.parquet")
    climate = _read_table(data_dir / "climate.parquet")
    roster = bridges.loc[:, list(SIDECAR_BRIDGE_COLUMNS)].copy()
    roster = roster.loc[roster["ENTRY_YEAR"] <= as_of_year - 1].copy()
    roster[DURATION_COL] = (as_of_year - roster["ENTRY_YEAR"]).astype(np.float64)
    roster[EVENT_COL] = 0
    roster = roster.loc[:, list(RAW_INPUT_COLUMNS)]

    output_dir.mkdir(parents=True, exist_ok=True)
    roster_path = output_dir / "bridges_template.csv"
    climate_path = output_dir / "climate.csv"
    roster.to_csv(roster_path, index=False)
    climate.to_csv(climate_path, index=False)
    return roster_path, climate_path


def _event_bridge_ids(metadata: pd.DataFrame, event: DisasterEvent) -> list[str]:
    """Resolve an event's district/state/bridge scope to concrete bridge IDs."""

    if event.bridge_ids:
        return list(event.bridge_ids)
    mask = pd.Series(True, index=metadata.index)
    if event.district is not None:
        mask &= metadata["DISTRICT"].astype(str) == str(event.district)
    if event.state is not None:
        mask &= metadata["STATE"].astype(str) == str(event.state)
    return metadata.loc[mask, ID_COL].astype(str).tolist()


def _build_event(args: argparse.Namespace) -> DisasterEvent | None:
    if args.event_district is None:
        return None
    return DisasterEvent(
        event_id=args.event_id,
        event_type="COMPOUND",
        district=args.event_district,
        severity=args.severity,
        monsoon_anomaly_increment=args.monsoon_increment,
        rainfall_multiplier=args.rainfall_multiplier,
        discharge_multiplier=args.discharge_multiplier,
        velocity_multiplier=args.velocity_multiplier,
        scour_multiplier=args.scour_multiplier,
        flood_exceedance_increment=args.flood_increment,
        cumulative_scour_increment=args.scour_increment,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score bridges with the trained survival engine.")
    parser.add_argument("--engine-dir", type=Path, default=ROOT / "artifacts" / "module2")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts" / "data")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--split", type=str, default="test", choices=("train", "validation", "test")
    )
    parser.add_argument(
        "--bridge-ids", type=str, default=None, help="Comma-separated IDs from the cohort."
    )
    parser.add_argument("--bridges", type=Path, default=None, help="Raw bridge roster CSV/Parquet.")
    parser.add_argument("--climate", type=Path, default=None, help="District-year climate panel.")
    parser.add_argument("--as-of-year", type=int, default=2024)
    parser.add_argument("--horizons", type=str, default="5,10,25,40")
    parser.add_argument("--rul-threshold", type=float, default=0.68)
    parser.add_argument("--rul-horizon", type=float, default=60.0)
    parser.add_argument("--rul-step", type=float, default=1.0)
    parser.add_argument("--event-district", type=str, default=None)
    parser.add_argument("--event-id", type=str, default="USER-EVENT")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--monsoon-increment", type=float, default=2.5)
    parser.add_argument("--rainfall-multiplier", type=float, default=1.8)
    parser.add_argument("--discharge-multiplier", type=float, default=2.0)
    parser.add_argument("--velocity-multiplier", type=float, default=1.5)
    parser.add_argument("--scour-multiplier", type=float, default=2.2)
    parser.add_argument("--flood-increment", type=int, default=1)
    parser.add_argument("--scour-increment", type=float, default=12.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--export-sidecar", action="store_true")
    parser.add_argument("--print-schema", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.print_schema:
        print(json.dumps(inventory_schema(), indent=2))
        return

    if args.export_sidecar:
        roster_path, climate_path = export_sidecar(
            args.data_dir, args.data_dir / "scoring", as_of_year=args.as_of_year
        )
        print(f"roster template: {roster_path}")
        print(f"climate panel:   {climate_path}")
        if args.bridges is None and args.climate is None:
            return
        if args.bridges is None:
            args.bridges = roster_path
        if args.climate is None:
            args.climate = climate_path

    bundle = load_engine(args.engine_dir)
    model_name = args.model or str(bundle.get("selected_model", "cox_time_varying"))
    if model_name not in bundle["models"]:
        raise ValueError(f"unknown model {model_name!r}; available: {sorted(bundle['models'])}")
    horizons = np.asarray(
        sorted(float(value) for value in args.horizons.split(",") if value.strip()),
        dtype=np.float64,
    )
    event = _build_event(args)
    event_ids: list[str] | None = None
    if event is not None:
        metadata = _read_table(args.data_dir / "bridges.parquet")
        event_ids = _event_bridge_ids(metadata, event)
        if not event_ids:
            raise SystemExit(
                f"event target {event.target_description!r} matches no bridge in the cohort"
            )

    common = dict(
        bundle=bundle,
        model_name=model_name,
        horizons=horizons,
        rul_threshold=args.rul_threshold,
        rul_horizon=args.rul_horizon,
        rul_step=args.rul_step,
        event=event,
    )
    if args.bridge_ids is not None:
        ids = [value.strip() for value in args.bridge_ids.split(",") if value.strip()]
        table = score_existing_bridges(
            bridge_ids=ids,
            data_dir=args.data_dir,
            split=args.split,
            event_ids=event_ids,
            **common,
        )
    elif args.bridges is not None and args.climate is not None:
        table = score_raw_roster(
            bridges=_read_table(args.bridges),
            climate=_read_table(args.climate),
            data_dir=args.data_dir,
            as_of_year=args.as_of_year,
            event_ids=event_ids,
            **common,
        )
    else:
        raise SystemExit(
            "provide --bridge-ids for cohort rows, or --bridges/--climate for a raw roster. "
            "Run --export-sidecar to create a template."
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"wrote {len(table)} scored bridges to {args.output}")
    else:
        with pd.option_context("display.width", 220, "display.max_columns", 50):
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
