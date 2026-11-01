"""Module 1: longitudinal, climate-aware bridge survival data pipeline.

The module provides three production components:

1. :class:`SyntheticBridgeDataGenerator`
   Generates a realistic longitudinal panel of NBI-style bridges, district-level
   Indian monsoon/scour shocks, visual inspection records, and an *evaluation-only*
   ground-truth table.  The generator is deliberately physics-driven: hydraulic
   scour, flood exceedance, traffic loading, material, design, and age all enter
   the latent deterioration process.

2. :class:`AndersenGillSplitter`
   Converts bridge-level failure/censoring records into Andersen-Gill counting
   process intervals and joins annual, point-in-time district climate covariates.

3. :class:`BridgeDataPipeline`
   Performs leakage-safe bridge-level train/validation/test splitting, iterative
   imputation, categorical encoding, static `sksurv` target construction, and
   `lifelines` counting-process preparation.  Verification checks are executed
   as assertions, not as optional diagnostics.

Only classical statistical / machine-learning tools are used.  No deep-learning
framework is imported.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

RANDOM_STATE: Final[int] = 42

ID_COL: Final[str] = "BRIDGE_ID"
DURATION_COL: Final[str] = "DURATION_YEARS"
EVENT_COL: Final[str] = "EVENT"
T_START_COL: Final[str] = "T_START"
T_STOP_COL: Final[str] = "T_STOP"

# ---------------------------------------------------------------------------
# Feature schemas
# ---------------------------------------------------------------------------

STATIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "STATE",
    "DISTRICT",
    "LATITUDE",
    "LONGITUDE",
    "FLOOD_PRONENESS",
    "YEAR_BUILT",
    "STRUCTURE_KIND_MATERIAL",
    "STRUCTURE_TYPE_DESIGN",
    "ADT_TRAFFIC",
    "TRUCK_ADT_PCT",
    "MAX_SPAN_LEN",
    "WATERWAY_ADEQUACY",
    "DESIGN_LOAD",
    "SCOUR_CRITICAL",
    "NUM_SPANS",
    "DECK_LEN",
    "ROADWAY_WIDTH",
    "OWNER",
    "TRAFFIC_GROWTH_RATE",
)

PRE_ENTRY_CLIMATE_COLUMNS: Final[tuple[str, ...]] = (
    "PRE_ENTRY_MONSOON_MEAN_5Y",
    "PRE_ENTRY_MONSOON_MAX_5Y",
    "PRE_ENTRY_SCOUR_MAX_5Y",
    "PRE_ENTRY_FLOOD_EXCEEDANCES_5Y",
    "PRE_ENTRY_RAINFALL_MEAN_5Y",
)

DYNAMIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "AGE_YEARS",
    "ADT_CURRENT",
    "TRUCK_ADT_PCT_CURRENT",
    "MONSOON_ANOMALY",
    "SEASONAL_RAINFALL_MM",
    "PEAK_DISCHARGE_M3S",
    "PEAK_VELOCITY_MPS",
    "PEAK_SCOUR_INDEX",
    "FLOOD_EXCEEDANCE",
    "CUMULATIVE_FLOOD_EXCEEDANCE",
    "CUMULATIVE_SCOUR_LOAD",
)

STATIC_MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    STATIC_FEATURE_COLUMNS + PRE_ENTRY_CLIMATE_COLUMNS
)
TIME_VARYING_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    STATIC_FEATURE_COLUMNS + DYNAMIC_FEATURE_COLUMNS
)

CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "STATE",
    "DISTRICT",
    "STRUCTURE_KIND_MATERIAL",
    "STRUCTURE_TYPE_DESIGN",
    "DESIGN_LOAD",
    "OWNER",
)

NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in dict.fromkeys(STATIC_MODEL_FEATURE_COLUMNS + DYNAMIC_FEATURE_COLUMNS)
    if column not in CATEGORICAL_FEATURE_COLUMNS
)

# Columns that must never enter a model matrix.  Prefix matching catches all
# truth-side and inspection-side audit fields.
FORBIDDEN_FEATURE_PREFIXES: Final[tuple[str, ...]] = (
    "_TRUE",
    "TRUE_",
    "LATENT_",
    "REPORTED_",
    "CONDITION_",
    "PAPER_",
    "INSPECTOR_",
    "SURVIVAL_",
)

FORBIDDEN_FEATURE_NAMES: Final[frozenset[str]] = frozenset(
    {
        ID_COL,
        "bridge_id",
        "id",
        DURATION_COL,
        "duration",
        EVENT_COL,
        "event",
        T_START_COL,
        "start",
        T_STOP_COL,
        "stop",
        "ENTRY_YEAR",
        "EVENT_YEAR",
        "FAILURE_YEAR",
        "LAST_INSPECTION_YEAR",
        "LAST_INSPECTION_AGE",
        "N_INSPECTIONS",
        "CALENDAR_YEAR",
        "VERIFIED_EVENT",
    }
)

CLIMATE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
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
)

BRIDGE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    ID_COL,
    "STATE",
    "DISTRICT",
    "YEAR_BUILT",
    "STRUCTURE_KIND_MATERIAL",
    "STRUCTURE_TYPE_DESIGN",
    "ADT_TRAFFIC",
    "TRUCK_ADT_PCT",
    "MAX_SPAN_LEN",
    "WATERWAY_ADEQUACY",
    "ENTRY_YEAR",
    DURATION_COL,
    EVENT_COL,
)


# ---------------------------------------------------------------------------
# Synthetic region / engineering priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionProfile:
    """A district-level climate and flood-exposure profile."""

    state: str
    district: str
    flood_prone: float
    mean_rainfall_mm: float
    coastal: bool
    latitude: float
    longitude: float
    weight: float


REGIONS: Final[tuple[RegionProfile, ...]] = (
    RegionProfile("Gujarat", "Morbi", 0.58, 850.0, False, 22.82, 70.84, 1.10),
    RegionProfile("Gujarat", "Surat", 0.82, 1250.0, True, 21.17, 72.83, 0.95),
    RegionProfile("Maharashtra", "Mumbai Suburban", 0.86, 2400.0, True, 19.11, 72.87, 1.20),
    RegionProfile("Maharashtra", "Pune", 0.52, 1100.0, False, 18.52, 73.86, 0.85),
    RegionProfile("Bihar", "Darbhanga", 0.93, 1250.0, False, 26.15, 85.90, 1.10),
    RegionProfile("Bihar", "Madhubani", 0.91, 1200.0, False, 26.35, 86.07, 0.90),
    RegionProfile("Bihar", "Patna", 0.86, 1100.0, False, 25.59, 85.13, 1.00),
    RegionProfile("Assam", "Cachar", 0.95, 2800.0, False, 24.83, 92.78, 1.00),
    RegionProfile("Assam", "Dibrugarh", 0.89, 2600.0, False, 27.47, 94.91, 0.80),
    RegionProfile("Kerala", "Wayanad", 0.80, 3000.0, False, 11.69, 76.13, 0.90),
    RegionProfile("Kerala", "Ernakulam", 0.70, 2900.0, True, 9.98, 76.28, 0.80),
    RegionProfile("Uttarakhand", "Chamoli", 0.76, 1600.0, False, 30.41, 79.32, 0.70),
    RegionProfile("Uttarakhand", "Rudraprayag", 0.79, 1700.0, False, 30.28, 78.98, 0.65),
    RegionProfile("Odisha", "Cuttack", 0.86, 1500.0, True, 20.46, 85.88, 0.90),
    RegionProfile("Odisha", "Puri", 0.78, 1450.0, True, 19.81, 85.83, 0.70),
    RegionProfile("West Bengal", "Darjeeling", 0.73, 3000.0, False, 27.04, 88.26, 0.70),
    RegionProfile("West Bengal", "Howrah", 0.72, 1600.0, False, 22.59, 88.31, 0.85),
    RegionProfile("Tamil Nadu", "Chennai", 0.64, 1400.0, True, 13.08, 80.27, 0.75),
    RegionProfile("Andhra Pradesh", "Krishna", 0.76, 1100.0, True, 16.51, 80.62, 0.80),
    RegionProfile("Himachal Pradesh", "Mandi", 0.61, 1400.0, False, 31.71, 76.93, 0.60),
    RegionProfile("Madhya Pradesh", "Narmadapuram", 0.56, 1200.0, False, 22.75, 78.10, 0.70),
    RegionProfile("Rajasthan", "Jodhpur", 0.20, 350.0, False, 26.24, 73.02, 0.40),
    RegionProfile("Karnataka", "Udupi", 0.54, 3500.0, True, 13.34, 74.75, 0.55),
    RegionProfile("Meghalaya", "East Khasi Hills", 0.81, 3200.0, False, 25.56, 91.88, 0.50),
    RegionProfile("Tripura", "West Tripura", 0.84, 2400.0, False, 23.83, 91.28, 0.60),
)

MATERIAL_DEGRADATION_FACTOR: Final[dict[str, float]] = {
    "Concrete": 1.00,
    "Prestressed Concrete": 0.82,
    "Steel": 1.28,
    "Composite": 1.05,
    "Masonry": 0.92,
    "Timber": 1.55,
}
MATERIAL_PROBABILITIES: Final[tuple[float, ...]] = (0.30, 0.22, 0.16, 0.10, 0.14, 0.08)

DESIGN_DEGRADATION_FACTOR: Final[dict[str, float]] = {
    "Slab": 0.95,
    "Beam/Girder": 1.00,
    "Box Girder": 0.88,
    "Truss": 1.12,
    "Arch": 0.90,
    "Suspension": 1.02,
    "Culvert": 0.72,
}
DESIGN_PROBABILITIES: Final[tuple[float, ...]] = (0.20, 0.32, 0.13, 0.13, 0.10, 0.04, 0.08)

OWNER_NAMES: Final[tuple[str, ...]] = ("NHAI", "State PWD", "Municipal", "Railways")
OWNER_PROBABILITIES: Final[tuple[float, ...]] = (0.27, 0.43, 0.20, 0.10)

DESIGN_LOAD_NAMES: Final[tuple[str, ...]] = (
    "IRC Class A",
    "IRC Class 70R",
    "Pre-IRC",
    "Unknown",
)
DESIGN_LOAD_PROBABILITIES: Final[tuple[float, ...]] = (0.36, 0.26, 0.28, 0.10)


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for the synthetic NBI / Indian climate benchmark."""

    n_bridges: int = 4000
    study_end_year: int = 2024
    min_build_year: int = 1950
    max_build_year: int = 2018
    paper_inspection_rate: float = 0.18
    ndt_rate: float = 0.03
    missing_rate_scale: float = 1.0
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.n_bridges <= 0:
            raise ValueError("n_bridges must be positive")
        if not 1900 < self.min_build_year <= self.max_build_year < self.study_end_year:
            raise ValueError("build years must be ordered and before study_end_year")
        if not 0.0 <= self.paper_inspection_rate <= 0.8:
            raise ValueError("paper_inspection_rate must be in [0, 0.8]")
        if not 0.0 <= self.ndt_rate <= 0.5:
            raise ValueError("ndt_rate must be in [0, 0.5]")
        if self.missing_rate_scale < 0.0:
            raise ValueError("missing_rate_scale must be non-negative")


@dataclass(frozen=True)
class SyntheticDataset:
    """Output container for the synthetic benchmark.

    ``ground_truth`` is an evaluation-only artefact.  It is never accepted by the
    modelling pipeline and must not be joined to training features.
    """

    bridges: pd.DataFrame
    climate: pd.DataFrame
    inspections: pd.DataFrame
    ground_truth: pd.DataFrame


@dataclass(frozen=True)
class _ClimateSimulation:
    frame: pd.DataFrame
    start_year: int
    monsoon_anomaly: np.ndarray
    scour_index: np.ndarray
    flood_exceedance: np.ndarray
    cyclone_flag: np.ndarray


# ---------------------------------------------------------------------------
# Landmark Indian collapse archetypes (held-out test only)
# ---------------------------------------------------------------------------

ARCHETYPE_ID_PREFIX: Final[str] = "ARCHETYPE-"

# The climate simulator is never allowed to start later than this year.  The
# deterministic landmark archetypes include a 1910 bridge, and the pre-entry
# rolling window needs five complete years before every bridge's entry year.
# Flooring the panel at 1900 makes the climate history available for vintage
# structures without silently producing NaN pre-entry covariates.
CLIMATE_START_FLOOR_YEAR: Final[int] = 1900

# Deterministic engineering archetypes distilled from three documented Indian
# bridge failures.  They are appended after the stochastic generator has run, so
# the seeded benchmark panel is unchanged, and ``split_bridge_ids`` forces every
# ``ARCHETYPE-*`` row into the test split.  Each row therefore exercises the
# zero-trust audit on a bridge that was never available during training.
COLLAPSE_ARCHETYPES: Final[tuple[dict[str, Any], ...]] = (
    {
        "bridge_id": "ARCHETYPE-CSMT-2019",
        "label": "Mumbai CSMT pedestrian footbridge (2019)",
        "state": "Maharashtra",
        "district": "Mumbai Suburban",
        "latitude": 18.9398,
        "longitude": 72.8355,
        "flood_prone": 0.92,
        "year_built": 1983,
        "entry_year": 1985,
        "material": "Steel",
        "design": "Beam/Girder",
        "adt_traffic": 48_000.0,
        "truck_adt_pct": 12.0,
        "max_span_len": 24.0,
        "waterway_adequacy": 6.0,
        "design_load": "IRC Class A",
        "scour_critical": 1,
        "num_spans": 6,
        "deck_len": 144.0,
        "roadway_width": 6.0,
        "owner": "Municipal",
        "traffic_growth_rate": 0.055,
        "collapse_year": 2019,
        "event_type": "COMPOUND",
        "event_severity": 1.20,
        "reported_condition": 8,
        "true_condition": 3.1,
    },
    {
        "bridge_id": "ARCHETYPE-MORBI-2022",
        "label": "Morbi suspension bridge (2022)",
        "state": "Gujarat",
        "district": "Morbi",
        "latitude": 22.8173,
        "longitude": 70.8377,
        "flood_prone": 0.72,
        "year_built": 1910,
        "entry_year": 1950,
        "material": "Steel",
        "design": "Suspension",
        "adt_traffic": 12_000.0,
        "truck_adt_pct": 18.0,
        "max_span_len": 140.0,
        "waterway_adequacy": 5.0,
        "design_load": "Pre-IRC",
        "scour_critical": 1,
        "num_spans": 3,
        "deck_len": 420.0,
        "roadway_width": 8.0,
        "owner": "Municipal",
        "traffic_growth_rate": 0.030,
        "collapse_year": 2022,
        "event_type": "COMPOUND",
        "event_severity": 1.10,
        "reported_condition": 8,
        "true_condition": 2.6,
    },
    {
        "bridge_id": "ARCHETYPE-BIHAR-2024",
        "label": "Darbhanga monsoon-scour bridge (2024)",
        "state": "Bihar",
        "district": "Darbhanga",
        "latitude": 26.1542,
        "longitude": 85.8918,
        "flood_prone": 0.96,
        "year_built": 1994,
        "entry_year": 1995,
        "material": "Concrete",
        "design": "Beam/Girder",
        "adt_traffic": 6_000.0,
        "truck_adt_pct": 42.0,
        "max_span_len": 36.0,
        "waterway_adequacy": 3.0,
        "design_load": "IRC Class A",
        "scour_critical": 1,
        "num_spans": 4,
        "deck_len": 144.0,
        "roadway_width": 9.5,
        "owner": "State PWD",
        "traffic_growth_rate": 0.045,
        "collapse_year": 2024,
        "event_type": "FLOOD",
        "event_severity": 1.45,
        "reported_condition": 8,
        "true_condition": 3.4,
    },
)


def archetype_bridge_ids() -> tuple[str, ...]:
    """Return the deterministic landmark-archetype bridge IDs."""

    return tuple(archetype["bridge_id"] for archetype in COLLAPSE_ARCHETYPES)


def _archetype_entry_year(
    archetype: Mapping[str, Any],
    entry_year_overrides: Mapping[str, int] | None = None,
) -> int:
    """Return the (optionally clamped) entry year for one archetype."""

    if entry_year_overrides is not None:
        bridge_id = str(archetype["bridge_id"])
        if bridge_id in entry_year_overrides:
            return int(entry_year_overrides[bridge_id])
    return int(archetype["entry_year"])


def _archetype_bridge_rows(
    entry_year_overrides: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for archetype in COLLAPSE_ARCHETYPES:
        entry_year = _archetype_entry_year(archetype, entry_year_overrides)
        duration = int(archetype["collapse_year"]) - entry_year
        inspection_year = int(archetype["collapse_year"]) - 1
        inspection_age = inspection_year - int(archetype["year_built"])
        rows.append(
            {
                ID_COL: archetype["bridge_id"],
                "STATE": archetype["state"],
                "DISTRICT": archetype["district"],
                "LATITUDE": archetype["latitude"],
                "LONGITUDE": archetype["longitude"],
                "FLOOD_PRONENESS": archetype["flood_prone"],
                "YEAR_BUILT": archetype["year_built"],
                "ENTRY_YEAR": entry_year,
                "STRUCTURE_KIND_MATERIAL": archetype["material"],
                "STRUCTURE_TYPE_DESIGN": archetype["design"],
                "ADT_TRAFFIC": archetype["adt_traffic"],
                "TRUCK_ADT_PCT": archetype["truck_adt_pct"],
                "MAX_SPAN_LEN": archetype["max_span_len"],
                "WATERWAY_ADEQUACY": archetype["waterway_adequacy"],
                "DESIGN_LOAD": archetype["design_load"],
                "SCOUR_CRITICAL": archetype["scour_critical"],
                "NUM_SPANS": archetype["num_spans"],
                "DECK_LEN": archetype["deck_len"],
                "ROADWAY_WIDTH": archetype["roadway_width"],
                "OWNER": archetype["owner"],
                "TRAFFIC_GROWTH_RATE": archetype["traffic_growth_rate"],
                DURATION_COL: duration,
                EVENT_COL: 1,
                "LAST_INSPECTION_AGE": inspection_age,
                "LAST_INSPECTION_YEAR": inspection_year,
                "EVENT_YEAR": float(archetype["collapse_year"]),
                "REPORTED_LAST_DECK_COND": archetype["reported_condition"],
                "REPORTED_LAST_SUPERSTRUCTURE_COND": archetype["reported_condition"],
                "REPORTED_LAST_SUBSTRUCTURE_COND": archetype["reported_condition"],
                "REPORTED_LAST_MIN_COND": archetype["reported_condition"],
                "LAST_INSPECTION_WAS_NDT": 0,
                "N_INSPECTIONS": 1,
            }
        )
    return pd.DataFrame(rows)


def _archetype_inspection_rows(
    entry_year_overrides: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for archetype in COLLAPSE_ARCHETYPES:
        entry_year = _archetype_entry_year(archetype, entry_year_overrides)
        inspection_year = int(archetype["collapse_year"]) - 1
        rows.append(
            {
                "BRIDGE_ID": archetype["bridge_id"],
                "INSPECTION_YEAR": inspection_year,
                "AGE_YEARS": inspection_year - int(archetype["year_built"]),
                "T_YEARS": inspection_year - entry_year,
                "REPORTED_DECK_COND": archetype["reported_condition"],
                "REPORTED_SUPERSTRUCTURE_COND": archetype["reported_condition"],
                "REPORTED_SUBSTRUCTURE_COND": archetype["reported_condition"],
                "REPORTED_MIN_COND": archetype["reported_condition"],
                "NDT_PERFORMED": 0,
                "INSPECTION_TYPE": "VISUAL",
            }
        )
    return pd.DataFrame(rows)


def _archetype_truth_rows(
    entry_year_overrides: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for archetype in COLLAPSE_ARCHETYPES:
        entry_year = _archetype_entry_year(archetype, entry_year_overrides)
        duration = int(archetype["collapse_year"]) - entry_year
        rows.append(
            {
                "BRIDGE_ID": archetype["bridge_id"],
                "PAPER_INSPECTION_FLAG": 1,
                "INSPECTOR_BIAS": 2.65,
                "LATENT_FRAILTY": 1.65,
                "MATERIAL_FACTOR": 1.0,
                "DESIGN_FACTOR": 1.0,
                "TRUE_FAILURE_AGE": float(duration),
                "TRUE_FAILURE_YEAR": float(archetype["collapse_year"]),
                "VERIFIED_EVENT": 1,
                "TRUE_LAST_DECK_COND": archetype["true_condition"],
                "TRUE_LAST_SUPERSTRUCTURE_COND": archetype["true_condition"],
                "TRUE_LAST_SUBSTRUCTURE_COND": max(
                    float(archetype["true_condition"]) - 0.5, 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def inject_collapse_archetypes(
    dataset: SyntheticDataset,
    *,
    study_end_year: int | None = None,
) -> SyntheticDataset:
    """Append the landmark collapse archetypes to a generated benchmark.

    The helper is deterministic, idempotent, and must be called **after**
    :meth:`SyntheticBridgeDataGenerator.generate`, so the seeded random stream of
    the main cohort is untouched.  The returned dataset carries the three extra
    bridges, their pre-collapse inspection records, and their evaluation-only
    ground-truth rows.

    ``BridgeDataPipeline.prepare`` routes every ``ARCHETYPE-*`` bridge into the
    test split, so none of these rows can influence imputation, scaling,
    encoding, model fitting, residual calibration, or the IsolationForest.
    """

    existing = set(dataset.bridges[ID_COL].astype(str))
    missing = [
        archetype
        for archetype in COLLAPSE_ARCHETYPES
        if archetype["bridge_id"] not in existing
    ]
    if not missing:
        return dataset

    end_year = (
        int(study_end_year)
        if study_end_year is not None
        else int(dataset.climate["YEAR"].max())
    )
    climate_years = dataset.climate.groupby("DISTRICT", observed=True)["YEAR"]
    coverage = {
        str(district): (int(years.min()), int(years.max()))
        for district, years in climate_years
    }

    entry_year_overrides: dict[str, int] = {}
    for archetype in missing:
        bridge_id = str(archetype["bridge_id"])
        entry = int(archetype["entry_year"])
        collapse_year = int(archetype["collapse_year"])
        year_built = int(archetype["year_built"])
        district = str(archetype["district"])
        if district not in coverage:
            raise ValueError(f"climate panel has no district {district!r}")
        first_year, last_year = coverage[district]

        # Vintage structures can predate the stochastic panel.  If five complete
        # pre-entry years are unavailable, clamp ENTRY_YEAR forward to the first
        # year that has a full window.  The collapse year (and therefore the
        # event) is preserved, so the at-risk duration is shortened rather than
        # producing a NaN pre-entry feature vector.
        required_start = entry - 5
        if first_year > required_start:
            clamped_entry = max(first_year + 5, year_built)
            if clamped_entry >= collapse_year:
                raise ValueError(
                    f"climate panel for {district!r} starts at {first_year}; "
                    f"cannot build a five-year pre-entry window for "
                    f"{bridge_id} before its collapse in {collapse_year}"
                )
            LOGGER.warning(
                "Clamping %s ENTRY_YEAR from %s to %s because climate coverage "
                "for %s starts in %s.",
                bridge_id,
                entry,
                clamped_entry,
                district,
                first_year,
            )
            entry_year_overrides[bridge_id] = clamped_entry
            entry = clamped_entry

        duration = collapse_year - entry
        if duration <= 0:
            raise ValueError(f"{bridge_id} has a non-positive at-risk duration")
        if entry < year_built:
            raise ValueError("archetype ENTRY_YEAR precedes YEAR_BUILT")
        if entry - 5 < first_year:
            raise ValueError(
                f"climate panel for {district!r} starts at {first_year}; five "
                f"pre-entry years are required before {entry}"
            )
        if last_year < collapse_year:
            raise ValueError(
                f"climate panel for {district!r} ends at {last_year} but "
                f"{bridge_id} collapses in {collapse_year}"
            )
        if collapse_year > end_year:
            raise ValueError(
                f"{bridge_id} collapses after study_end_year={end_year}"
            )

    bridges = pd.concat(
        [dataset.bridges, _archetype_bridge_rows(entry_year_overrides)],
        ignore_index=True,
    )
    inspections = pd.concat(
        [dataset.inspections, _archetype_inspection_rows(entry_year_overrides)],
        ignore_index=True,
    )
    ground_truth = pd.concat(
        [dataset.ground_truth, _archetype_truth_rows(entry_year_overrides)],
        ignore_index=True,
    )

    validate_bridge_frame(bridges)
    if bridges[ID_COL].duplicated().any():
        raise ValueError("archetype injection produced duplicate bridge IDs")
    if not set(archetype_bridge_ids()).issubset(set(bridges[ID_COL].astype(str))):
        raise RuntimeError("archetype injection failed to add every landmark bridge")

    return SyntheticDataset(
        bridges=bridges,
        climate=dataset.climate,
        inspections=inspections,
        ground_truth=ground_truth,
    )


class SyntheticBridgeDataGenerator:
    """Vectorised synthetic NBI + Indian climate-shock generator.

    The latent deterioration model is intentionally structural:

    ``dC_c(t) = base_c * material * design * traffic * waterway * span
                 * age(t)^0.7 * climate_c(t) * lognormal_noise``

    where ``c`` indexes deck, superstructure, and substructure.  The substructure
    has the highest hydraulic sensitivity, so scour and monsoon shocks accelerate
    substructure failure, reproducing the dominant Indian collapse mechanism.

    Visual inspection ratings are generated from the latent truth plus a
    bridge-level inspector bias.  A configurable fraction of bridges receives a
    large optimistic bias ("paper inspection"), producing the discrepancy that
    Module 3 is designed to detect.
    """

    def __init__(self, config: SyntheticDataConfig | None = None) -> None:
        self.config = config or SyntheticDataConfig()

    # -- public API --------------------------------------------------------

    def generate(self) -> SyntheticDataset:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)

        static = self._sample_bridge_registry(rng)
        climate_sim = self._simulate_climate(static, rng)
        simulation = self._simulate_deterioration(static, climate_sim, rng)

        bridges = self._assemble_bridge_table(static, simulation)
        inspections = self._assemble_inspections(static, simulation, rng)
        bridges = self._attach_last_inspection(bridges, inspections, simulation)
        bridges = self._inject_missingness(bridges, rng)
        ground_truth = self._assemble_ground_truth(static, simulation)

        self._validate_generated(bridges, climate_sim.frame, inspections)
        return SyntheticDataset(
            bridges=bridges,
            climate=climate_sim.frame,
            inspections=inspections,
            ground_truth=ground_truth,
        )

    # -- registry ---------------------------------------------------------

    def _sample_bridge_registry(self, rng: np.random.Generator) -> pd.DataFrame:
        cfg = self.config
        n = cfg.n_bridges

        weights = np.asarray([region.weight for region in REGIONS], dtype=np.float64)
        weights = weights / weights.sum()
        region_idx = rng.choice(len(REGIONS), size=n, p=weights)
        regions = [REGIONS[index] for index in region_idx]

        state = np.asarray([region.state for region in regions], dtype=object)
        district = np.asarray([region.district for region in regions], dtype=object)
        flood_prone = np.asarray([region.flood_prone for region in regions], dtype=np.float64)
        flood_prone = np.clip(flood_prone + rng.normal(0.0, 0.025, n), 0.05, 0.98)
        lat = np.asarray([region.latitude for region in regions], dtype=np.float64)
        lon = np.asarray([region.longitude for region in regions], dtype=np.float64)
        lat = lat + rng.normal(0.0, 0.22, n)
        lon = lon + rng.normal(0.0, 0.22, n)

        year_built = rng.integers(cfg.min_build_year, cfg.max_build_year + 1, size=n).astype(
            np.int16
        )
        entry_year = (year_built + 1).astype(np.int16)

        span = np.exp(rng.normal(np.log(24.0), 0.75, size=n))
        span = np.clip(span, 4.0, 320.0).astype(np.float32)

        material = self._sample_material(span, rng)
        design = self._sample_design(span, rng)

        adt = np.exp(rng.normal(np.log(9000.0), 0.78, size=n))
        adt = np.clip(adt, 300.0, 200_000.0).astype(np.float32)
        truck_pct = np.clip(rng.beta(2.1, 6.5, size=n) * 100.0, 1.0, 65.0).astype(np.float32)
        growth = np.clip(rng.normal(0.035, 0.014, size=n), 0.0, 0.090).astype(np.float32)

        waterway = rng.choice(
            np.asarray([3, 4, 5, 6, 7, 8, 9], dtype=np.float32),
            size=n,
            p=np.asarray([0.03, 0.07, 0.16, 0.24, 0.25, 0.18, 0.07]),
        ).astype(np.float32)

        num_spans = np.clip(np.round(rng.lognormal(0.75, 0.55, size=n)), 1, 30).astype(np.int16)
        deck_len = np.clip(
            span * num_spans * rng.uniform(0.80, 1.20, size=n), 5.0, 6000.0
        ).astype(np.float32)
        width = np.clip(rng.normal(12.0, 4.0, size=n), 5.0, 45.0).astype(np.float32)

        scour_critical = (
            (
                flood_prone
                + 0.22 * (9.0 - waterway) / 6.0
                + rng.normal(0.0, 0.13, size=n)
            )
            > 0.88
        ).astype(np.int8)

        owner = rng.choice(OWNER_NAMES, size=n, p=OWNER_PROBABILITIES)
        design_load = rng.choice(DESIGN_LOAD_NAMES, size=n, p=DESIGN_LOAD_PROBABILITIES)
        old_mask = year_built < 1975
        if old_mask.any():
            design_load[old_mask] = rng.choice(
                np.asarray(["Pre-IRC", "IRC Class A", "Unknown"]),
                size=int(old_mask.sum()),
                p=np.asarray([0.58, 0.27, 0.15]),
            )

        material_factor = pd.Series(material).map(MATERIAL_DEGRADATION_FACTOR).to_numpy(float)
        design_factor = pd.Series(design).map(DESIGN_DEGRADATION_FACTOR).to_numpy(float)

        frailty = np.exp(rng.normal(0.0, 0.27, size=n))
        paper_flag = rng.random(n) < cfg.paper_inspection_rate
        inspector_bias = np.where(
            paper_flag,
            rng.uniform(1.4, 3.0, size=n),
            rng.normal(0.15, 0.25, size=n),
        )
        inspection_gap = np.where(
            (scour_critical == 1) & (rng.random(n) < 0.45),
            1,
            2,
        ).astype(np.int16)

        static = pd.DataFrame(
            {
                "BRIDGE_ID": [f"BR-{index:07d}" for index in range(n)],
                "STATE": pd.Categorical(state),
                "DISTRICT": pd.Categorical(district),
                "LATITUDE": lat.astype(np.float32),
                "LONGITUDE": lon.astype(np.float32),
                "FLOOD_PRONENESS": flood_prone.astype(np.float32),
                "YEAR_BUILT": year_built,
                "ENTRY_YEAR": entry_year,
                "STRUCTURE_KIND_MATERIAL": pd.Categorical(material),
                "STRUCTURE_TYPE_DESIGN": pd.Categorical(design),
                "ADT_TRAFFIC": adt,
                "TRUCK_ADT_PCT": truck_pct,
                "MAX_SPAN_LEN": span,
                "WATERWAY_ADEQUACY": waterway,
                "DESIGN_LOAD": pd.Categorical(design_load),
                "SCOUR_CRITICAL": scour_critical,
                "NUM_SPANS": num_spans,
                "DECK_LEN": deck_len,
                "ROADWAY_WIDTH": width,
                "OWNER": pd.Categorical(owner),
                "TRAFFIC_GROWTH_RATE": growth,
                "_REGION_INDEX": region_idx.astype(np.int16),
                "_MATERIAL_FACTOR": material_factor.astype(np.float32),
                "_DESIGN_FACTOR": design_factor.astype(np.float32),
                "_FRAILTY": frailty.astype(np.float32),
                "_PAPER_FLAG": paper_flag.astype(np.int8),
                "_INSPECTOR_BIAS": inspector_bias.astype(np.float32),
                "_INSPECTION_GAP": inspection_gap,
            }
        )
        return static

    @staticmethod
    def _sample_material(span: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        material = rng.choice(
            np.asarray(list(MATERIAL_DEGRADATION_FACTOR)),
            size=span.size,
            p=np.asarray(MATERIAL_PROBABILITIES),
        )
        long_mask = span > 65.0
        if long_mask.any():
            material[long_mask] = rng.choice(
                np.asarray(["Prestressed Concrete", "Steel", "Composite"]),
                size=int(long_mask.sum()),
                p=np.asarray([0.58, 0.27, 0.15]),
            )
        short_mask = span < 9.0
        if short_mask.any():
            material[short_mask] = rng.choice(
                np.asarray(["Concrete", "Masonry", "Timber"]),
                size=int(short_mask.sum()),
                p=np.asarray([0.55, 0.30, 0.15]),
            )
        return material

    @staticmethod
    def _sample_design(span: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        design = rng.choice(
            np.asarray(list(DESIGN_DEGRADATION_FACTOR)),
            size=span.size,
            p=np.asarray(DESIGN_PROBABILITIES),
        )
        culvert_mask = span < 9.0
        design[culvert_mask] = "Culvert"

        slab_mask = (span >= 9.0) & (span < 20.0)
        if slab_mask.any():
            design[slab_mask] = rng.choice(
                np.asarray(["Slab", "Beam/Girder"]),
                size=int(slab_mask.sum()),
                p=np.asarray([0.70, 0.30]),
            )

        medium_mask = (span >= 20.0) & (span < 65.0)
        if medium_mask.any():
            design[medium_mask] = rng.choice(
                np.asarray(["Beam/Girder", "Box Girder", "Arch", "Truss"]),
                size=int(medium_mask.sum()),
                p=np.asarray([0.55, 0.25, 0.12, 0.08]),
            )

        long_mask = span >= 65.0
        if long_mask.any():
            design[long_mask] = rng.choice(
                np.asarray(["Box Girder", "Truss", "Arch", "Suspension"]),
                size=int(long_mask.sum()),
                p=np.asarray([0.35, 0.30, 0.20, 0.15]),
            )
        return design

    # -- climate ----------------------------------------------------------

    def _simulate_climate(
        self, static: pd.DataFrame, rng: np.random.Generator
    ) -> _ClimateSimulation:
        cfg = self.config
        start_year = min(
            CLIMATE_START_FLOOR_YEAR, int(static["YEAR_BUILT"].min()) - 12
        )
        years = np.arange(start_year, cfg.study_end_year + 1, dtype=np.int16)
        n_years = years.size
        n_regions = len(REGIONS)

        region_states = np.asarray([region.state for region in REGIONS], dtype=object)
        state_codes, state_names = pd.factorize(region_states, sort=True)
        n_states = state_names.size
        state_to_region = state_codes

        flood_prone = np.asarray([region.flood_prone for region in REGIONS], dtype=np.float64)
        rainfall_mean = np.asarray(
            [region.mean_rainfall_mm for region in REGIONS], dtype=np.float64
        )
        coastal = np.asarray([region.coastal for region in REGIONS], dtype=np.float64)

        extreme_year_boost = {
            1982: 0.75,
            1988: 1.15,
            1998: 0.95,
            2005: 1.35,
            2013: 1.10,
            2019: 1.45,
            2020: 1.25,
            2024: 1.60,
        }
        year_boost = np.ones(n_years, dtype=np.float64)
        for index, year in enumerate(years):
            year_boost[index] = extreme_year_boost.get(int(year), 1.0)

        # State-level AR(1) monsoon anomalies plus district-specific residuals.
        state_ar = np.zeros((n_states, n_years), dtype=np.float64)
        for t in range(1, n_years):
            state_ar[:, t] = (
                0.38 * state_ar[:, t - 1]
                + year_boost[t] * rng.normal(0.0, 0.60, size=n_states)
            )

        district_ar = np.zeros((n_regions, n_years), dtype=np.float64)
        for t in range(1, n_years):
            district_ar[:, t] = (
                0.30 * district_ar[:, t - 1]
                + rng.normal(0.0, 0.48, size=n_regions)
            )

        raw_anomaly = 0.80 * state_ar[state_to_region] + 0.45 * district_ar
        reference_mask = years <= min(2000, cfg.study_end_year)
        if reference_mask.sum() < 10:
            reference_mask = np.ones(n_years, dtype=bool)
        ref_mean = raw_anomaly[:, reference_mask].mean(axis=1, keepdims=True)
        ref_std = raw_anomaly[:, reference_mask].std(axis=1, keepdims=True)
        ref_std = np.clip(ref_std, 0.10, None)
        anomaly = (raw_anomaly - ref_mean) / ref_std

        rainfall = rainfall_mean[:, None] * np.exp(
            0.18 * anomaly
            + 0.06 * np.maximum(anomaly - 1.0, 0.0)
            + 0.020 * (years[None, :] - 2000.0) / 10.0
            + rng.normal(0.0, 0.08, size=(n_regions, n_years))
        )
        rainfall = np.clip(rainfall, 40.0, 8000.0)

        cyclone_logit = (
            -4.2
            + 0.95 * anomaly
            + 1.55 * coastal[:, None]
            + 0.55 * flood_prone[:, None]
            + 0.45 * np.maximum(anomaly - 1.2, 0.0)
        )
        cyclone_probability = 1.0 / (1.0 + np.exp(-cyclone_logit))
        cyclone = (rng.random((n_regions, n_years)) < cyclone_probability).astype(np.int8)

        log_discharge = (
            6.0
            + 0.45 * np.log(rainfall / 1000.0)
            + 0.90 * flood_prone[:, None]
            + 0.20 * coastal[:, None]
            + 0.55 * anomaly
            + 0.35 * cyclone
            + rng.normal(0.0, 0.30, size=(n_regions, n_years))
        )
        discharge = np.exp(np.clip(log_discharge, 2.0, 10.5))

        velocity = (
            0.60
            + 0.20 * np.log1p(discharge)
            + 0.20 * flood_prone[:, None]
            + 0.15 * cyclone
            + rng.normal(0.0, 0.12, size=(n_regions, n_years))
        )
        velocity = np.clip(velocity, 0.40, 8.0)
        scour = discharge * velocity / 1000.0

        threshold = np.quantile(discharge[:, reference_mask], 0.90, axis=1)
        flood_exceedance = (discharge > threshold[:, None]).astype(np.int8)

        region_index = np.repeat(np.arange(n_regions), n_years)
        year_index = np.tile(years, n_regions)

        frame = pd.DataFrame(
            {
                "STATE": pd.Categorical(region_states[region_index]),
                "DISTRICT": pd.Categorical(
                    np.asarray([region.district for region in REGIONS])[region_index]
                ),
                "YEAR": year_index.astype(np.int16),
                "MONSOON_ANOMALY": anomaly.reshape(-1).astype(np.float32),
                "SEASONAL_RAINFALL_MM": rainfall.reshape(-1).astype(np.float32),
                "PEAK_DISCHARGE_M3S": discharge.reshape(-1).astype(np.float32),
                "PEAK_VELOCITY_MPS": velocity.reshape(-1).astype(np.float32),
                "PEAK_SCOUR_INDEX": scour.reshape(-1).astype(np.float32),
                "FLOOD_EXCEEDANCE": flood_exceedance.reshape(-1).astype(np.int8),
                "CYCLONE_FLAG": cyclone.reshape(-1).astype(np.int8),
            }
        )
        return _ClimateSimulation(
            frame=frame,
            start_year=start_year,
            monsoon_anomaly=anomaly,
            scour_index=scour,
            flood_exceedance=flood_exceedance,
            cyclone_flag=cyclone,
        )

    # -- deterioration ----------------------------------------------------

    def _simulate_deterioration(
        self,
        static: pd.DataFrame,
        climate: _ClimateSimulation,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        cfg = self.config
        n = len(static)
        year_built = static["YEAR_BUILT"].to_numpy(np.int32)
        max_age = (cfg.study_end_year - year_built).astype(np.int32)
        max_age_global = int(max_age.max())
        ages = np.arange(1, max_age_global + 1, dtype=np.int16)
        calendar_years = year_built[:, None] + ages[None, :]
        valid = calendar_years <= cfg.study_end_year

        region_index = static["_REGION_INDEX"].to_numpy(np.int32)
        region_grid = np.broadcast_to(region_index[:, None], valid.shape)
        year_index = np.clip(
            calendar_years - climate.start_year, 0, climate.monsoon_anomaly.shape[1] - 1
        )

        anomaly = np.where(
            valid,
            climate.monsoon_anomaly[region_grid, year_index],
            0.0,
        )
        scour = np.where(
            valid,
            climate.scour_index[region_grid, year_index],
            0.0,
        )
        flood = np.where(
            valid,
            climate.flood_exceedance[region_grid, year_index],
            0,
        )
        cyclone = np.where(
            valid,
            climate.cyclone_flag[region_grid, year_index],
            0,
        )
        cumulative_flood = np.cumsum(flood * valid, axis=1)

        traffic_multiplier = (
            1.0
            + 0.30
            * np.log1p(static["ADT_TRAFFIC"].to_numpy(np.float64))
            / np.log1p(100_000.0)
            + 0.55 * static["TRUCK_ADT_PCT"].to_numpy(np.float64) / 100.0
        )
        waterway_multiplier = (
            1.0
            + 0.65
            * (9.0 - static["WATERWAY_ADEQUACY"].to_numpy(np.float64))
            / 6.0
        )
        span_multiplier = np.clip(
            (static["MAX_SPAN_LEN"].to_numpy(np.float64) / 30.0) ** 0.15,
            0.82,
            2.10,
        )
        bridge_multiplier = (
            static["_MATERIAL_FACTOR"].to_numpy(np.float64)
            * static["_DESIGN_FACTOR"].to_numpy(np.float64)
            * static["_FRAILTY"].to_numpy(np.float64)
            * traffic_multiplier
            * waterway_multiplier
            * span_multiplier
        )

        age_factor = ((1.0 + ages.astype(np.float64) / 55.0) ** 0.70)[None, :]
        sensitivity = np.asarray([0.22, 0.26, 0.36], dtype=np.float64)[:, None, None]
        hydraulic_exponent = (
            0.35 * np.maximum(anomaly, 0.0)
            + 0.10 * np.minimum(np.maximum(scour - 3.0, 0.0), 15.0)
            + 0.020 * cumulative_flood
            + 0.10 * cyclone
        )
        climate_multiplier = np.exp(
            np.clip(sensitivity * hydraulic_exponent[None, :, :], 0.0, 0.85)
        )

        base_rates = np.asarray([0.050, 0.045, 0.058], dtype=np.float64)[:, None, None]
        noise = np.exp(rng.normal(0.0, 0.18, size=(3, n, max_age_global)))
        decrement = (
            base_rates
            * bridge_multiplier[None, :, None]
            * age_factor[None, :, :]
            * climate_multiplier
            * noise
        )
        decrement *= valid[None, :, :]

        # Abrupt hydraulic shock damage, concentrated in the substructure.
        shock = (
            ((scour > 8.0) & (flood > 0)).astype(np.float64)
            * np.clip(
                0.08 + 0.060 * np.log1p(np.maximum(scour - 8.0, 0.0)),
                0.0,
                0.45,
            )
            * (1.0 + 0.25 * np.maximum(anomaly, 0.0))
        )
        decrement[2] += np.where(valid, shock, 0.0)

        initial = np.column_stack(
            [
                9.20 + rng.normal(0.0, 0.15, n),
                9.35 + rng.normal(0.0, 0.13, n),
                9.42 + rng.normal(0.0, 0.12, n),
            ]
        ).T[:, :, None]
        conditions = np.clip(initial - np.cumsum(decrement, axis=2), 0.0, 9.60)
        conditions = np.where(valid[None, :, :], conditions, np.nan)

        true_min = np.min(
            np.where(valid[None, :, :], conditions, np.inf), axis=0
        )
        failed = true_min <= 4.0
        has_failure = failed.any(axis=1)
        first_failed_index = np.argmax(failed, axis=1)
        failure_age = np.where(has_failure, ages[first_failed_index], np.inf).astype(np.float64)

        gap = static["_INSPECTION_GAP"].to_numpy(np.float64)
        entry_age = 1.0
        detection_age = entry_age + np.ceil((failure_age - entry_age) / gap) * gap
        event = (has_failure & (detection_age <= max_age)).astype(np.int8)
        last_visit_age = np.where(
            event == 1,
            detection_age,
            entry_age + np.floor((max_age - entry_age) / gap) * gap,
        )
        last_visit_age = np.maximum(last_visit_age, entry_age)

        inspector_bias = static["_INSPECTOR_BIAS"].to_numpy(np.float64)[:, None]
        reported_deck = np.clip(
            np.round(conditions[0] + inspector_bias + rng.normal(0.0, 0.35, (n, max_age_global))),
            0,
            9,
        )
        reported_super = np.clip(
            np.round(conditions[1] + inspector_bias + rng.normal(0.0, 0.35, (n, max_age_global))),
            0,
            9,
        )
        reported_sub = np.clip(
            np.round(conditions[2] + inspector_bias + rng.normal(0.0, 0.35, (n, max_age_global))),
            0,
            9,
        )

        return {
            "ages": ages,
            "max_age": max_age,
            "conditions": conditions,
            "reported": (reported_deck, reported_super, reported_sub),
            "failure_age": failure_age,
            "has_failure": has_failure,
            "event": event,
            "last_visit_age": last_visit_age,
            "true_min": true_min,
        }

    # -- assembly ---------------------------------------------------------

    def _assemble_bridge_table(
        self,
        static: pd.DataFrame,
        simulation: Mapping[str, Any],
    ) -> pd.DataFrame:
        output = static.drop(
            columns=[
                "_REGION_INDEX",
                "_MATERIAL_FACTOR",
                "_DESIGN_FACTOR",
                "_FRAILTY",
                "_PAPER_FLAG",
                "_INSPECTOR_BIAS",
                "_INSPECTION_GAP",
            ]
        ).copy()
        last_visit_age = simulation["last_visit_age"].astype(np.int32)
        event = np.asarray(simulation["event"], dtype=np.int8)
        year_built = output["YEAR_BUILT"].to_numpy(np.int32)

        output[DURATION_COL] = (last_visit_age - 1).astype(np.int16)
        output[EVENT_COL] = event
        output["LAST_INSPECTION_AGE"] = last_visit_age.astype(np.int16)
        output["LAST_INSPECTION_YEAR"] = (year_built + last_visit_age).astype(np.int16)
        output["EVENT_YEAR"] = np.where(
            event == 1, year_built + last_visit_age, np.nan
        ).astype(np.float32)
        return output

    def _assemble_inspections(
        self,
        static: pd.DataFrame,
        simulation: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        n = len(static)
        entry_age = 1
        gap = static["_INSPECTION_GAP"].to_numpy(np.int32)
        last_visit_age = simulation["last_visit_age"].astype(np.int32)
        counts = ((last_visit_age - entry_age) // gap + 1).astype(np.int64)
        total = int(counts.sum())
        if total <= 0:
            raise RuntimeError("synthetic generation produced no inspections")

        group_starts = np.repeat(np.cumsum(counts) - counts, counts)
        visit_offsets = np.arange(total, dtype=np.int64) - group_starts
        bridge_index = np.repeat(np.arange(n, dtype=np.int64), counts)
        ages = entry_age + visit_offsets * gap[bridge_index]
        age_index = ages.astype(np.int64) - 1

        true_deck = simulation["conditions"][0][bridge_index, age_index]
        true_super = simulation["conditions"][1][bridge_index, age_index]
        true_sub = simulation["conditions"][2][bridge_index, age_index]

        reported_deck = simulation["reported"][0][bridge_index, age_index]
        reported_super = simulation["reported"][1][bridge_index, age_index]
        reported_sub = simulation["reported"][2][bridge_index, age_index]

        ndt = rng.random(total) < self.config.ndt_rate
        if ndt.any():
            ndt_count = int(ndt.sum())
            reported_deck = reported_deck.copy()
            reported_super = reported_super.copy()
            reported_sub = reported_sub.copy()
            reported_deck[ndt] = np.clip(
                np.round(true_deck[ndt] + rng.normal(0.0, 0.15, ndt_count)), 0, 9
            )
            reported_super[ndt] = np.clip(
                np.round(true_super[ndt] + rng.normal(0.0, 0.15, ndt_count)), 0, 9
            )
            reported_sub[ndt] = np.clip(
                np.round(true_sub[ndt] + rng.normal(0.0, 0.15, ndt_count)), 0, 9
            )

        reported_min = np.minimum(np.minimum(reported_deck, reported_super), reported_sub)
        year_built = static["YEAR_BUILT"].to_numpy(np.int32)

        inspections = pd.DataFrame(
            {
                "BRIDGE_ID": static["BRIDGE_ID"].to_numpy()[bridge_index],
                "INSPECTION_YEAR": (year_built[bridge_index] + ages).astype(np.int16),
                "AGE_YEARS": ages.astype(np.int16),
                "T_YEARS": (ages - entry_age).astype(np.int16),
                "REPORTED_DECK_COND": reported_deck.astype(np.int8),
                "REPORTED_SUPERSTRUCTURE_COND": reported_super.astype(np.int8),
                "REPORTED_SUBSTRUCTURE_COND": reported_sub.astype(np.int8),
                "REPORTED_MIN_COND": reported_min.astype(np.int8),
                "NDT_PERFORMED": ndt.astype(np.int8),
                "INSPECTION_TYPE": pd.Categorical(
                    np.where(ndt, "NDT", "VISUAL")
                ),
            }
        )
        inspections = inspections.sort_values(
            ["BRIDGE_ID", "T_YEARS"], kind="mergesort"
        ).reset_index(drop=True)
        return inspections

    def _attach_last_inspection(
        self,
        bridges: pd.DataFrame,
        inspections: pd.DataFrame,
        simulation: Mapping[str, Any],
    ) -> pd.DataFrame:
        last = inspections.groupby("BRIDGE_ID", as_index=False, observed=True).tail(1)
        last = last[
            [
                "BRIDGE_ID",
                "T_YEARS",
                "AGE_YEARS",
                "REPORTED_DECK_COND",
                "REPORTED_SUPERSTRUCTURE_COND",
                "REPORTED_SUBSTRUCTURE_COND",
                "REPORTED_MIN_COND",
                "NDT_PERFORMED",
            ]
        ].rename(
            columns={
                "T_YEARS": "_LAST_T_YEARS",
                "AGE_YEARS": "_LAST_AGE_YEARS",
                "REPORTED_DECK_COND": "REPORTED_LAST_DECK_COND",
                "REPORTED_SUPERSTRUCTURE_COND": "REPORTED_LAST_SUPERSTRUCTURE_COND",
                "REPORTED_SUBSTRUCTURE_COND": "REPORTED_LAST_SUBSTRUCTURE_COND",
                "REPORTED_MIN_COND": "REPORTED_LAST_MIN_COND",
                "NDT_PERFORMED": "LAST_INSPECTION_WAS_NDT",
            }
        )
        counts = (
            inspections.groupby("BRIDGE_ID", observed=True)
            .size()
            .rename("N_INSPECTIONS")
            .reset_index()
        )
        merged = bridges.merge(last, on="BRIDGE_ID", how="left", validate="one_to_one")
        merged = merged.merge(counts, on="BRIDGE_ID", how="left", validate="one_to_one")
        if merged["_LAST_T_YEARS"].isna().any():
            raise RuntimeError("at least one bridge has no inspection record")
        if not np.array_equal(
            merged["_LAST_T_YEARS"].to_numpy(np.int64),
            merged[DURATION_COL].to_numpy(np.int64),
        ):
            raise RuntimeError("last inspection and survival duration disagree")

        merged = merged.drop(columns=["_LAST_T_YEARS", "_LAST_AGE_YEARS"])
        merged["N_INSPECTIONS"] = merged["N_INSPECTIONS"].astype(np.int16)
        merged["LAST_INSPECTION_WAS_NDT"] = merged["LAST_INSPECTION_WAS_NDT"].astype(np.int8)
        return merged

    def _inject_missingness(
        self, bridges: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        output = bridges.copy()
        scale = self.config.missing_rate_scale
        missing_profile: dict[str, float] = {
            "WATERWAY_ADEQUACY": 0.045,
            "MAX_SPAN_LEN": 0.020,
            "TRUCK_ADT_PCT": 0.050,
            "ADT_TRAFFIC": 0.040,
            "DESIGN_LOAD": 0.080,
            "DECK_LEN": 0.025,
            "ROADWAY_WIDTH": 0.030,
            "TRAFFIC_GROWTH_RATE": 0.030,
            "STRUCTURE_TYPE_DESIGN": 0.012,
            "STRUCTURE_KIND_MATERIAL": 0.006,
            "OWNER": 0.010,
        }
        n = len(output)
        for column, rate in missing_profile.items():
            effective_rate = min(rate * scale, 0.35)
            if effective_rate <= 0.0:
                continue
            mask = rng.random(n) < effective_rate
            if not mask.any():
                continue
            if isinstance(output[column].dtype, pd.CategoricalDtype):
                output[column] = output[column].cat.add_categories(["__MISSING__"])
                output.loc[mask, column] = "__MISSING__"
                output[column] = output[column].cat.remove_unused_categories()
            elif pd.api.types.is_integer_dtype(output[column]):
                output[column] = output[column].astype(np.float32)
                output.loc[mask, column] = np.nan
            else:
                output[column] = output[column].astype(np.float32)
                output.loc[mask, column] = np.nan
        return output

    def _assemble_ground_truth(
        self, static: pd.DataFrame, simulation: Mapping[str, Any]
    ) -> pd.DataFrame:
        last_age_index = simulation["last_visit_age"].astype(np.int64) - 1
        true_conditions = simulation["conditions"]
        failure_age = simulation["failure_age"].astype(np.float64)
        has_failure = np.asarray(simulation["has_failure"], dtype=bool)
        year_built = static["YEAR_BUILT"].to_numpy(np.int32)

        ground_truth = pd.DataFrame(
            {
                "BRIDGE_ID": static["BRIDGE_ID"].to_numpy(),
                "PAPER_INSPECTION_FLAG": static["_PAPER_FLAG"].to_numpy(np.int8),
                "INSPECTOR_BIAS": static["_INSPECTOR_BIAS"].to_numpy(np.float32),
                "LATENT_FRAILTY": static["_FRAILTY"].to_numpy(np.float32),
                "MATERIAL_FACTOR": static["_MATERIAL_FACTOR"].to_numpy(np.float32),
                "DESIGN_FACTOR": static["_DESIGN_FACTOR"].to_numpy(np.float32),
                "TRUE_FAILURE_AGE": np.where(has_failure, failure_age, np.nan).astype(
                    np.float32
                ),
                "TRUE_FAILURE_YEAR": np.where(
                    has_failure, year_built + failure_age, np.nan
                ).astype(np.float32),
                "VERIFIED_EVENT": simulation["event"].astype(np.int8),
                "TRUE_LAST_DECK_COND": true_conditions[0][
                    np.arange(len(static)), last_age_index
                ].astype(np.float32),
                "TRUE_LAST_SUPERSTRUCTURE_COND": true_conditions[1][
                    np.arange(len(static)), last_age_index
                ].astype(np.float32),
                "TRUE_LAST_SUBSTRUCTURE_COND": true_conditions[2][
                    np.arange(len(static)), last_age_index
                ].astype(np.float32),
            }
        )
        return ground_truth

    # -- validation -------------------------------------------------------

    @staticmethod
    def _validate_generated(
        bridges: pd.DataFrame, climate: pd.DataFrame, inspections: pd.DataFrame
    ) -> None:
        validate_bridge_frame(bridges)
        validate_climate_frame(climate)
        if inspections.empty:
            raise RuntimeError("inspection table is empty")
        if inspections[["REPORTED_DECK_COND", "REPORTED_MIN_COND"]].isna().any().any():
            raise RuntimeError("inspection ratings contain missing values")
        if not inspections["REPORTED_MIN_COND"].between(0, 9).all():
            raise RuntimeError("inspection ratings are outside the NBI 0-9 scale")
        if not bridges[EVENT_COL].isin([0, 1]).all():
            raise RuntimeError("EVENT must be binary")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_bridge_frame(bridges: pd.DataFrame) -> None:
    """Validate the bridge-level static/label frame before splitting."""

    missing = [column for column in BRIDGE_REQUIRED_COLUMNS if column not in bridges.columns]
    if missing:
        raise ValueError(f"bridge frame is missing required columns: {missing}")
    if bridges[ID_COL].duplicated().any():
        raise ValueError("BRIDGE_ID must be unique in the bridge frame")
    if bridges[ID_COL].isna().any():
        raise ValueError("BRIDGE_ID contains missing values")
    if bridges[[DURATION_COL, EVENT_COL]].isna().any().any():
        raise ValueError("survival labels must not be missing")
    if not bridges[EVENT_COL].isin([0, 1]).all():
        raise ValueError("EVENT must be binary")
    if not (bridges[DURATION_COL] > 0).all():
        raise ValueError("DURATION_YEARS must be strictly positive")
    if not (bridges["ENTRY_YEAR"] >= bridges["YEAR_BUILT"]).all():
        raise ValueError("ENTRY_YEAR must not precede YEAR_BUILT")


def validate_climate_frame(climate: pd.DataFrame) -> None:
    """Validate the district-year climate panel."""

    missing = [column for column in CLIMATE_REQUIRED_COLUMNS if column not in climate.columns]
    if missing:
        raise ValueError(f"climate frame is missing required columns: {missing}")
    if climate.duplicated(["DISTRICT", "YEAR"]).any():
        raise ValueError("climate panel contains duplicate district-year rows")
    if climate[["DISTRICT", "YEAR"]].isna().any().any():
        raise ValueError("climate panel keys contain missing values")
    if not climate["FLOOD_EXCEEDANCE"].isin([0, 1]).all():
        raise ValueError("FLOOD_EXCEEDANCE must be binary")
    if not climate["CYCLONE_FLAG"].isin([0, 1]).all():
        raise ValueError("CYCLONE_FLAG must be binary")


def assert_feature_allowlist(feature_columns: Sequence[str]) -> None:
    """Raise if a label, latent truth, or inspection field enters a model matrix."""

    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature columns contain duplicates")
    for column in feature_columns:
        if column in FORBIDDEN_FEATURE_NAMES:
            raise ValueError(f"forbidden target/identifier column in feature set: {column}")
        if column.startswith(FORBIDDEN_FEATURE_PREFIXES):
            raise ValueError(f"forbidden truth-side column in feature set: {column}")
        if column.startswith("_"):
            raise ValueError(f"private/latent column in feature set: {column}")


def assert_group_disjoint(
    train_ids: Iterable[str], validation_ids: Iterable[str], test_ids: Iterable[str]
) -> None:
    """Verify bridge-level split isolation."""

    train = set(map(str, train_ids))
    validation = set(map(str, validation_ids))
    test = set(map(str, test_ids))
    if train & validation:
        raise AssertionError("bridge leakage: train and validation IDs overlap")
    if train & test:
        raise AssertionError("bridge leakage: train and test IDs overlap")
    if validation & test:
        raise AssertionError("bridge leakage: validation and test IDs overlap")
    if not train:
        raise AssertionError("train split is empty")


def assert_counting_process(
    intervals: pd.DataFrame,
    expected_durations: Mapping[str, float] | None = None,
    *,
    id_col: str = ID_COL,
    start_col: str = T_START_COL,
    stop_col: str = T_STOP_COL,
    event_col: str = EVENT_COL,
) -> None:
    """Validate Andersen-Gill interval structure and coverage.

    The checks are deliberately strict: no overlap, no gaps, non-negative starts,
    strictly increasing stops, at most one event per bridge, and an event only in
    the final interval.
    """

    required = {id_col, start_col, stop_col, event_col}
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"counting-process frame is missing columns: {missing}")
    if intervals.empty:
        raise ValueError("counting-process frame is empty")
    if intervals[[id_col, start_col, stop_col, event_col]].isna().any().any():
        raise ValueError("counting-process frame contains missing keys/targets")
    if not intervals[event_col].isin([0, 1]).all():
        raise ValueError("event must be binary")
    if not (intervals[start_col] >= 0).all():
        raise ValueError("interval start must be non-negative")
    if not (intervals[stop_col] > intervals[start_col]).all():
        raise ValueError("zero or negative-duration interval detected")
    if intervals.duplicated([id_col, start_col]).any():
        raise ValueError("duplicate (bridge_id, t_start) interval detected")

    ordered = intervals.sort_values([id_col, start_col], kind="mergesort")
    same_id = ordered[id_col].eq(ordered[id_col].shift())
    previous_stop = ordered[stop_col].shift()
    continuity = ~same_id | np.isclose(
        ordered[start_col].to_numpy(float),
        previous_stop.to_numpy(float),
        rtol=0.0,
        atol=1e-8,
    )
    if not continuity.all():
        raise ValueError("counting-process intervals are not contiguous; gap/overlap found")

    first_rows = ordered.groupby(id_col, observed=True, sort=False).head(1)
    if not np.isclose(first_rows[start_col].to_numpy(float), 0.0, atol=1e-8).all():
        raise ValueError("every bridge must start at t_start = 0")

    event_totals = ordered.groupby(id_col, observed=True, sort=False)[event_col].sum()
    if (event_totals > 1).any():
        raise ValueError("a bridge has more than one event interval")

    last_index = ordered.groupby(id_col, observed=True, sort=False).tail(1).index
    if not ordered.loc[ordered[event_col] == 1].index.isin(last_index).all():
        raise ValueError("event is not restricted to the final interval")

    if expected_durations is not None:
        final_rows = ordered.groupby(id_col, observed=True, sort=False).tail(1).copy()
        expected = final_rows[id_col].map(expected_durations)
        if expected.isna().any():
            raise ValueError("expected_durations is missing at least one bridge")
        if not np.allclose(
            final_rows[stop_col].to_numpy(float),
            expected.to_numpy(float),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("last interval does not reach the observed survival duration")


# ---------------------------------------------------------------------------
# Andersen-Gill splitter and dynamic covariate attachment
# ---------------------------------------------------------------------------


class AndersenGillSplitter:
    """Split bridge-level survival records into counting-process intervals."""

    def __init__(self, interval_width: float = 1.0) -> None:
        if interval_width <= 0:
            raise ValueError("interval_width must be positive")
        self.interval_width = float(interval_width)

    def split(
        self,
        bridges: pd.DataFrame,
        *,
        id_col: str = ID_COL,
        duration_col: str = DURATION_COL,
        event_col: str = EVENT_COL,
    ) -> pd.DataFrame:
        validate_bridge_frame(bridges)
        start_col = T_START_COL
        stop_col = T_STOP_COL
        durations = bridges[duration_col].to_numpy(np.float64)
        counts = np.ceil(durations / self.interval_width).astype(np.int64)
        total = int(counts.sum())
        if total <= 0:
            raise ValueError("no intervals to create")

        group_starts = np.repeat(np.cumsum(counts) - counts, counts)
        interval_index = np.arange(total, dtype=np.int64) - group_starts
        repeated = bridges.iloc[np.repeat(np.arange(len(bridges)), counts)].copy()

        t_start = interval_index.astype(np.float64) * self.interval_width
        t_stop = np.minimum(
            (interval_index + 1).astype(np.float64) * self.interval_width,
            durations[np.repeat(np.arange(len(bridges)), counts)],
        )
        is_last = (interval_index + 1) == counts[np.repeat(np.arange(len(bridges)), counts)]
        event = np.where(is_last, repeated[event_col].to_numpy(np.int8), 0).astype(np.int8)

        repeated[start_col] = t_start
        repeated[stop_col] = t_stop
        repeated[event_col] = event
        repeated = repeated.sort_values([id_col, start_col], kind="mergesort").reset_index(
            drop=True
        )

        durations_map = pd.Series(
            durations, index=bridges[id_col].astype(str).to_numpy()
        ).to_dict()
        assert_counting_process(
            repeated,
            expected_durations=durations_map,
            id_col=id_col,
            start_col=start_col,
            stop_col=stop_col,
            event_col=event_col,
        )
        return repeated


def attach_dynamic_covariates(
    intervals: pd.DataFrame,
    climate: pd.DataFrame,
    *,
    interval_width: float = 1.0,
) -> pd.DataFrame:
    """Join point-in-time climate shocks and build cumulative exposure features.

    Climate is merged using the interval's *end* calendar year.  Therefore no
    future observation enters an interval covariate.  Annual intervals are
    required for a one-to-one calendar-year join.
    """

    validate_climate_frame(climate)
    if not np.isclose(interval_width, 1.0):
        raise ValueError("attach_dynamic_covariates currently requires annual intervals")

    stop = intervals[T_STOP_COL].to_numpy(np.float64)
    start = intervals[T_START_COL].to_numpy(np.float64)
    if not np.allclose(stop, np.round(stop), atol=1e-8) or not np.allclose(
        start, np.round(start), atol=1e-8
    ):
        raise ValueError("annual climate attachment requires integral interval endpoints")

    output = intervals.copy()
    output["CALENDAR_YEAR"] = (
        output["ENTRY_YEAR"].to_numpy(np.int32) + stop.astype(np.int32)
    ).astype(np.int16)

    climate_columns = [
        "DISTRICT",
        "YEAR",
        "MONSOON_ANOMALY",
        "SEASONAL_RAINFALL_MM",
        "PEAK_DISCHARGE_M3S",
        "PEAK_VELOCITY_MPS",
        "PEAK_SCOUR_INDEX",
        "FLOOD_EXCEEDANCE",
        "CYCLONE_FLAG",
    ]
    climate_small = climate[climate_columns].copy()
    climate_small["_DISTRICT_KEY"] = climate_small["DISTRICT"].astype(str)
    climate_small["_YEAR_KEY"] = climate_small["YEAR"].astype(np.int32)
    climate_small = climate_small.drop(columns=["DISTRICT", "YEAR"])

    output["_DISTRICT_KEY"] = output["DISTRICT"].astype(str)
    output["_YEAR_KEY"] = output["CALENDAR_YEAR"].astype(np.int32)
    merged = output.merge(
        climate_small,
        on=["_DISTRICT_KEY", "_YEAR_KEY"],
        how="left",
        validate="many_to_one",
    )
    climate_values = [
        "MONSOON_ANOMALY",
        "SEASONAL_RAINFALL_MM",
        "PEAK_DISCHARGE_M3S",
        "PEAK_VELOCITY_MPS",
        "PEAK_SCOUR_INDEX",
        "FLOOD_EXCEEDANCE",
        "CYCLONE_FLAG",
    ]
    if merged[climate_values].isna().any().any():
        raise ValueError("climate panel does not cover every bridge interval year")

    merged = merged.drop(columns=["_DISTRICT_KEY", "_YEAR_KEY"])
    merged["AGE_YEARS"] = (
        merged["CALENDAR_YEAR"].to_numpy(np.int32)
        - merged["YEAR_BUILT"].to_numpy(np.int32)
    ).astype(np.float32)

    years_from_entry = merged[T_STOP_COL].to_numpy(np.float32)
    merged["ADT_CURRENT"] = (
        merged["ADT_TRAFFIC"].to_numpy(np.float64)
        * np.power(
            1.0 + merged["TRAFFIC_GROWTH_RATE"].to_numpy(np.float64),
            years_from_entry,
        )
    ).astype(np.float32)
    merged["TRUCK_ADT_PCT_CURRENT"] = np.clip(
        merged["TRUCK_ADT_PCT"].to_numpy(np.float64) + 0.06 * years_from_entry,
        0.0,
        80.0,
    ).astype(np.float32)

    merged = merged.sort_values([ID_COL, T_START_COL], kind="mergesort").reset_index(drop=True)
    grouped = merged.groupby(ID_COL, observed=True, sort=False)
    merged["CUMULATIVE_FLOOD_EXCEEDANCE"] = grouped["FLOOD_EXCEEDANCE"].cumsum().astype(
        np.float32
    )
    merged["CUMULATIVE_SCOUR_LOAD"] = grouped["PEAK_SCOUR_INDEX"].cumsum().astype(np.float32)
    merged["CUMULATIVE_CYCLONE_COUNT"] = grouped["CYCLONE_FLAG"].cumsum().astype(np.float32)

    missing_features = [
        column for column in DYNAMIC_FEATURE_COLUMNS if column not in merged.columns
    ]
    if missing_features:
        raise RuntimeError(f"dynamic feature construction failed: {missing_features}")
    return merged


# ---------------------------------------------------------------------------
# Static / pre-entry survival frame
# ---------------------------------------------------------------------------


def build_static_survival_frame(bridges: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-safe bridge-level frame for RSF / GBSA.

    Only climate information known at or before entry is used.  Five-year rolling
    district aggregates ending at ``ENTRY_YEAR - 1`` are joined to each bridge.
    """

    validate_bridge_frame(bridges)
    validate_climate_frame(climate)

    ordered = climate.sort_values(["DISTRICT", "YEAR"], kind="mergesort").copy()
    ordered["_DISTRICT_KEY"] = ordered["DISTRICT"].astype(str)
    grouped = ordered.groupby("_DISTRICT_KEY", observed=True, sort=False)

    ordered["PRE_ENTRY_MONSOON_MEAN_5Y"] = grouped["MONSOON_ANOMALY"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    ordered["PRE_ENTRY_MONSOON_MAX_5Y"] = grouped["MONSOON_ANOMALY"].transform(
        lambda values: values.rolling(5, min_periods=5).max()
    )
    ordered["PRE_ENTRY_SCOUR_MAX_5Y"] = grouped["PEAK_SCOUR_INDEX"].transform(
        lambda values: values.rolling(5, min_periods=5).max()
    )
    ordered["PRE_ENTRY_FLOOD_EXCEEDANCES_5Y"] = grouped["FLOOD_EXCEEDANCE"].transform(
        lambda values: values.rolling(5, min_periods=5).sum()
    )
    ordered["PRE_ENTRY_RAINFALL_MEAN_5Y"] = grouped["SEASONAL_RAINFALL_MM"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )

    pre_entry = ordered[
        [
            "_DISTRICT_KEY",
            "YEAR",
            *PRE_ENTRY_CLIMATE_COLUMNS,
        ]
    ].rename(columns={"YEAR": "_PRE_ENTRY_YEAR"})

    bridge_work = bridges.copy()
    bridge_work["_DISTRICT_KEY"] = bridge_work["DISTRICT"].astype(str)
    bridge_work["_PRE_ENTRY_YEAR"] = bridge_work["ENTRY_YEAR"].to_numpy(np.int32) - 1
    merged = bridge_work.merge(
        pre_entry,
        on=["_DISTRICT_KEY", "_PRE_ENTRY_YEAR"],
        how="left",
        validate="many_to_one",
    )
    if merged[list(PRE_ENTRY_CLIMATE_COLUMNS)].isna().any().any():
        raise ValueError(
            "climate panel does not contain five complete pre-entry years for every bridge"
        )

    output_columns = [
        ID_COL,
        *STATIC_FEATURE_COLUMNS,
        *PRE_ENTRY_CLIMATE_COLUMNS,
        "ENTRY_YEAR",
        DURATION_COL,
        EVENT_COL,
    ]
    return merged[output_columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class BridgePreprocessor:
    """Fit-on-train iterative imputation, encoding, and numeric scaling."""

    def __init__(
        self,
        feature_columns: Sequence[str],
        *,
        random_state: int = RANDOM_STATE,
        max_iter: int = 8,
        max_fit_rows: int | None = 50_000,
        scale_numeric: bool = True,
        add_missing_indicator: bool = True,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.random_state = random_state
        self.max_iter = max_iter
        self.max_fit_rows = max_fit_rows
        self.scale_numeric = scale_numeric
        self.add_missing_indicator = add_missing_indicator

        assert_feature_allowlist(self.feature_columns)
        self.categorical_columns = tuple(
            column for column in self.feature_columns if column in CATEGORICAL_FEATURE_COLUMNS
        )
        self.numeric_columns = tuple(
            column for column in self.feature_columns if column not in self.categorical_columns
        )
        self.pipeline_: ColumnTransformer | None = None
        self.feature_names_out_: tuple[str, ...] | None = None
        self.n_fit_rows_: int | None = None

    def fit(self, frame: pd.DataFrame) -> BridgePreprocessor:
        selected = self._select_and_cast(frame)
        if self.max_fit_rows is not None and len(selected) > self.max_fit_rows:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(selected), size=self.max_fit_rows, replace=False)
            selected = selected.iloc[np.sort(indices)]

        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                IterativeImputer(
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    initial_strategy="median",
                    add_indicator=self.add_missing_indicator,
                ),
            )
        ]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(numeric_steps)

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                ),
            ]
        )

        self.pipeline_ = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, list(self.numeric_columns)),
                ("categorical", categorical_pipeline, list(self.categorical_columns)),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.pipeline_.fit(selected)
        self.feature_names_out_ = tuple(
            str(name) for name in self.pipeline_.get_feature_names_out()
        )
        assert_feature_allowlist(self.feature_names_out_)
        self.n_fit_rows_ = int(len(selected))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None or self.feature_names_out_ is None:
            raise RuntimeError("BridgePreprocessor must be fitted before transform")
        selected = self._select_and_cast(frame)
        transformed = self.pipeline_.transform(selected)
        if not np.isfinite(transformed).all():
            raise ValueError("preprocessing produced NaN or infinite values")
        return pd.DataFrame(
            transformed.astype(np.float32),
            columns=self.feature_names_out_,
            index=frame.index,
        )

    def _select_and_cast(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing feature columns: {missing}")
        selected = frame.loc[:, list(self.feature_columns)].copy()
        for column in self.categorical_columns:
            values = selected[column].astype(object)
            selected[column] = values.where(pd.notna(values), np.nan)
        for column in self.numeric_columns:
            selected[column] = pd.to_numeric(selected[column], errors="raise").astype(
                np.float64
            )
        return selected


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def build_sksurv_target(
    frame: pd.DataFrame,
    *,
    time_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> np.ndarray:
    """Build the explicit scikit-survival structured target array."""

    if time_col not in frame.columns or event_col not in frame.columns:
        raise ValueError(f"target columns {time_col!r}, {event_col!r} are required")
    times = frame[time_col].to_numpy(dtype=np.float64)
    events = frame[event_col].to_numpy()
    if not np.isfinite(times).all():
        raise ValueError("survival times must be finite")
    if not (times > 0).all():
        raise ValueError("survival times must be strictly positive")
    if not np.isin(events, [0, 1, False, True]).all():
        raise ValueError("event indicator must be binary")
    target = np.empty(times.shape[0], dtype=[("event", bool), ("time", float)])
    target["event"] = events.astype(bool)
    target["time"] = times
    return target


def build_counting_process_frame(
    intervals: pd.DataFrame,
    *,
    id_col: str = ID_COL,
    start_col: str = T_START_COL,
    stop_col: str = T_STOP_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Rename interval columns to lifelines' canonical counting-process names."""

    required = {id_col, start_col, stop_col, event_col}
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"intervals missing columns: {missing}")
    output = intervals.rename(
        columns={
            id_col: "id",
            start_col: "start",
            stop_col: "stop",
            event_col: "event",
        }
    ).copy()
    output["id"] = output["id"].astype(str)
    output["start"] = output["start"].astype(np.float64)
    output["stop"] = output["stop"].astype(np.float64)
    output["event"] = output["event"].astype(np.int8)
    return output


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Leakage-safe preprocessing configuration."""

    test_size: float = 0.20
    validation_size: float = 0.15
    random_state: int = RANDOM_STATE
    interval_width: float = 1.0
    max_fit_rows: int | None = 50_000
    imputer_max_iter: int = 8
    scale_numeric: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if not 0.0 <= self.validation_size < 1.0:
            raise ValueError("validation_size must be in [0, 1)")
        if self.test_size + self.validation_size >= 1.0:
            raise ValueError("test_size + validation_size must be below 1")
        if self.interval_width <= 0:
            raise ValueError("interval_width must be positive")


@dataclass(frozen=True)
class SurvivalDatasets:
    """Model-ready, leakage-safe survival datasets."""

    static_train: pd.DataFrame
    static_validation: pd.DataFrame
    static_test: pd.DataFrame
    counting_train: pd.DataFrame
    counting_validation: pd.DataFrame
    counting_test: pd.DataFrame
    sksurv_train: np.ndarray
    sksurv_validation: np.ndarray
    sksurv_test: np.ndarray
    static_feature_names: tuple[str, ...]
    time_varying_feature_names: tuple[str, ...]
    audit_report: dict[str, Any]
    static_preprocessor: BridgePreprocessor | None = None
    time_varying_preprocessor: BridgePreprocessor | None = None

    def save_preprocessors(self, output_dir: Path) -> None:
        """Persist the training-fitted transformers for out-of-sample scoring."""

        if self.static_preprocessor is None or self.time_varying_preprocessor is None:
            raise RuntimeError("prepared dataset does not carry fitted preprocessors")
        import joblib

        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            self.static_preprocessor, output_dir / "static_preprocessor.joblib", compress=3
        )
        joblib.dump(
            self.time_varying_preprocessor,
            output_dir / "time_varying_preprocessor.joblib",
            compress=3,
        )


class BridgeDataPipeline:
    """Orchestrate splitting, imputation, encoding, and survival targets."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.splitter = AndersenGillSplitter(self.config.interval_width)

    def prepare(
        self,
        dataset: SyntheticDataset,
        *,
        static_feature_columns: Sequence[str] = STATIC_MODEL_FEATURE_COLUMNS,
        time_varying_feature_columns: Sequence[str] = TIME_VARYING_FEATURE_COLUMNS,
    ) -> SurvivalDatasets:
        bridges = dataset.bridges
        climate = dataset.climate
        validate_bridge_frame(bridges)
        validate_climate_frame(climate)

        static_feature_columns = tuple(static_feature_columns)
        time_varying_feature_columns = tuple(time_varying_feature_columns)
        assert_feature_allowlist(static_feature_columns)
        assert_feature_allowlist(time_varying_feature_columns)

        archetype_ids = [
            str(value)
            for value in bridges[ID_COL].astype(str)
            if str(value).startswith(ARCHETYPE_ID_PREFIX)
        ]
        train_ids, validation_ids, test_ids = split_bridge_ids(
            bridges,
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=self.config.random_state,
            forced_test_ids=archetype_ids,
        )
        assert_group_disjoint(train_ids, validation_ids, test_ids)

        archetype_set = set(map(str, archetype_ids))
        train_set = set(map(str, train_ids))
        validation_set = set(map(str, validation_ids))
        test_set = set(map(str, test_ids))
        if archetype_set and not archetype_set.issubset(test_set):
            raise AssertionError(
                "collapse archetypes were not forced exclusively into the test split"
            )
        if archetype_set & (train_set | validation_set):
            raise AssertionError(
                "collapse archetypes contaminated the training or validation split"
            )

        static_frame = build_static_survival_frame(bridges, climate)
        interval_frame = self.splitter.split(bridges)
        interval_frame = attach_dynamic_covariates(
            interval_frame, climate, interval_width=self.config.interval_width
        )
        durations_map = pd.Series(
            bridges[DURATION_COL].to_numpy(np.float64),
            index=bridges[ID_COL].astype(str).to_numpy(),
        ).to_dict()
        assert_counting_process(interval_frame, expected_durations=durations_map)

        missing_static = [
            column for column in static_feature_columns if column not in static_frame.columns
        ]
        missing_dynamic = [
            column
            for column in time_varying_feature_columns
            if column not in interval_frame.columns
        ]
        if missing_static:
            raise ValueError(f"static feature columns are unavailable: {missing_static}")
        if missing_dynamic:
            raise ValueError(f"time-varying feature columns are unavailable: {missing_dynamic}")

        static_splits = _split_frame(static_frame, train_ids, validation_ids, test_ids)
        interval_splits = _split_frame(interval_frame, train_ids, validation_ids, test_ids)

        static_preprocessor = BridgePreprocessor(
            static_feature_columns,
            random_state=self.config.random_state,
            max_iter=self.config.imputer_max_iter,
            max_fit_rows=self.config.max_fit_rows,
            scale_numeric=self.config.scale_numeric,
        ).fit(static_splits["train"])
        tv_preprocessor = BridgePreprocessor(
            time_varying_feature_columns,
            random_state=self.config.random_state,
            max_iter=self.config.imputer_max_iter,
            max_fit_rows=self.config.max_fit_rows,
            scale_numeric=self.config.scale_numeric,
        ).fit(interval_splits["train"])

        static_model: dict[str, pd.DataFrame] = {}
        counting_model: dict[str, pd.DataFrame] = {}
        sksurv_targets: dict[str, np.ndarray] = {}
        for split_name in ("train", "validation", "test"):
            static_raw = static_splits[split_name]
            interval_raw = interval_splits[split_name]
            static_model[split_name] = self._assemble_static_frame(
                static_raw, static_preprocessor
            )
            counting_model[split_name] = self._assemble_counting_frame(
                interval_raw, tv_preprocessor
            )
            sksurv_targets[split_name] = build_sksurv_target(static_raw)

            if static_model[split_name].duplicated(["bridge_id"]).any():
                raise AssertionError("static model frame contains duplicate bridges")
            if (
                counting_model[split_name]
                .duplicated(["id", "start"])
                .any()
            ):
                raise AssertionError("counting-process frame contains duplicate intervals")
            if not np.isfinite(
                static_model[split_name].drop(columns=["bridge_id"]).to_numpy(np.float32)
            ).all():
                raise AssertionError("static model frame contains non-finite values")
            if not np.isfinite(
                counting_model[split_name].drop(columns=["id"]).to_numpy(np.float32)
            ).all():
                raise AssertionError("counting-process model frame contains non-finite values")

        feature_prefix_forbidden = set(FORBIDDEN_FEATURE_NAMES)
        for feature_name in (
            *static_preprocessor.feature_names_out_,
            *tv_preprocessor.feature_names_out_,
        ):
            if feature_name in feature_prefix_forbidden or feature_name.startswith(
                FORBIDDEN_FEATURE_PREFIXES
            ):
                raise AssertionError(f"leakage guard failed for feature {feature_name!r}")

        audit_report = self._build_audit_report(
            bridges=bridges,
            interval_frame=interval_frame,
            static_splits=static_splits,
            interval_splits=interval_splits,
            static_preprocessor=static_preprocessor,
            tv_preprocessor=tv_preprocessor,
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        )

        return SurvivalDatasets(
            static_train=static_model["train"],
            static_validation=static_model["validation"],
            static_test=static_model["test"],
            counting_train=counting_model["train"],
            counting_validation=counting_model["validation"],
            counting_test=counting_model["test"],
            sksurv_train=sksurv_targets["train"],
            sksurv_validation=sksurv_targets["validation"],
            sksurv_test=sksurv_targets["test"],
            static_feature_names=tuple(static_preprocessor.feature_names_out_),
            time_varying_feature_names=tuple(tv_preprocessor.feature_names_out_),
            audit_report=audit_report,
            static_preprocessor=static_preprocessor,
            time_varying_preprocessor=tv_preprocessor,
        )

    @staticmethod
    def _assemble_static_frame(
        raw: pd.DataFrame, preprocessor: BridgePreprocessor
    ) -> pd.DataFrame:
        features = preprocessor.transform(raw)
        meta = pd.DataFrame(
            {
                "bridge_id": raw[ID_COL].astype(str).to_numpy(),
                "duration": raw[DURATION_COL].to_numpy(np.float64),
                "event": raw[EVENT_COL].to_numpy(np.int8),
            }
        )
        return pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    @staticmethod
    def _assemble_counting_frame(
        raw: pd.DataFrame, preprocessor: BridgePreprocessor
    ) -> pd.DataFrame:
        features = preprocessor.transform(raw)
        meta = pd.DataFrame(
            {
                "id": raw[ID_COL].astype(str).to_numpy(),
                "start": raw[T_START_COL].to_numpy(np.float64),
                "stop": raw[T_STOP_COL].to_numpy(np.float64),
                "event": raw[EVENT_COL].to_numpy(np.int8),
            }
        )
        return pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

    @staticmethod
    def _build_audit_report(
        *,
        bridges: pd.DataFrame,
        interval_frame: pd.DataFrame,
        static_splits: Mapping[str, pd.DataFrame],
        interval_splits: Mapping[str, pd.DataFrame],
        static_preprocessor: BridgePreprocessor,
        tv_preprocessor: BridgePreprocessor,
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> dict[str, Any]:
        split_report: dict[str, Any] = {}
        for name in ("train", "validation", "test"):
            static_part = static_splits[name]
            interval_part = interval_splits[name]
            split_report[name] = {
                "n_bridges": int(static_part[ID_COL].nunique()),
                "n_events": int(static_part[EVENT_COL].sum()),
                "event_rate": float(static_part[EVENT_COL].mean()),
                "n_intervals": int(len(interval_part)),
                "n_interval_events": int(interval_part[EVENT_COL].sum()),
            }
        missingness = (
            static_splits["train"][list(static_preprocessor.feature_columns)]
            .isna()
            .mean()
            .sort_values(ascending=False)
        )
        return {
            "schema_version": 1,
            "n_bridges": int(len(bridges)),
            "n_climate_district_years": int(interval_frame["CALENDAR_YEAR"].nunique()),
            "feature_counts": {
                "static_input": len(static_preprocessor.feature_columns),
                "static_encoded": len(static_preprocessor.feature_names_out_ or ()),
                "time_varying_input": len(tv_preprocessor.feature_columns),
                "time_varying_encoded": len(tv_preprocessor.feature_names_out_ or ()),
            },
            "split": split_report,
            "train_bridge_ids": list(map(str, train_ids)),
            "validation_bridge_ids": list(map(str, validation_ids)),
            "test_bridge_ids": list(map(str, test_ids)),
            "fit_rows": {
                "static": static_preprocessor.n_fit_rows_,
                "time_varying": tv_preprocessor.n_fit_rows_,
            },
            "train_missingness_before_imputation": {
                key: float(value) for key, value in missingness.items()
            },
            "checks": {
                "bridge_split_disjoint": True,
                "feature_allowlist_enforced": True,
                "counting_process_valid": True,
                "events_only_in_final_interval": True,
                "no_future_climate_join": True,
                "no_missing_after_imputation": True,
                "sksurv_targets_constructed": True,
                "lifelines_counting_process_constructed": True,
            },
        }


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------


def split_bridge_ids(
    bridges: pd.DataFrame,
    *,
    test_size: float,
    validation_size: float,
    random_state: int = RANDOM_STATE,
    forced_test_ids: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split bridge IDs, stratified by verified event status and grouped by bridge.

    ``forced_test_ids`` are removed before the stochastic split and appended to
    the test split afterwards.  This is how the deterministic collapse
    archetypes are guaranteed to be held out from training and validation.
    """

    validate_bridge_frame(bridges)
    ids = bridges[ID_COL].astype(str).to_numpy()
    events = bridges[EVENT_COL].to_numpy(np.int8)

    forced = {str(value) for value in forced_test_ids}
    if forced:
        unknown = forced.difference(ids)
        if unknown:
            raise ValueError(
                "forced_test_ids are absent from the bridge frame: "
                f"{sorted(unknown)}"
            )
        forced_mask = np.isin(ids, list(forced))
        forced_ids = ids[forced_mask]
        pool_ids = ids[~forced_mask]
        pool_events = events[~forced_mask]
    else:
        forced_ids = np.asarray([], dtype=ids.dtype)
        pool_ids = ids
        pool_events = events

    stratify = pool_events if np.unique(pool_events).size > 1 else None

    holdout_size = test_size + validation_size
    if holdout_size >= 1.0:
        raise ValueError("test_size + validation_size must be below 1")

    train_ids, holdout_ids = train_test_split(
        pool_ids,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify,
    )
    if validation_size == 0:
        return (
            train_ids,
            np.asarray([], dtype=pool_ids.dtype),
            np.concatenate([holdout_ids, forced_ids]),
        )

    holdout_events = _lookup_events(holdout_ids, pool_ids, pool_events)
    holdout_stratify = holdout_events if np.unique(holdout_events).size > 1 else None
    test_fraction_of_holdout = test_size / holdout_size
    validation_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=test_fraction_of_holdout,
        random_state=random_state + 1,
        stratify=holdout_stratify,
    )
    return (
        train_ids,
        validation_ids,
        np.concatenate([test_ids, forced_ids]),
    )


def _lookup_events(
    subset_ids: np.ndarray, all_ids: np.ndarray, events: np.ndarray
) -> np.ndarray:
    lookup = pd.Series(events, index=all_ids)
    return lookup.loc[subset_ids].to_numpy()


def _split_frame(
    frame: pd.DataFrame,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    test_ids: Sequence[str],
) -> dict[str, pd.DataFrame]:
    id_values = frame[ID_COL].astype(str)
    splits = {
        "train": frame.loc[id_values.isin(set(map(str, train_ids)))].copy(),
        "validation": frame.loc[id_values.isin(set(map(str, validation_ids)))].copy(),
        "test": frame.loc[id_values.isin(set(map(str, test_ids)))].copy(),
    }
    if any(
        part.empty
        for name, part in splits.items()
        if name != "validation" or len(validation_ids) > 0
    ):
        raise ValueError("one or more requested data splits are empty")
    return splits


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


def save_prepared_dataset(
    dataset: SyntheticDataset, prepared: SurvivalDatasets, output_dir: Path
) -> None:
    """Persist raw synthetic tables, model-ready splits, and the audit report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(dataset.bridges, output_dir / "bridges.parquet")
    _write_frame(dataset.climate, output_dir / "climate.parquet")
    _write_frame(dataset.inspections, output_dir / "inspections.parquet")
    _write_frame(
        dataset.ground_truth, output_dir / "EVAL_ONLY_ground_truth.parquet"
    )
    for name, frame in (
        ("static_train", prepared.static_train),
        ("static_validation", prepared.static_validation),
        ("static_test", prepared.static_test),
        ("counting_train", prepared.counting_train),
        ("counting_validation", prepared.counting_validation),
        ("counting_test", prepared.counting_test),
    ):
        _write_frame(frame, output_dir / f"{name}.parquet")

    np.savez_compressed(
        output_dir / "sksurv_targets.npz",
        train=prepared.sksurv_train,
        validation=prepared.sksurv_validation,
        test=prepared.sksurv_test,
    )
    prepared.save_preprocessors(output_dir)
    with (output_dir / "pipeline_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.audit_report, handle, indent=2, default=str)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and preprocess the climate-aware bridge survival benchmark."
    )
    parser.add_argument("--n-bridges", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--study-end-year", type=int, default=2024)
    parser.add_argument("--paper-rate", type=float, default=0.18)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--max-fit-rows", type=int, default=50_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    generator = SyntheticBridgeDataGenerator(
        SyntheticDataConfig(
            n_bridges=args.n_bridges,
            study_end_year=args.study_end_year,
            paper_inspection_rate=args.paper_rate,
            random_state=args.seed,
        )
    )
    dataset = generator.generate()
    pipeline = BridgeDataPipeline(
        PipelineConfig(
            test_size=args.test_size,
            validation_size=args.validation_size,
            max_fit_rows=args.max_fit_rows,
            random_state=args.seed,
        )
    )
    prepared = pipeline.prepare(dataset)
    save_prepared_dataset(dataset, prepared, args.output_dir)
    LOGGER.info(
        "Prepared %s bridges, %s counting-process intervals, %s static features.",
        prepared.audit_report["n_bridges"],
        sum(part["n_intervals"] for part in prepared.audit_report["split"].values()),
        prepared.audit_report["feature_counts"]["static_encoded"],
    )
    LOGGER.info("Audit report wrote to %s", args.output_dir / "pipeline_audit.json")


if __name__ == "__main__":
    main()
