"""Module 1 verification tests: leakage, interval validity, and targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bridge_engine.data_pipeline import (
    DYNAMIC_FEATURE_COLUMNS,
    EVENT_COL,
    FORBIDDEN_FEATURE_NAMES,
    FORBIDDEN_FEATURE_PREFIXES,
    ID_COL,
    PRE_ENTRY_CLIMATE_COLUMNS,
    STATIC_MODEL_FEATURE_COLUMNS,
    T_START_COL,
    T_STOP_COL,
    TIME_VARYING_FEATURE_COLUMNS,
    AndersenGillSplitter,
    BridgeDataPipeline,
    BridgePreprocessor,
    PipelineConfig,
    SyntheticBridgeDataGenerator,
    SyntheticDataConfig,
    archetype_bridge_ids,
    assert_counting_process,
    assert_feature_allowlist,
    attach_dynamic_covariates,
    build_sksurv_target,
    build_static_survival_frame,
    inject_collapse_archetypes,
    split_bridge_ids,
)

N_BRIDGES = 500
SEED = 2026


@pytest.fixture(scope="module")
def benchmark():
    generator = SyntheticBridgeDataGenerator(
        SyntheticDataConfig(n_bridges=N_BRIDGES, random_state=SEED)
    )
    return generator.generate()


@pytest.fixture(scope="module")
def prepared(benchmark):
    pipeline = BridgeDataPipeline(
        PipelineConfig(
            test_size=0.20,
            validation_size=0.15,
            random_state=SEED,
            max_fit_rows=20_000,
        )
    )
    return pipeline.prepare(benchmark)


def _true_last_min(ground_truth: pd.DataFrame) -> pd.Series:
    return ground_truth[
        [
            "TRUE_LAST_DECK_COND",
            "TRUE_LAST_SUPERSTRUCTURE_COND",
            "TRUE_LAST_SUBSTRUCTURE_COND",
        ]
    ].min(axis=1)


def test_generator_produces_physically_plausible_panel(benchmark) -> None:
    bridges = benchmark.bridges
    climate = benchmark.climate
    inspections = benchmark.inspections
    truth = benchmark.ground_truth

    assert len(bridges) == N_BRIDGES
    assert bridges[ID_COL].is_unique
    assert set(bridges[EVENT_COL].unique()) == {0, 1}
    assert (bridges["DURATION_YEARS"] > 0).all()
    assert bridges["DURATION_YEARS"].max() <= 2024 - 1950
    assert bridges["EVENT"].mean() == pytest.approx(0.53, abs=0.15)

    assert not climate.duplicated(["DISTRICT", "YEAR"]).any()
    assert climate["YEAR"].min() <= 1950
    assert climate["YEAR"].max() == 2024
    assert climate["FLOOD_EXCEEDANCE"].isin([0, 1]).all()
    assert np.isfinite(climate["PEAK_SCOUR_INDEX"]).all()

    assert not inspections.duplicated(["BRIDGE_ID", "T_YEARS"]).any()
    assert inspections["REPORTED_MIN_COND"].between(0, 9).all()
    assert (inspections.groupby("BRIDGE_ID").size() > 0).all()

    merged = bridges.merge(truth, on=ID_COL, validate="one_to_one")
    paper = merged[merged["PAPER_INSPECTION_FLAG"] == 1]
    true_min = _true_last_min(merged)

    assert paper["REPORTED_LAST_MIN_COND"].mean() >= true_min[paper.index].mean() + 1.0
    anomaly_count = int(
        ((merged["REPORTED_LAST_MIN_COND"] >= 7) & (true_min <= 4.0)).sum()
    )
    assert anomaly_count > 0


def test_vintage_climate_coverage_and_archetype_test_isolation(benchmark) -> None:
    assert benchmark.climate["YEAR"].min() <= 1900
    augmented = inject_collapse_archetypes(benchmark, study_end_year=2024)
    assert set(archetype_bridge_ids()).issubset(
        set(augmented.bridges[ID_COL].astype(str))
    )

    static_frame = build_static_survival_frame(
        augmented.bridges, augmented.climate
    )
    archetype_frame = static_frame[
        static_frame[ID_COL].astype(str).isin(archetype_bridge_ids())
    ]
    assert archetype_frame[list(PRE_ENTRY_CLIMATE_COLUMNS)].notna().all().all()

    train_ids, validation_ids, test_ids = split_bridge_ids(
        augmented.bridges,
        test_size=0.20,
        validation_size=0.15,
        random_state=SEED,
        forced_test_ids=archetype_bridge_ids(),
    )
    archetypes = set(archetype_bridge_ids())
    assert archetypes.issubset(set(map(str, test_ids)))
    assert not archetypes & set(map(str, train_ids))
    assert not archetypes & set(map(str, validation_ids))


def test_truth_and_observable_tables_are_separated(benchmark) -> None:
    hidden_in_bridges = [
        column
        for column in benchmark.bridges.columns
        if column.startswith("_")
        or column.startswith("TRUE_")
        or column.startswith("LATENT_")
    ]
    assert hidden_in_bridges == []
    assert {"TRUE_LAST_DECK_COND", "PAPER_INSPECTION_FLAG"}.issubset(
        benchmark.ground_truth.columns
    )
    assert ID_COL not in STATIC_MODEL_FEATURE_COLUMNS


def test_feature_allowlist_blocks_leakage() -> None:
    assert_feature_allowlist(STATIC_MODEL_FEATURE_COLUMNS)
    assert_feature_allowlist(TIME_VARYING_FEATURE_COLUMNS)

    for leaked in (
        "REPORTED_LAST_MIN_COND",
        "TRUE_LAST_DECK_COND",
        "EVENT",
        "DURATION_YEARS",
        "_FRAILTY",
    ):
        with pytest.raises(ValueError):
            assert_feature_allowlist([*STATIC_MODEL_FEATURE_COLUMNS, leaked])

    for feature in (*STATIC_MODEL_FEATURE_COLUMNS, *TIME_VARYING_FEATURE_COLUMNS):
        assert feature not in FORBIDDEN_FEATURE_NAMES
        assert not feature.startswith(FORBIDDEN_FEATURE_PREFIXES)


def test_static_pre_entry_climate_uses_only_history(benchmark) -> None:
    static_frame = build_static_survival_frame(benchmark.bridges, benchmark.climate)
    assert set(PRE_ENTRY_CLIMATE_COLUMNS).issubset(static_frame.columns)
    assert static_frame[list(PRE_ENTRY_CLIMATE_COLUMNS)].notna().all().all()

    # Reconstruct one bridge's five pre-entry monsoon years directly from the
    # climate panel.  The join must not include ENTRY_YEAR or later observations.
    bridge = benchmark.bridges.iloc[0]
    district = bridge["DISTRICT"]
    entry_year = int(bridge["ENTRY_YEAR"])
    window = benchmark.climate[
        (benchmark.climate["DISTRICT"].astype(str) == str(district))
        & (benchmark.climate["YEAR"] >= entry_year - 5)
        & (benchmark.climate["YEAR"] <= entry_year - 1)
    ].sort_values("YEAR")
    expected = window["MONSOON_ANOMALY"].mean()
    actual = static_frame.loc[
        static_frame[ID_COL] == bridge[ID_COL], "PRE_ENTRY_MONSOON_MEAN_5Y"
    ].iloc[0]
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_andersen_gill_intervals_and_climate_join(benchmark) -> None:
    splitter = AndersenGillSplitter(interval_width=1.0)
    intervals = splitter.split(benchmark.bridges)
    intervals = attach_dynamic_covariates(intervals, benchmark.climate)

    expected_durations = pd.Series(
        benchmark.bridges["DURATION_YEARS"].to_numpy(float),
        index=benchmark.bridges[ID_COL].astype(str).to_numpy(),
    ).to_dict()
    assert_counting_process(intervals, expected_durations=expected_durations)

    # Every bridge receives exactly one interval per observed year.
    durations = benchmark.bridges.set_index(ID_COL)["DURATION_YEARS"]
    counts = intervals.groupby(ID_COL).size()
    assert (counts == durations.loc[counts.index].astype(int)).all()

    # Event status is carried only by the final interval.
    final = intervals.sort_values([ID_COL, T_START_COL]).groupby(ID_COL).tail(1)
    assert (
        final.set_index(ID_COL)[EVENT_COL]
        .sort_index()
        .equals(benchmark.bridges.set_index(ID_COL)[EVENT_COL].sort_index())
    )

    # Climate join is point-in-time and cumulative exposure is monotone.
    assert (
        intervals["CALENDAR_YEAR"]
        == intervals["ENTRY_YEAR"] + intervals[T_STOP_COL].astype(int)
    ).all()
    ordered = intervals.sort_values([ID_COL, T_START_COL])
    grouped = ordered.groupby(ID_COL, observed=True)
    expected_cumulative = grouped["FLOOD_EXCEEDANCE"].cumsum()
    assert np.allclose(
        ordered["CUMULATIVE_FLOOD_EXCEEDANCE"].to_numpy(),
        expected_cumulative.to_numpy(),
    )
    assert (grouped["CUMULATIVE_FLOOD_EXCEEDANCE"].diff().dropna() >= 0).all()

    # A directly joined climate value must be the value for that end year.
    sample = ordered.iloc[0]
    climate_row = benchmark.climate[
        (benchmark.climate["DISTRICT"].astype(str) == str(sample["DISTRICT"]))
        & (benchmark.climate["YEAR"] == sample["CALENDAR_YEAR"])
    ].iloc[0]
    assert sample["MONSOON_ANOMALY"] == pytest.approx(
        climate_row["MONSOON_ANOMALY"], rel=1e-5
    )


def test_counting_process_validation_rejects_bad_intervals(benchmark) -> None:
    splitter = AndersenGillSplitter(1.0)
    intervals = splitter.split(benchmark.bridges)

    broken = intervals.copy()
    broken.loc[broken.index[0], T_STOP_COL] = broken.loc[broken.index[0], T_START_COL]
    with pytest.raises(ValueError):
        assert_counting_process(broken)

    wrong_durations = pd.Series(
        1.0, index=benchmark.bridges[ID_COL].astype(str).to_numpy()
    ).to_dict()
    with pytest.raises(ValueError):
        assert_counting_process(intervals, expected_durations=wrong_durations)


def test_split_is_grouped_and_stratified(benchmark) -> None:
    train_ids, validation_ids, test_ids = split_bridge_ids(
        benchmark.bridges,
        test_size=0.20,
        validation_size=0.15,
        random_state=SEED,
    )
    assert len(train_ids) + len(validation_ids) + len(test_ids) == len(benchmark.bridges)
    assert not set(train_ids) & set(validation_ids)
    assert not set(train_ids) & set(test_ids)
    assert not set(validation_ids) & set(test_ids)

    events = benchmark.bridges.set_index(ID_COL)[EVENT_COL]
    rates = [events.loc[ids].mean() for ids in (train_ids, validation_ids, test_ids)]
    assert max(rates) - min(rates) < 0.08


def test_pipeline_is_leakage_safe_and_model_ready(prepared) -> None:
    assert prepared.static_train["bridge_id"].is_unique
    assert prepared.static_validation["bridge_id"].is_unique
    assert prepared.static_test["bridge_id"].is_unique
    assert not (
        set(prepared.static_train["bridge_id"])
        & set(prepared.static_validation["bridge_id"])
    )
    assert not (
        set(prepared.static_train["bridge_id"]) & set(prepared.static_test["bridge_id"])
    )
    assert not (
        set(prepared.static_validation["bridge_id"]) & set(prepared.static_test["bridge_id"])
    )

    for frame in (
        prepared.static_train,
        prepared.static_validation,
        prepared.static_test,
        prepared.counting_train,
        prepared.counting_validation,
        prepared.counting_test,
    ):
        numeric = frame.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all()

    meta_columns = {"bridge_id", "id", "duration", "event", "start", "stop"}
    for frame in (prepared.static_train, prepared.counting_train):
        feature_columns = [column for column in frame.columns if column not in meta_columns]
        assert not any(column in FORBIDDEN_FEATURE_NAMES for column in feature_columns)

    for feature in (*prepared.static_feature_names, *prepared.time_varying_feature_names):
        assert feature not in FORBIDDEN_FEATURE_NAMES
        assert not feature.startswith(FORBIDDEN_FEATURE_PREFIXES)

    assert prepared.sksurv_train.dtype == np.dtype([("event", bool), ("time", float)])
    assert prepared.sksurv_validation.dtype == prepared.sksurv_train.dtype
    assert prepared.sksurv_test.dtype == prepared.sksurv_train.dtype

    assert np.array_equal(
        prepared.sksurv_train["event"],
        prepared.static_train["event"].to_numpy(bool),
    )
    assert np.allclose(
        prepared.sksurv_train["time"], prepared.static_train["duration"].to_numpy()
    )

    assert prepared.audit_report["fit_rows"]["static"] == len(prepared.static_train)
    assert prepared.audit_report["fit_rows"]["time_varying"] == len(
        prepared.counting_train
    )
    assert all(prepared.audit_report["checks"].values())
    assert set(prepared.counting_train.columns) >= {"id", "start", "stop", "event"}


def test_preprocessor_fits_only_on_supplied_training_rows(benchmark) -> None:
    static_frame = build_static_survival_frame(benchmark.bridges, benchmark.climate)
    train, _, _ = split_bridge_ids(
        benchmark.bridges,
        test_size=0.2,
        validation_size=0.15,
        random_state=SEED,
    )
    train_frame = static_frame[static_frame[ID_COL].isin(set(train))]
    preprocessor = BridgePreprocessor(
        STATIC_MODEL_FEATURE_COLUMNS,
        max_fit_rows=1000,
        max_iter=2,
        random_state=SEED,
    ).fit(train_frame)
    assert preprocessor.n_fit_rows_ == min(1000, len(train_frame))
    transformed = preprocessor.transform(train_frame.iloc[:10])
    assert transformed.shape == (10, len(preprocessor.feature_names_out_))
    assert np.isfinite(transformed.to_numpy()).all()


def test_sksurv_target_validation() -> None:
    frame = pd.DataFrame({"duration": [1.0, 5.0, 10.0], "event": [0, 1, 1]})
    target = build_sksurv_target(frame, time_col="duration", event_col="event")
    assert target.dtype == np.dtype([("event", bool), ("time", float)])
    assert target["event"].tolist() == [False, True, True]
    assert target["time"].tolist() == [1.0, 5.0, 10.0]

    with pytest.raises(ValueError):
        build_sksurv_target(
            pd.DataFrame({"duration": [0.0], "event": [1]}),
            time_col="duration",
            event_col="event",
        )

    with pytest.raises(ValueError):
        build_sksurv_target(
            pd.DataFrame({"duration": [1.0], "event": [2]}),
            time_col="duration",
            event_col="event",
        )


def test_dynamic_feature_schema_is_complete(benchmark) -> None:
    intervals = attach_dynamic_covariates(
        AndersenGillSplitter(1.0).split(benchmark.bridges), benchmark.climate
    )
    exogenous_climate_features = [
        "MONSOON_ANOMALY",
        "SEASONAL_RAINFALL_MM",
        "PEAK_DISCHARGE_M3S",
        "PEAK_VELOCITY_MPS",
        "PEAK_SCOUR_INDEX",
        "FLOOD_EXCEEDANCE",
        "CUMULATIVE_FLOOD_EXCEEDANCE",
        "CUMULATIVE_SCOUR_LOAD",
    ]
    for column in DYNAMIC_FEATURE_COLUMNS:
        assert column in intervals.columns
    assert intervals[exogenous_climate_features].notna().all().all()
    # Traffic-derived features may legitimately be missing before imputation.
    assert intervals["ADT_CURRENT"].isna().any()
