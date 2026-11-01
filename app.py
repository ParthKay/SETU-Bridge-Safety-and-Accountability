"""Zero-Trust Bridge Survival & Audit Portal.

A dual-portal Streamlit application:

* **Field Inspector Portal** — strict blind inspection submission.
* **NHAI / State PWD Command Center** — live meteorology, revised repair
  triage, zero-trust audit, survival/SHAP deep dives and benchmark scorecards.

All model scoring is delegated to the persisted modules under
``src/bridge_engine``.  This file only orchestrates UI, caching and session
state.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bridge_engine.gui_backend import (  # noqa: E402
    CURVE_GRID,
    DECISION_HORIZON,
    FORCE_PLOT_DIR,
    RUL_HORIZON,
    SURVIVAL_THRESHOLD,
    attachment_for_bridge,
    audit_overlay_signature,
    build_audit_table,
    build_forced_event,
    build_master_queue,
    curves_for_bridge,
    district_coordinate_table,
    events_summary,
    fetch_live_weather,
    hazard_ratio_payload,
    ledger_signature,
    load_or_create_ledger,
    load_runtime_bundle,
    record_inspection_submission,
    rescore_submitted_bridge,
    safe_filename,
    scenario_event_for_bridge,
    shap_contributions,
    shocks_to_events,
    run_module4_regions,
)

st.set_page_config(
    page_title="Zero-Trust Bridge Survival & Audit Portal",
    layout="wide",
    page_icon="🌉",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
        .zt-hero {
            background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 58%, #0e7490 100%);
            color: #f8fafc;
            padding: 1.15rem 1.35rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);
        }
        .zt-hero h1 {font-size: 1.7rem; margin: 0 0 0.2rem 0; letter-spacing: -0.02em;}
        .zt-hero p {margin: 0; color: #cbd5e1; font-size: 0.93rem;}
        .zt-kicker {
            color: #38bdf8;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .zt-pill {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 0.14rem 0.26rem 0.14rem 0;
        }
        .zt-danger {background: #fee2e2; color: #991b1b; border: 1px solid #fecaca;}
        .zt-warning {background: #fef3c7; color: #92400e; border: 1px solid #fde68a;}
        .zt-success {background: #dcfce7; color: #166534; border: 1px solid #bbf7d0;}
        .zt-info {background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe;}
        .zt-neutral {background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;}
        .zt-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.75rem;
        }
        .zt-muted {color: #64748b; font-size: 0.84rem;}
        .zt-mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem;}
        .zt-section-title {
            color: #0f172a;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.1rem 0 0.5rem 0;
        }
        [data-testid="stSidebar"] {background: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached runtime / weather loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_runtime(root_str: str) -> dict[str, Any]:
    """Load persisted models and data once per Streamlit process."""

    return load_runtime_bundle(Path(root_str))


@st.cache_data(ttl=900, show_spinner=False)
def cached_weather(coordinates: pd.DataFrame, nonce: int) -> dict[str, Any]:
    """Cache live IMD / Open-Meteo calls for fifteen minutes."""

    return fetch_live_weather(coordinates)


@st.cache_data(ttl=300, show_spinner=False)
def cached_audit_table(
    root_str: str,
    ledger_token: str,
    overlay_signature: str,
) -> pd.DataFrame:
    """Reuse the full-inventory Module 3 audit table between short interactions."""

    runtime = get_runtime(root_str)
    ledger = load_or_create_ledger(
        runtime.get("bridges"), runtime.get("inspections")
    )
    return build_audit_table(runtime, ledger)


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


def fmt_mm(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.1f} mm" if np.isfinite(number) else "—"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return int(number) if np.isfinite(number) else default


def pill(text: str, kind: str) -> str:
    return f'<span class="zt-pill zt-{kind}">{text}</span>'


def priority_kind(priority: str) -> str:
    value = str(priority).upper()
    if value in {"CRITICAL", "EXTREME"}:
        return "danger"
    if value in {"HIGH", "SEVERE"}:
        return "warning"
    if value in {"MEDIUM", "HEAVY"}:
        return "info"
    return "neutral"


def render_plotly(figure: go.Figure) -> None:
    try:
        st.plotly_chart(figure, width="stretch")
    except TypeError:
        st.plotly_chart(figure, use_container_width=True)


def render_dataframe(frame: pd.DataFrame, **kwargs: Any) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)


def embed_html(html: str, height: int) -> None:
    try:
        st.iframe(html, height=height)
    except (AttributeError, TypeError):
        components.html(html, height=height, scrolling=True)


def sync_label(value: Any) -> str:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.strftime("%d %b %Y · %H:%M UTC")
    except Exception:
        return str(value) if value else "Not synced"


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def ensure_select_state(key: str, options: list[str], fallback_index: int = 0) -> None:
    """Keep a keyed selectbox valid when its options change across reruns."""

    if not options:
        return
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[min(max(fallback_index, 0), len(options) - 1)]


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "weather_nonce": 0,
        "live_weather": None,
        "last_weather_sync": "",
        "active_events": [],
        "live_triage": pd.DataFrame(),
        "live_triage_errors": [],
        "auth_selected_bridge": "",
        "inspector_receipt": None,
        "force_alert_toggle": False,
        "event_notice": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Portal A: blind field inspector submission
# ---------------------------------------------------------------------------


def render_inspector_portal(
    runtime: Mapping[str, Any],
    ledger: pd.DataFrame,
) -> None:
    bridges = runtime.get("bridges")
    if bridges is None or bridges.empty:
        st.error("The persisted bridge inventory is unavailable.")
        return

    st.markdown(
        '<div class="zt-hero">'
        '<div class="zt-kicker">Portal A · Field Operations</div>'
        '<h1>Blind Field Inspection Submission</h1>'
        '<p>Strict zero-trust assessment. Historical condition ratings, survival curves, '
        'audit scores, climate history and other bridges’ rankings are intentionally '
        'withheld from the inspecting officer.</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "Blind mode is active. Submit only what you physically observe today. "
        "The central audit engine will verify the claim independently after submission."
    )

    inventory = bridges.copy()
    inventory["BRIDGE_ID"] = inventory["BRIDGE_ID"].astype(str)
    ledger_view = ledger.copy()
    ledger_view["BRIDGE_ID"] = ledger_view["BRIDGE_ID"].astype(str)
    record = inventory.merge(
        ledger_view,
        on="BRIDGE_ID",
        how="left",
        suffixes=("", "_LEDGER"),
        validate="one_to_one",
    )
    record = record.sort_values("BRIDGE_ID", kind="mergesort").reset_index(drop=True)
    record["LABEL"] = (
        record["BRIDGE_ID"].astype(str)
        + "  —  "
        + record["DISTRICT"].astype(str)
        + ", "
        + record["STATE"].astype(str)
        + "  ("
        + record["MANAGING_AGENCY"].fillna("Unassigned").astype(str)
        + ")"
    )
    label_lookup = dict(zip(record["BRIDGE_ID"], record["LABEL"], strict=False))
    options = record["BRIDGE_ID"].tolist()

    selected = st.selectbox(
        "Select assigned bridge",
        options,
        format_func=lambda value: label_lookup.get(value, value),
        key="inspector_bridge_select",
    )
    chosen = record.loc[record["BRIDGE_ID"].eq(selected)].iloc[0]
    st.markdown(
        f'<div class="zt-card">'
        f'<div class="zt-section-title">Assigned Bridge</div>'
        f'<b>{selected}</b> — {chosen["DISTRICT"]}, {chosen["STATE"]}<br>'
        f'<span class="zt-muted">Managing agency: <b>{chosen["MANAGING_AGENCY"]}</b> · '
        f'Current inspector: <b>{chosen["CURRENT_INSPECTOR_NAME"]}</b> '
        f'({chosen["CURRENT_INSPECTOR_ID"]})</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("blind_inspection_form", clear_on_submit=False):
        st.markdown("#### Today's Field Assessment")
        overall = st.slider(
            "Reported Condition Rating (0 = Failed, 9 = Excellent)",
            min_value=0,
            max_value=9,
            value=5,
            step=1,
            key="insp_overall_condition",
        )
        component_mode = st.checkbox(
            "Provide component-level condition breakdown",
            key="insp_component_mode",
        )
        if component_mode:
            columns = st.columns(3)
            with columns[0]:
                deck = st.slider("Deck", 0, 9, int(overall), key="insp_deck_cond")
            with columns[1]:
                superstructure = st.slider(
                    "Superstructure", 0, 9, int(overall), key="insp_super_cond"
                )
            with columns[2]:
                substructure = st.slider(
                    "Substructure", 0, 9, int(overall), key="insp_sub_cond"
                )
            reported = int(min(deck, superstructure, substructure))
            st.caption(
                f"REPORTED_LAST_MIN_COND is set by the minimum component: **{reported}**"
            )
        else:
            deck = superstructure = substructure = int(overall)
            reported = int(overall)

        method = st.radio(
            "Inspection Method",
            ("Visual Only", "Physical NDT Performed"),
            horizontal=True,
            key="insp_method",
        )
        remarks = st.text_area(
            "Inspector Remarks",
            placeholder=(
                "Record visible cracks, spalling, scour, bearing condition, drainage, "
                "monsoon damage, or maintenance actions required."
            ),
            height=150,
            key="insp_remarks",
        )
        upload = st.file_uploader(
            "Working report attachment",
            type=["pdf", "png", "jpg", "jpeg", "docx"],
            key="insp_upload",
            help="A PDF, image or Word inspection report will be stored with this submission.",
        )
        submitted = st.form_submit_button(
            "Submit blind inspection to central verification",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            upload_bytes = upload.getvalue() if upload is not None else None
            upload_name = upload.name if upload is not None else None
            updated_ledger, submission = record_inspection_submission(
                ledger,
                bridge_id=str(selected),
                reported_condition=reported,
                deck_condition=int(deck),
                superstructure_condition=int(superstructure),
                substructure_condition=int(substructure),
                inspection_method=str(method),
                remarks=str(remarks),
                upload_name=upload_name,
                upload_bytes=upload_bytes,
                root=ROOT,
            )
            audit_note = "central audit re-score completed"
            try:
                rescore_submitted_bridge(
                    runtime,
                    bridge_id=str(selected),
                    reported_condition=reported,
                    inspection_method=str(method),
                    receipt=str(submission.get("RECEIPT_SHA256", "")),
                )
            except Exception as exc:
                audit_note = f"queued for retry ({type(exc).__name__})"
                runtime.setdefault("errors", []).append(
                    f"live audit re-score {selected}: {type(exc).__name__}: {exc}"
                )
            submission["AUDIT_NOTE"] = audit_note
            st.session_state["inspector_receipt"] = submission
            st.rerun()
        except Exception as exc:
            st.error(f"Submission failed: {type(exc).__name__}: {exc}")

    receipt = st.session_state.get("inspector_receipt")
    if receipt:
        st.success(
            "Inspection logged and encrypted for central NHAI/PWD verification."
        )
        st.markdown(
            f'<div class="zt-card">'
            f'<div class="zt-section-title">Submission Receipt</div>'
            f'Bridge: <b>{receipt.get("BRIDGE_ID")}</b><br>'
            f'Logged at: <b>{sync_label(receipt.get("SUBMITTED_AT"))}</b><br>'
            f'Reported condition: <b>{receipt.get("REPORTED_LAST_MIN_COND")} / 9</b><br>'
            f'Verification digest: <span class="zt-mono">{receipt.get("RECEIPT_SHA256")}</span><br>'
            f'<span class="zt-muted">{receipt.get("AUDIT_NOTE", "")}. '
            f'Audit outcomes are withheld from this portal.</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Dismiss receipt", key="dismiss_inspector_receipt"):
            st.session_state["inspector_receipt"] = None
            st.rerun()


# ---------------------------------------------------------------------------
# Weather sync and authority state
# ---------------------------------------------------------------------------


def sync_weather(runtime: Mapping[str, Any]) -> dict[str, Any]:
    coordinates = district_coordinate_table(runtime.get("bridges", pd.DataFrame()))
    nonce = int(st.session_state.get("weather_nonce", 0))
    with st.spinner("Syncing live IMD bulletins and Open-Meteo district forecasts..."):
        weather = cached_weather(coordinates, nonce)
    st.session_state["live_weather"] = weather

    if weather.get("synced_at") != st.session_state.get("last_weather_sync"):
        st.session_state["last_weather_sync"] = weather.get("synced_at", "")
        forced = [
            event
            for event in st.session_state.get("active_events", [])
            if str(getattr(event, "event_id", "")).startswith("FORCED-")
        ]
        automatic = shocks_to_events(weather.get("districts", pd.DataFrame()))
        forced_districts = {
            str(getattr(event, "district", ""))
            for event in forced
            if getattr(event, "district", None) is not None
        }
        if forced_districts:
            automatic = [
                event
                for event in automatic
                if str(getattr(event, "district", "")) not in forced_districts
            ]
        st.session_state["active_events"] = forced + automatic
        if automatic:
            with st.spinner(
                "Live flood threshold exceeded. Running Module 4 triage on affected regions..."
            ):
                live, errors = run_module4_regions(
                    runtime, automatic, max_regions=3, top_n_explanations=2
                )
            st.session_state["live_triage"] = live
            st.session_state["live_triage_errors"] = errors
            st.session_state["event_notice"] = (
                f"Auto-triggered Module 4 for {len(automatic)} live affected district(s); "
                f"{len(live)} held-out bridges received SHAP-backed triage."
            )
        elif not forced:
            st.session_state["live_triage"] = pd.DataFrame()
            st.session_state["live_triage_errors"] = []
            st.session_state["event_notice"] = ""
    return weather


def _weather_bar(weather: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    districts = weather.get("districts", pd.DataFrame())
    st.markdown("#### Live Meteorological & Flash-Flood Status")
    if weather.get("offline"):
        st.warning(
            "Live network feeds are currently unavailable (offline environment or "
            "government firewall block). The portal is operating with a safe fallback; "
            "no automatic shock has been applied."
        )
    controls = st.columns([1, 1, 1, 1.4])
    controls[0].metric("Last IMD / Open-Meteo Sync", sync_label(weather.get("synced_at")))
    if districts is not None and not districts.empty:
        alerts = int(districts["EXCEEDS_THRESHOLD"].astype(bool).sum())
        severe = int(districts["ALERT_LEVEL"].isin(["SEVERE", "EXTREME"]).sum())
        total = int(len(districts))
        bridge_counts = districts.get("N_BRIDGES", pd.Series(0, index=districts.index))
        n_bridges = int(pd.to_numeric(bridge_counts, errors="coerce").fillna(0).sum())
    else:
        alerts = severe = total = n_bridges = 0
    controls[1].metric("Active Flood Alerts", f"{alerts}")
    controls[2].metric("Severe / Extreme", f"{severe}")
    controls[3].metric("Districts / Bridges Monitored", f"{total} / {n_bridges:,}")

    button_columns = st.columns([1, 1, 2.4])
    if button_columns[0].button(
        "Refresh live weather",
        key="refresh_weather",
        use_container_width=True,
    ):
        st.session_state["weather_nonce"] = int(st.session_state.get("weather_nonce", 0)) + 1
        st.rerun()
    if button_columns[1].button(
        "Clear live shock overlay",
        key="clear_live_shocks",
        use_container_width=True,
    ):
        st.session_state["active_events"] = []
        st.session_state["live_triage"] = pd.DataFrame()
        st.session_state["live_triage_errors"] = []
        st.session_state["event_notice"] = "Live shock overlay cleared."
        st.rerun()
    with button_columns[2]:
        active = st.session_state.get("active_events", [])
        if active:
            st.caption(f"Active Module 4 shocks: **{events_summary(active)}**")
        else:
            st.caption("No live shock overlay is currently applied to the repair queue.")

    if districts is not None and not districts.empty:
        display = districts.copy()
        display["RAINFALL_24H_MM"] = display["RAINFALL_24H_MM"].map(fmt_mm)
        display["RAINFALL_72H_MM"] = display["RAINFALL_72H_MM"].map(fmt_mm)
        display["STATUS"] = np.where(
            display["EXCEEDS_THRESHOLD"].astype(bool),
            display["ALERT_LEVEL"],
            "NORMAL",
        )
        columns = [
            "STATE",
            "DISTRICT",
            "N_BRIDGES",
            "RAINFALL_24H_MM",
            "RAINFALL_72H_MM",
            "WEATHER_LABEL",
            "IMD_LEVEL",
            "STATUS",
            "SOURCE",
            "IMD_HEADLINE",
        ]
        columns = [column for column in columns if column in display.columns]
        render_dataframe(display.loc[:, columns])
        with st.expander("State-wide IMD warning bulletins", expanded=False):
            bulletins = weather.get("bulletins", [])
            if bulletins:
                for bulletin in bulletins[:20]:
                    st.markdown(
                        f"{pill(bulletin.get('level', 'INFO'), priority_kind(bulletin.get('level', 'INFO')))} "
                        f"**{bulletin.get('state', 'India')}** — {bulletin.get('headline', '')}",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No parsable state warning bulletins were returned by IMD.")

    st.divider()
    st.markdown("#### Authority Flash-Flood Trigger")
    coordinates = district_coordinate_table(runtime.get("bridges", pd.DataFrame()))
    force = st.toggle(
        "Force Live IMD Flash-Flood Alert",
        key="force_alert_toggle",
        help=(
            "Apply a governed module-4 shock even when the live feed is unavailable "
            "or below threshold."
        ),
    )
    if force and not coordinates.empty:
        state_options = sorted(coordinates["STATE"].astype(str).unique().tolist())
        selector = st.columns([1, 1, 0.9, 1.1])
        state_choice = selector[0].selectbox(
            "State", state_options, key="force_state_select"
        )
        district_options = sorted(
            coordinates.loc[coordinates["STATE"].astype(str).eq(state_choice), "DISTRICT"]
            .astype(str)
            .unique()
            .tolist()
        )
        ensure_select_state("force_district_select", district_options)
        district_choice = selector[1].selectbox(
            "District", district_options, key="force_district_select"
        )
        severity = selector[2].slider(
            "Shock severity", 1.0, 2.2, 1.65, 0.05, key="force_severity"
        )
        if selector[3].button(
            "Trigger alert & run Module 4",
            key="force_trigger_alert",
            type="primary",
            use_container_width=True,
        ):
            event = build_forced_event(
                state=str(state_choice),
                district=str(district_choice),
                severity=float(severity),
            )
            active = [
                existing
                for existing in st.session_state.get("active_events", [])
                if str(getattr(existing, "district", "")) != str(district_choice)
            ]
            active.append(event)
            st.session_state["active_events"] = active
            with st.spinner("Running Module 4 on every bridge in the alerted district..."):
                live, errors = run_module4_regions(
                    runtime, [event], max_regions=1, top_n_explanations=3
                )
            if live is not None and not live.empty:
                existing_live = st.session_state.get("live_triage", pd.DataFrame())
                combined = (
                    pd.concat([existing_live, live], ignore_index=True)
                    if existing_live is not None and not existing_live.empty
                    else live
                )
                combined = combined.drop_duplicates("BRIDGE_ID", keep="first")
                st.session_state["live_triage"] = combined
            st.session_state["live_triage_errors"] = errors
            st.session_state["event_notice"] = (
                f"Forced flash-flood alert applied to {district_choice}, {state_choice}. "
                "The repair queue and RUL rankings have been revised."
            )
            st.rerun()
    elif force:
        st.caption("No district coordinates are available for a forced alert.")

    notice = st.session_state.get("event_notice")
    if notice:
        st.success(notice)
    errors = st.session_state.get("live_triage_errors", [])
    if errors:
        with st.expander("Module 4 regional triage diagnostics", expanded=False):
            for error in errors:
                st.warning(error)


# ---------------------------------------------------------------------------
# Portal B · Sub-tab 1: live feed and revised repair queue
# ---------------------------------------------------------------------------


def _queue_display(queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        return queue
    display = pd.DataFrame(
        {
            "Rank": queue["RANK"],
            "Bridge ID": queue["BRIDGE_ID"],
            "State / District": (
                queue["STATE"].fillna("Unknown").astype(str)
                + " / "
                + queue["DISTRICT"].fillna("Unknown").astype(str)
            ),
            "Agency": queue["MANAGING_AGENCY"].fillna("Unassigned"),
            "Triage Priority": queue["TRIAGE_PRIORITY"],
            "Immediate NDT Required": np.where(
                queue["IMMEDIATE_NDT_DEPLOYMENT"].astype(bool), "YES", "NO"
            ),
            "5-Year Survival S(t0+5|t0)": queue["SURVIVAL_5Y"].map(lambda value: fmt(value, 3)),
            "Post-Event RUL (Years)": queue["POST_EVENT_RUL_YEARS"].map(lambda value: fmt(value, 2)),
            "RUL Loss (Years)": queue["RUL_LOSS_YEARS"].map(lambda value: fmt(value, 2)),
            "Last Inspection Date": queue["LAST_INSPECTION_DATE"].fillna("—"),
            "Current Inspector & Phone": (
                queue["CURRENT_INSPECTOR_NAME"].fillna("—").astype(str)
                + " · "
                + queue["CURRENT_INSPECTOR_PHONE"].fillna("—").astype(str)
            ),
            "Previous Inspector & Phone": (
                queue["PREVIOUS_INSPECTOR_NAME"].fillna("—").astype(str)
                + " · "
                + queue["PREVIOUS_INSPECTOR_PHONE"].fillna("—").astype(str)
            ),
            "Priority Score": queue["TRIAGE_PRIORITY_SCORE"].map(lambda value: fmt(value, 4)),
        }
    )
    return display


def render_live_queue_tab(
    runtime: Mapping[str, Any],
    ledger: pd.DataFrame,
    queue: pd.DataFrame,
    weather: Mapping[str, Any],
) -> None:
    _weather_bar(weather, runtime)

    st.divider()
    st.markdown("#### Master Revised Bridge Repair Queue")
    if queue.empty:
        st.error("The revised repair queue could not be computed from the persisted artefacts.")
        return

    top = st.columns(4)
    top[0].metric("Bridges Ranked", f"{len(queue):,}")
    top[1].metric(
        "CRITICAL / HIGH",
        f"{int(queue['TRIAGE_PRIORITY'].eq('CRITICAL').sum()):,} / "
        f"{int(queue['TRIAGE_PRIORITY'].eq('HIGH').sum()):,}",
    )
    top[2].metric(
        "Immediate NDT",
        f"{int(queue['IMMEDIATE_NDT_DEPLOYMENT'].astype(bool).sum()):,}",
    )
    top[3].metric(
        "Total Net RUL Loss",
        f"{fmt(pd.to_numeric(queue['RUL_LOSS_YEARS'], errors='coerce').sum(), 1)} years",
    )

    filter_columns = st.columns(4)
    states = ["All"] + sorted(queue["STATE"].dropna().astype(str).unique().tolist())
    districts = ["All"] + sorted(queue["DISTRICT"].dropna().astype(str).unique().tolist())
    agencies = ["All"] + sorted(queue["MANAGING_AGENCY"].dropna().astype(str).unique().tolist())
    priorities = ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    state_filter = filter_columns[0].selectbox("State", states, key="queue_state_filter")
    district_filter = filter_columns[1].selectbox("District", districts, key="queue_district_filter")
    agency_filter = filter_columns[2].selectbox("Managing Agency", agencies, key="queue_agency_filter")
    priority_filter = filter_columns[3].selectbox("Priority Level", priorities, key="queue_priority_filter")

    filtered = queue.copy()
    if state_filter != "All":
        filtered = filtered.loc[filtered["STATE"].astype(str).eq(state_filter)]
    if district_filter != "All":
        filtered = filtered.loc[filtered["DISTRICT"].astype(str).eq(district_filter)]
    if agency_filter != "All":
        filtered = filtered.loc[filtered["MANAGING_AGENCY"].astype(str).eq(agency_filter)]
    if priority_filter != "All":
        filtered = filtered.loc[filtered["TRIAGE_PRIORITY"].astype(str).eq(priority_filter)]

    display = _queue_display(filtered)
    render_dataframe(display, height=520)

    if filtered.empty:
        st.caption("No bridges match the selected filters.")
        return

    selection_columns = st.columns([2.4, 1])
    queue_options = filtered["BRIDGE_ID"].astype(str).tolist()
    ensure_select_state("queue_bridge_select", queue_options)
    selected = selection_columns[0].selectbox(
        "Inspect a bridge from the queue",
        queue_options,
        key="queue_bridge_select",
    )
    st.session_state["auth_selected_bridge"] = str(selected)
    if selection_columns[1].button(
        "Open in deep dive",
        key="queue_open_deep_dive",
        use_container_width=True,
    ):
        st.session_state["deep_dive_bridge"] = str(selected)
        st.rerun()

    detail = filtered.loc[filtered["BRIDGE_ID"].astype(str).eq(str(selected))].iloc[0]
    st.markdown(
        " ".join(
            [
                pill(f"Triage: {detail['TRIAGE_PRIORITY']}", priority_kind(detail["TRIAGE_PRIORITY"])),
                pill(
                    f"Immediate NDT: {'YES' if bool(detail['IMMEDIATE_NDT_DEPLOYMENT']) else 'NO'}",
                    "danger" if bool(detail["IMMEDIATE_NDT_DEPLOYMENT"]) else "success",
                ),
                pill(f"Audit: {detail.get('AUDIT_PRIORITY', 'NOT_AUDITED')}", priority_kind(detail.get("AUDIT_PRIORITY", ""))),
                pill(
                    f"Paper suspect: {bool(detail.get('PAPER_INSPECTION_SUSPECT', False))}",
                    "danger" if bool(detail.get("PAPER_INSPECTION_SUSPECT", False)) else "success",
                ),
                pill(
                    f"Live Module 4: {'YES' if bool(detail.get('LIVE_TRIAGE_APPLIED', False)) else 'BASELINE'}",
                    "info" if bool(detail.get("LIVE_TRIAGE_APPLIED", False)) else "neutral",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Portal B · Sub-tab 2: zero-trust vigilance and accountability
# ---------------------------------------------------------------------------


def _audit_display(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    return pd.DataFrame(
        {
            "Rank": audit["AUDIT_QUEUE_RANK"],
            "Bridge ID": audit["BRIDGE_ID"],
            "State / District": (
                audit["STATE"].fillna("Unknown").astype(str)
                + " / "
                + audit["DISTRICT"].fillna("Unknown").astype(str)
            ),
            "Agency": audit["MANAGING_AGENCY"].fillna("Unassigned"),
            "Claimed Rating (0–9)": audit["REPORTED_LAST_MIN_COND"],
            "Physics-Implied Condition": audit["PHYSICS_EXPECTED_CONDITION"].map(
                lambda value: fmt(value, 2)
            ),
            "Condition Gap": audit["CONDITION_GAP"].map(lambda value: fmt(value, 2)),
            "Audit Z-Score": audit["AUDIT_Z_SCORE"].map(lambda value: fmt(value, 2)),
            "IsolationForest": np.where(
                audit["ISOLATION_FLAG"].astype(bool), "ANOMALY", "NORMAL"
            ),
            "Paper Inspection Suspect": np.where(
                audit["PAPER_INSPECTION_SUSPECT"].astype(bool), "FLAGGED", "—"
            ),
            "Audit Alert": np.where(
                audit["AUDIT_ALERT"].astype(bool), "ALERT", "—"
            ),
            "Audit Priority": audit["AUDIT_PRIORITY"],
            "Last Inspection": audit["LAST_INSPECTION_DATE"].fillna("—"),
            "Current Inspector": audit["CURRENT_INSPECTOR_NAME"].fillna("—"),
        }
    )


def _style_audit_rows(row: pd.Series) -> list[str]:
    if row.get("Paper Inspection Suspect") == "FLAGGED" or row.get("Audit Alert") == "ALERT":
        return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
    if row.get("IsolationForest") == "ANOMALY":
        return ["background-color: #fef3c7; color: #78350f"] * len(row)
    return [""] * len(row)


def _render_accountability_card(
    runtime: Mapping[str, Any],
    ledger: pd.DataFrame,
    audit_row: Mapping[str, Any],
) -> None:
    bridge_id = str(audit_row.get("BRIDGE_ID", ""))
    st.markdown("#### Inspector Accountability Card")
    card = st.container(border=True)
    with card:
        st.markdown(f"##### {bridge_id}")
        st.caption(
            f"{audit_row.get('DISTRICT', 'Unknown')}, {audit_row.get('STATE', 'Unknown')}  ·  "
            f"{audit_row.get('MANAGING_AGENCY', 'Unassigned')}"
        )
        top = st.columns(4)
        top[0].metric("Last Inspection Date", str(audit_row.get("LAST_INSPECTION_DATE", "—")))
        top[1].metric("Claimed Rating", f"{int(audit_row.get('REPORTED_LAST_MIN_COND', 0))} / 9")
        top[2].metric("Physics-Implied", f"{fmt(audit_row.get('PHYSICS_EXPECTED_CONDITION'), 2)} / 9")
        top[3].metric("Audit Z-Score", fmt(audit_row.get("AUDIT_Z_SCORE"), 2))

        current, previous = st.columns(2)
        with current:
            st.markdown("**Current Inspector**")
            st.markdown(
                f"**{audit_row.get('CURRENT_INSPECTOR_NAME', 'Not recorded')}**  \n"
                f"ID: `{audit_row.get('CURRENT_INSPECTOR_ID', '—')}`  \n"
                f"Phone: `{audit_row.get('CURRENT_INSPECTOR_PHONE', '—')}`  \n"
                f"Email: `{audit_row.get('CURRENT_INSPECTOR_EMAIL', '—')}`"
            )
        with previous:
            st.markdown("**Previous Inspector**")
            st.markdown(
                f"**{audit_row.get('PREVIOUS_INSPECTOR_NAME', 'Not recorded')}**  \n"
                f"ID: `{audit_row.get('PREVIOUS_INSPECTOR_ID', '—')}`  \n"
                f"Phone: `{audit_row.get('PREVIOUS_INSPECTOR_PHONE', '—')}`  \n"
                f"Email: `{audit_row.get('PREVIOUS_INSPECTOR_EMAIL', '—')}`"
            )

        st.markdown("**Exact Inspector Remarks**")
        st.info(str(audit_row.get("INSPECTOR_REMARKS", "No remarks recorded.")))
        st.markdown(
            " ".join(
                [
                    pill(
                        f"PAPER_INSPECTION_SUSPECT = {bool(audit_row.get('PAPER_INSPECTION_SUSPECT', False))}",
                        "danger" if bool(audit_row.get("PAPER_INSPECTION_SUSPECT", False)) else "success",
                    ),
                    pill(
                        f"AUDIT_ALERT = {bool(audit_row.get('AUDIT_ALERT', False))}",
                        "danger" if bool(audit_row.get("AUDIT_ALERT", False)) else "success",
                    ),
                    pill(
                        f"ISOLATION_ANOMALY = {bool(audit_row.get('ISOLATION_FLAG', False))}",
                        "warning" if bool(audit_row.get("ISOLATION_FLAG", False)) else "neutral",
                    ),
                    pill(
                        f"Submissions: {safe_int(audit_row.get('SUBMISSION_COUNT', 0))}",
                        "info",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )

        path, generated = attachment_for_bridge(runtime, ledger, bridge_id)
        if path is None or not path.exists():
            st.warning("No uploaded or generated inspection attachment is available.")
        else:
            try:
                payload = path.read_bytes()
                st.download_button(
                    "Download inspection report / signed receipt",
                    data=payload,
                    file_name=path.name,
                    mime=_mime_for(path),
                    key=f"download_inspection_{safe_filename(bridge_id)}",
                    use_container_width=True,
                )
                st.caption(
                    ("Generated default inspection receipt" if generated else "Uploaded inspection attachment")
                    + f" · `{path}`"
                )
            except Exception as exc:
                st.error(f"Attachment could not be read: {type(exc).__name__}: {exc}")


def render_audit_tab(
    runtime: Mapping[str, Any],
    ledger: pd.DataFrame,
    audit_table: pd.DataFrame,
) -> None:
    st.markdown("#### Zero-Trust Vigilance & Paper-Inspection Audit Queue")
    st.caption(
        "Ranked by Audit Z-Score. A PAPER_INSPECTION_SUSPECT is a reported condition "
        "of 7–9 while physics-implied five-year survival is below τ = 0.68."
    )
    if audit_table.empty:
        st.error("No audit findings are available.")
        return

    flag_columns = st.columns(4)
    flagged_only = flag_columns[0].toggle("Flagged bridges only", value=False, key="audit_flagged_only")
    state_filter = flag_columns[1].selectbox(
        "State",
        ["All"] + sorted(audit_table["STATE"].dropna().astype(str).unique().tolist()),
        key="audit_state_filter",
    )
    priority_filter = flag_columns[2].selectbox(
        "Audit Priority",
        ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
        key="audit_priority_filter",
    )
    search = flag_columns[3].text_input(
        "Search Bridge ID", placeholder="e.g. ARCHETYPE-CSMT-2019", key="audit_search"
    )

    filtered = audit_table.copy()
    if flagged_only:
        flagged = (
            filtered["PAPER_INSPECTION_SUSPECT"].astype(bool)
            | filtered["AUDIT_ALERT"].astype(bool)
            | filtered["ISOLATION_FLAG"].astype(bool)
        )
        filtered = filtered.loc[flagged]
    if state_filter != "All":
        filtered = filtered.loc[filtered["STATE"].astype(str).eq(state_filter)]
    if priority_filter != "All":
        filtered = filtered.loc[filtered["AUDIT_PRIORITY"].astype(str).eq(priority_filter)]
    if search.strip():
        filtered = filtered.loc[
            filtered["BRIDGE_ID"].astype(str).str.contains(search.strip(), case=False, regex=False)
        ]
    if filtered.empty:
        st.info("No audit rows match the current filters.")
        return

    summary = st.columns(4)
    summary[0].metric("Rows in View", f"{len(filtered):,}")
    summary[1].metric(
        "Paper Suspects",
        f"{int(filtered['PAPER_INSPECTION_SUSPECT'].astype(bool).sum()):,}",
    )
    summary[2].metric(
        "Audit Alerts", f"{int(filtered['AUDIT_ALERT'].astype(bool).sum()):,}"
    )
    summary[3].metric(
        "IsolationForest Anomalies",
        f"{int(filtered['ISOLATION_FLAG'].astype(bool).sum()):,}",
    )

    audit_display = _audit_display(filtered)
    render_dataframe(audit_display.style.apply(_style_audit_rows, axis=1), height=480)
    options = filtered["BRIDGE_ID"].astype(str).tolist()
    preferred = st.session_state.get("auth_selected_bridge")
    default_index = options.index(preferred) if preferred in options else 0
    ensure_select_state("audit_bridge_select", options, default_index)
    selected = st.selectbox(
        "Select a flagged bridge for accountability review",
        options,
        key="audit_bridge_select",
    )
    st.session_state["auth_selected_bridge"] = str(selected)
    row = filtered.loc[filtered["BRIDGE_ID"].astype(str).eq(str(selected))].iloc[0]

    comparison = st.columns(4)
    comparison[0].metric(
        "Inspector's Claimed Rating",
        f"{int(row.get('REPORTED_LAST_MIN_COND', 0))} / 9",
    )
    comparison[1].metric(
        "Physics-Implied Condition",
        f"{fmt(row.get('PHYSICS_EXPECTED_CONDITION'), 2)} / 9",
        help="9 × S(t0 + 5 | t0), a monotone NBI-scale survival proxy.",
    )
    comparison[2].metric("Condition Gap", f"{fmt(row.get('CONDITION_GAP'), 2)} points")
    comparison[3].metric(
        "IsolationForest",
        "ANOMALY" if bool(row.get("ISOLATION_FLAG", False)) else "NORMAL",
    )
    if bool(row.get("PAPER_INSPECTION_SUSPECT", False)):
        st.error(
            "ZERO-TRUST ALERT: the reported healthy rating contradicts the "
            "physics-implied five-year survival of this bridge."
        )
    _render_accountability_card(runtime, ledger, {**row.to_dict(), "STATE": row.get("STATE"), "DISTRICT": row.get("DISTRICT")})


# ---------------------------------------------------------------------------
# Portal B · Sub-tab 3: survival curves & SHAP
# ---------------------------------------------------------------------------


def render_deep_dive_tab(
    runtime: Mapping[str, Any],
    queue: pd.DataFrame,
    ledger: pd.DataFrame,
) -> None:
    st.markdown("#### Bridge Deep-Dive, Conditional Survival & SHAP Explainability")
    if queue.empty:
        st.error("The bridge portfolio is unavailable.")
        return
    options = queue["BRIDGE_ID"].astype(str).tolist()
    preferred = st.session_state.get("auth_selected_bridge")
    default_index = options.index(preferred) if preferred in options else 0
    ensure_select_state("deep_dive_bridge", options, default_index)
    bridge_id = st.selectbox(
        "Select bridge for deep dive",
        options,
        key="deep_dive_bridge",
    )
    st.session_state["auth_selected_bridge"] = str(bridge_id)
    queue_row = queue.loc[queue["BRIDGE_ID"].astype(str).eq(str(bridge_id))].iloc[0]
    state = str(queue_row.get("STATE", "Unknown"))
    district = str(queue_row.get("DISTRICT", "Unknown"))
    active_events = list(st.session_state.get("active_events", []))
    scenario = scenario_event_for_bridge(str(bridge_id), state, district)
    curves = curves_for_bridge(
        runtime,
        str(bridge_id),
        events=active_events,
        scenario_event=scenario,
    )
    if curves is None:
        st.error("No transformed survival curve is available for this bridge.")
        return

    is_scenario = str(curves["event_label"]).startswith("SCENARIO")
    st.caption(
        f"{district}, {state}  ·  decision time t0 = {curves['t0']:.1f} years  ·  "
        + ("illustrative scenario shock" if is_scenario else f"live shock `{curves['event_label']}`")
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=CURVE_GRID,
            y=curves["baseline_curve"],
            mode="lines",
            name="Baseline conditional survival",
            line={"color": "#2563eb", "width": 3},
            hovertemplate="Baseline %{x:.0f} y<br>S = %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=CURVE_GRID,
            y=curves["post_curve"],
            mode="lines",
            name="Post-flood-shock survival",
            line={"color": "#dc2626", "width": 3, "dash": "dash"},
            hovertemplate="Post-shock %{x:.0f} y<br>S = %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=SURVIVAL_THRESHOLD,
        line_dash="dash",
        line_color="#991b1b",
        annotation_text="Safety threshold τ = 0.68",
        annotation_position="top right",
    )
    figure.add_vline(
        x=DECISION_HORIZON,
        line_dash="dot",
        line_color="#64748b",
        annotation_text="5-year horizon",
        annotation_position="top left",
    )
    figure.add_vline(
        x=curves["baseline_rul"],
        line_dash="dot",
        line_color="#2563eb",
        annotation_text=f"Baseline RUL {curves['baseline_rul']:.2f} y",
        annotation_position="bottom left",
    )
    figure.add_vline(
        x=curves["post_rul"],
        line_dash="dot",
        line_color="#dc2626",
        annotation_text=f"Post-shock RUL {curves['post_rul']:.2f} y",
        annotation_position="bottom right",
    )
    figure.add_trace(
        go.Scatter(
            x=[curves["post_rul"], curves["baseline_rul"]],
            y=[SURVIVAL_THRESHOLD, SURVIVAL_THRESHOLD],
            mode="markers",
            name="Exact RUL crossover",
            marker={"size": 11, "color": ["#dc2626", "#2563eb"], "symbol": "diamond"},
            hovertemplate="Crossover at %{x:.2f} years<extra></extra>",
        )
    )
    figure.update_layout(
        height=500,
        template="plotly_white",
        title="0–60-Year Conditional Survival: Baseline vs Post-Flood Shock",
        xaxis_title="Years from decision date",
        yaxis_title="Conditional survival probability",
        yaxis_range=[0.0, 1.02],
        xaxis_range=[0.0, RUL_HORIZON],
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 10, "r": 20, "t": 70, "b": 10},
    )
    render_plotly(figure)

    metrics = st.columns(5)
    metrics[0].metric("Baseline S(t0+5)", fmt(curves["baseline_survival_5y"], 3))
    metrics[1].metric("Post-Shock S(t0+5)", fmt(curves["post_survival_5y"], 3))
    metrics[2].metric(
        "Baseline RUL",
        f"{fmt(curves['baseline_rul'], 2)} y" + (" · censored" if curves["baseline_censored"] else ""),
    )
    metrics[3].metric(
        "Post-Shock RUL",
        f"{fmt(curves['post_rul'], 2)} y" + (" · censored" if curves["post_censored"] else ""),
    )
    metrics[4].metric("Net RUL Loss", f"{fmt(curves['rul_loss'], 2)} y")

    st.divider()
    st.markdown("#### Why It Got This Rating — Top SHAP Risk Contributions")
    contributions, source_note = shap_contributions(
        runtime,
        str(bridge_id),
        events=active_events,
        scenario_event=scenario,
        use_persisted=False,
    )
    if contributions.empty:
        contributions, source_note = shap_contributions(
            runtime, str(bridge_id), use_persisted=True
        )
    if contributions.empty:
        st.info("No SHAP contribution records are available for this bridge.")
    else:
        chart_data = contributions.sort_values("SHAP_VALUE", ascending=True).copy()
        colors = [
            "#dc2626" if float(value) >= 0.0 else "#2563eb"
            for value in chart_data["SHAP_VALUE"]
        ]
        shap_figure = go.Figure(
            go.Bar(
                y=chart_data["FEATURE"].astype(str),
                x=chart_data["SHAP_VALUE"].astype(float),
                orientation="h",
                marker_color=colors,
                text=[fmt(value, 3) for value in chart_data["SHAP_VALUE"]],
                textposition="outside",
                hovertemplate="%{y}<br>SHAP contribution = %{x:.4f}<extra></extra>",
            )
        )
        shap_figure.add_vline(x=0.0, line_color="#64748b", line_width=1)
        shap_figure.update_layout(
            height=max(380, 34 * len(chart_data)),
            template="plotly_white",
            title="Physical factors pushing the bridge toward failure (positive = higher risk)",
            xaxis_title="Risk contribution",
            yaxis_title="",
            margin={"l": 10, "r": 40, "t": 60, "b": 10},
        )
        render_plotly(shap_figure)
        st.caption(source_note)

    force_candidates = [
        FORCE_PLOT_DIR / f"{safe_filename(str(bridge_id))}_force.html",
        ROOT / "artifacts" / "module4" / "live" / "force_plots" / f"{safe_filename(str(bridge_id))}_force.html",
    ]
    force_path = next((path for path in force_candidates if path.exists()), None)
    if force_path is not None:
        try:
            embed_html(force_path.read_text(encoding="utf-8"), 360)
            st.caption(f"Embedded standalone SHAP force plot: `{force_path.name}`")
        except Exception as exc:
            st.warning(f"Could not embed `{force_path.name}`: {type(exc).__name__}: {exc}")
    else:
        st.caption(
            "No persisted standalone SHAP force plot exists for this bridge; the "
            "contribution chart above provides the local physical attribution."
        )


# ---------------------------------------------------------------------------
# Portal B · Sub-tab 4: landmark benchmarks and model health
# ---------------------------------------------------------------------------


def _archetype_records(runtime: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = runtime.get("archetypes")
    records: list[Mapping[str, Any]] = []
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, Mapping)]
    elif isinstance(payload, Mapping):
        candidate = payload.get("archetype_validation_table", payload.get("archetypes", []))
        if isinstance(candidate, list):
            records = [item for item in candidate if isinstance(item, Mapping)]
    if not records:
        summary = _mapping(runtime.get("summary"))
        candidate = summary.get("archetype_validation_table", [])
        if isinstance(candidate, list):
            records = [item for item in candidate if isinstance(item, Mapping)]
    return records[:3]


def render_benchmarks_tab(runtime: Mapping[str, Any]) -> None:
    st.markdown("#### Landmark Indian Collapse Benchmarks & Model Health")
    st.caption(
        "Hydraulic failure dominates the Indian bridge portfolio. These landmarks "
        "validate the zero-trust premise: paper-healthy bridges can carry extreme "
        "physics-implied collapse risk."
    )
    records = _archetype_records(runtime)
    if not records:
        st.warning("Landmark archetype scorecards are unavailable.")
    else:
        columns = st.columns(len(records))
        for column, record in zip(columns, records, strict=False):
            with column:
                with st.container(border=True):
                    st.markdown(f"##### {record.get('label', record.get('bridge_id', 'Archetype'))}")
                    st.caption(
                        f"`{record.get('bridge_id', '—')}` · {record.get('district', 'Unknown')}, "
                        f"{record.get('state', 'Unknown')} · collapse {record.get('collapse_year', '—')}"
                    )
                    gap = st.columns(2)
                    gap[0].metric(
                        "Reported Claim",
                        f"{int(record.get('reported_condition', 0))} / 9",
                    )
                    gap[1].metric(
                        "Physics-Implied",
                        f"{fmt(record.get('physics_expected_condition'), 2)} / 9",
                    )
                    st.metric(
                        "5-Year Conditional Survival",
                        fmt(record.get("predicted_5y_conditional_survival"), 3),
                    )
                    st.metric(
                        "Post-Event RUL Loss",
                        f"{fmt(record.get('post_disaster_rul_loss_years'), 2)} y",
                    )
                    st.markdown(
                        " ".join(
                            [
                                pill(
                                    f"Triage: {record.get('triage_priority', '—')}",
                                    priority_kind(record.get("triage_priority", "")),
                                ),
                                pill(
                                    f"Paper suspect: {bool(record.get('paper_inspection_suspect', False))}",
                                    "danger" if bool(record.get("paper_inspection_suspect", False)) else "success",
                                ),
                                pill(
                                    f"Immediate NDT: {'YES' if bool(record.get('immediate_ndt_deployment', False)) else 'NO'}",
                                    "danger" if bool(record.get("immediate_ndt_deployment", False)) else "success",
                                ),
                            ]
                        ),
                        unsafe_allow_html=True,
                    )

    st.divider()
    summary = _mapping(runtime.get("summary"))
    module2 = _mapping(summary.get("module2"))
    test_metrics = _mapping(module2.get("test_metrics"))
    module3 = _mapping(summary.get("module3"))
    audit_metrics = _mapping(module3.get("metrics"))
    audit_z = _mapping(audit_metrics.get("audit_z_score"))
    report = _mapping(runtime.get("survival_report"))

    st.markdown("#### Portfolio Model Health")
    health = st.columns(5)
    health[0].metric("Harrell's C (Test)", fmt(test_metrics.get("harrell_c"), 4))
    health[1].metric("Uno IPCW C (Test)", fmt(test_metrics.get("uno_ipcw_c"), 4))
    health[2].metric(
        "Integrated Brier Score",
        fmt(test_metrics.get("integrated_brier_score"), 4),
    )
    health[3].metric(
        "Audit Z-Score ROC-AUC",
        fmt(audit_z.get("roc_auc"), 4),
    )
    health[4].metric(
        "Selected Survival Model",
        str(module2.get("selected_model", "cox_time_varying")),
    )

    st.divider()
    st.markdown("#### Standardised Climate Hazard Ratios")
    payload = hazard_ratio_payload(runtime)
    per_1sd = _mapping(payload.get("per_1sd"))
    per_2sd = _mapping(payload.get("per_2sd"))
    categories = [
        "Monsoon Anomaly",
        "Cumulative Scour Load",
        "Cumulative Flood Exceedances",
        "Combined Extreme Shock",
    ]
    one_sd = [
        float(per_1sd.get("MONSOON_ANOMALY", np.nan)),
        float(per_1sd.get("CUMULATIVE_SCOUR_LOAD", np.nan)),
        float(per_1sd.get("CUMULATIVE_FLOOD_EXCEEDANCE", np.nan)),
        float(payload.get("combined_1sd", np.nan)),
    ]
    two_sd = [
        float(per_2sd.get("MONSOON_ANOMALY", np.nan)),
        float(per_2sd.get("CUMULATIVE_SCOUR_LOAD", np.nan)),
        float(per_2sd.get("CUMULATIVE_FLOOD_EXCEEDANCE", np.nan)),
        float(payload.get("combined_2sd", np.nan)),
    ]
    if not np.isfinite(np.asarray(one_sd + two_sd, dtype=float)).any():
        st.warning("Hazard-ratio values are unavailable in `survival_report.json`.")
    else:
        figure = go.Figure()
        figure.add_trace(
            go.Bar(
                y=categories,
                x=one_sd,
                name="+1 SD climate shock",
                orientation="h",
                marker_color="#60a5fa",
                text=[fmt(value, 3) for value in one_sd],
                textposition="outside",
                hovertemplate="%{y}<br>+1 SD HR = %{x:.3f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Bar(
                y=categories,
                x=two_sd,
                name="+2 SD climate shock",
                orientation="h",
                marker_color="#1d4ed8",
                text=[fmt(value, 3) for value in two_sd],
                textposition="outside",
                hovertemplate="%{y}<br>+2 SD HR = %{x:.3f}<extra></extra>",
            )
        )
        figure.add_vline(x=1.0, line_dash="dot", line_color="#64748b")
        figure.update_layout(
            barmode="group",
            height=430,
            template="plotly_white",
            xaxis_title="Hazard ratio versus baseline",
            yaxis_title="",
            legend={"orientation": "h", "y": 1.08},
            margin={"l": 10, "r": 30, "t": 30, "b": 10},
        )
        render_plotly(figure)
        st.caption(
            "Continuous shocks are reported at +1 SD and +2 SD on the training-standardised "
            "physical scale; the combined extreme shock exponentiates the summed linear predictor."
        )

    st.divider()
    triage = runtime.get("triage_findings")
    if triage is not None and not triage.empty:
        with st.expander("Persisted post-disaster triage reference", expanded=False):
            columns = [
                column
                for column in (
                    "BRIDGE_ID",
                    "DISTRICT",
                    "TRIAGE_PRIORITY",
                    "BASELINE_SURVIVAL_AT_DECISION",
                    "POST_EVENT_SURVIVAL_AT_DECISION",
                    "RUL_LOSS_YEARS",
                    "IMMEDIATE_NDT_DEPLOYMENT",
                )
                if column in triage.columns
            ]
            if columns:
                render_dataframe(triage.loc[:, columns].head(15))
    st.caption(
        "Classical stack only: Cox counting-process, Random Survival Forest, "
        "Gradient-Boosted Survival Analysis, IsolationForest, robust Z-residuals and SHAP."
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()
    runtime = get_runtime(str(ROOT))

    with st.sidebar:
        st.markdown("### Operational Role")
        role = st.radio(
            "Select portal",
            ("Field Inspector Portal", "NHAI / State PWD Command Center"),
            key="portal_role",
            label_visibility="collapsed",
        )
        st.caption(
            "Portal A is blind by design. Portal B is the executive, vigilance and "
            "meteorological command view."
        )
        st.divider()
        st.markdown("### Live Meteorology")
        weather = st.session_state.get("live_weather")
        if weather:
            districts = weather.get("districts", pd.DataFrame())
            n_alerts = (
                int(districts["EXCEEDS_THRESHOLD"].astype(bool).sum())
                if districts is not None and not districts.empty
                else 0
            )
            st.metric("Active flood alerts", n_alerts)
            st.caption(f"Last sync: {sync_label(weather.get('synced_at'))}")
            if weather.get("offline"):
                st.caption("Offline-safe fallback active.")
        else:
            st.caption("Not synced yet. Open the Command Center to fetch live weather.")
        st.divider()
        st.caption(
            "Persistence: `artifacts/inspections_live_ledger.parquet` · "
            "uploads: `artifacts/uploads/<bridge_id>/`"
        )
        st.caption(
            "All scoring is read-only against the persisted classical ML artefacts; "
            "no deep-learning framework is used."
        )

    bridges = runtime.get("bridges")
    if bridges is None or bridges.empty:
        st.error(
            "The persisted bridge inventory `artifacts/data/bridges.parquet` could not be loaded."
        )
        if runtime.get("errors"):
            with st.expander("Artefact diagnostics", expanded=True):
                for error in runtime["errors"]:
                    st.warning(error)
        st.stop()

    ledger = load_or_create_ledger(bridges, runtime.get("inspections"))
    st.caption(f"Persistent inspector ledger: `{ledger_signature()}` · {len(ledger):,} records")

    if role == "Field Inspector Portal":
        render_inspector_portal(runtime, ledger)
        return

    st.markdown(
        '<div class="zt-hero">'
        '<div class="zt-kicker">Portal B · NHAI / MoRTH / State PWD</div>'
        '<h1>Command, Vigilance & Live Flash-Flood Triage</h1>'
        '<p>Executive transparency across the bridge portfolio: live meteorology, '
        'physics-implied survival, zero-trust audit, revised RUL rankings and SHAP '
        'explanations.</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    if runtime.get("errors"):
        with st.expander("Artefact availability warnings", expanded=False):
            for error in runtime["errors"]:
                st.warning(error)

    weather = sync_weather(runtime)
    audit_table = cached_audit_table(
        str(ROOT), ledger_signature(), audit_overlay_signature()
    )
    active_events = list(st.session_state.get("active_events", []))
    live_triage = st.session_state.get("live_triage", pd.DataFrame())
    queue = build_master_queue(
        runtime,
        ledger,
        audit_table,
        events=active_events,
        live_triage=live_triage,
    )

    quick = st.columns(4)
    quick[0].metric("Bridges Monitored", f"{len(queue):,}" if not queue.empty else f"{len(bridges):,}")
    quick[1].metric(
        "Critical Triage",
        f"{int(queue['TRIAGE_PRIORITY'].eq('CRITICAL').sum()):,}" if not queue.empty else "0",
    )
    quick[2].metric(
        "Paper-Inspection Suspects",
        f"{int(audit_table['PAPER_INSPECTION_SUSPECT'].astype(bool).sum()):,}"
        if not audit_table.empty
        else "0",
    )
    weather_districts = weather.get("districts", pd.DataFrame())
    live_weather_alerts = (
        int(weather_districts["EXCEEDS_THRESHOLD"].astype(bool).sum())
        if weather_districts is not None and not weather_districts.empty
        else 0
    )
    quick[3].metric(
        "Live Weather Alerts / Active Shocks",
        f"{live_weather_alerts} / {len(active_events)}",
    )

    tabs = st.tabs(
        [
            "1 · Live Weather & Repair Queue",
            "2 · Zero-Trust Vigilance & Audit",
            "3 · Survival Deep-Dive & SHAP",
            "4 · Collapse Benchmarks & Model Health",
        ]
    )
    with tabs[0]:
        render_live_queue_tab(runtime, ledger, queue, weather)
    with tabs[1]:
        render_audit_tab(runtime, ledger, audit_table)
    with tabs[2]:
        render_deep_dive_tab(runtime, queue, ledger)
    with tabs[3]:
        render_benchmarks_tab(runtime)

    st.divider()
    st.caption(
        "Right-censored survival handling · Harrell and Uno C-index · Integrated Brier "
        "Score · robust Audit Z-score · IsolationForest · SHAP · Module 4 post-disaster RUL."
    )


if __name__ == "__main__":
    main()
