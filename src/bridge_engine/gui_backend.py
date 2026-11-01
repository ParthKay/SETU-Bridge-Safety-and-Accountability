"""Backend services for the Zero-Trust dual-portal Streamlit application.

The module keeps every stateful, network, filesystem, survival and audit
operation out of ``app.py`` so that the UI layer remains declarative.

Responsibilities
----------------
* Automated live meteorological ingestion (IMD bulletins + Open-Meteo).
* Deterministic inspector directory and persistent live inspection ledger.
* Incremental Module 3 re-scoring of field submissions.
* Module 4 regional triage and the portfolio-wide revised repair queue.
* Survival-curve, RUL, SHAP and downloadable attachment helpers.

Only classical survival / audit / SHAP code is imported from the project
modules.  No deep-learning framework is used anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
MODULE2_DIR = ARTIFACTS / "module2"
MODULE3_DIR = ARTIFACTS / "module3"
MODULE4_DIR = ARTIFACTS / "module4"
UPLOAD_DIR = ARTIFACTS / "uploads"
LEDGER_PATH = ARTIFACTS / "inspections_live_ledger.parquet"
AUDIT_OVERLAY_PATH = MODULE3_DIR / "live_audit_overlay.parquet"
LIVE_TRIAGE_DIR = MODULE4_DIR / "live"
LIVE_TRIAGE_PATH = LIVE_TRIAGE_DIR / "live_triage_findings.parquet"
FORCE_PLOT_DIR = MODULE4_DIR / "force_plots"

BRIDGES_PATH = DATA_DIR / "bridges.parquet"
INSPECTIONS_PATH = DATA_DIR / "inspections.parquet"
CLIMATE_PATH = DATA_DIR / "climate.parquet"
SURVIVAL_ENGINE_PATH = MODULE2_DIR / "survival_engine.joblib"
AUDIT_MODELS_PATH = MODULE3_DIR / "audit_models.joblib"
TV_PREPROCESSOR_PATH = DATA_DIR / "time_varying_preprocessor.joblib"
STATIC_PREPROCESSOR_PATH = DATA_DIR / "static_preprocessor.joblib"
AUDIT_FINDINGS_PATH = MODULE3_DIR / "audit_findings.parquet"
TRIAGE_FINDINGS_PATH = MODULE4_DIR / "triage_findings.parquet"
XAI_CONTRIBUTIONS_PATH = MODULE4_DIR / "xai_contributions.parquet"
SUMMARY_PATH = ARTIFACTS / "end_to_end_summary.json"
SURVIVAL_REPORT_PATH = MODULE2_DIR / "survival_report.json"
ARCHETYPE_PATH = MODULE4_DIR / "archetype_validation.json"

AS_OF_YEAR = 2024
SURVIVAL_THRESHOLD = 0.68
DECISION_HORIZON = 5.0
RUL_HORIZON = 60.0
CURVE_GRID = np.arange(0.0, RUL_HORIZON + 1.0, 1.0, dtype=np.float64)
SURVIVAL_FLOOR = 1e-6
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
IMD_ENDPOINTS = (
    "https://mausam.imd.gov.in/api/warning_api.php",
    "https://mausam.imd.gov.in/api/statewise_warning.php",
    "https://mausam.imd.gov.in/rss/weather_warning.xml",
)
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

MANAGING_AGENCIES = ("NHAI", "State PWD", "BRO", "Municipal Corporation")
AGENCY_CODES = {
    "NHAI": "NHAI",
    "State PWD": "PWD",
    "BRO": "BRO",
    "Municipal Corporation": "MC",
}
AGENCY_DOMAINS = {
    "NHAI": "nhai.gov.in",
    "State PWD": "pwd.gov.in",
    "BRO": "bro.gov.in",
    "Municipal Corporation": "municipal.gov.in",
}

FIRST_NAMES = (
    "Aarav", "Aditi", "Ajay", "Akash", "Ananya", "Anil", "Arjun", "Asha",
    "Deepak", "Divya", "Farhan", "Gaurav", "Harish", "Ishita", "Jaya",
    "Kabir", "Kavya", "Lakshmi", "Manish", "Meera", "Mohan", "Nandini",
    "Naveen", "Neha", "Pooja", "Pranav", "Priya", "Rahul", "Ramesh",
    "Ravi", "Ritu", "Rohit", "Sandeep", "Sanjay", "Sanya", "Shreya",
    "Siddharth", "Sunita", "Varun", "Vikram", "Vivek", "Zoya",
)
LAST_NAMES = (
    "Agarwal", "Bhat", "Chandra", "Das", "Deshmukh", "Ghosh", "Gupta",
    "Iyer", "Jain", "Joshi", "Kapoor", "Khan", "Kulkarni", "Kumar",
    "Mehta", "Mishra", "Nair", "Naidu", "Patel", "Patil", "Pillai",
    "Rao", "Reddy", "Roy", "Saxena", "Shah", "Sharma", "Singh",
    "Sinha", "Srinivasan", "Thakur", "Varma", "Verma", "Yadav",
)
REMARKS_HIGH = (
    "Routine visual inspection completed. Superstructure and bearings appear sound.",
    "No distress observed in deck, girders or expansion joints during this cycle.",
    "Monsoon preparedness check completed. Waterway observed clear of debris.",
    "Protective railings and approach slab found serviceable.",
)
REMARKS_MEDIUM = (
    "Minor hairline cracking noted on deck; monitoring recommended at next cycle.",
    "Localised spalling observed with exposed reinforcement at one pier cap.",
    "Approach embankment shows minor settlement; drainage channel partly blocked.",
    "Bearing pads show moderate ageing; cleaning and lubrication advised.",
)
REMARKS_LOW = (
    "Significant scour observed around pier 2; immediate hydraulic review required.",
    "Widespread concrete spalling and corrosion of reinforcement in superstructure.",
    "Deck slab exhibits structural cracking; restricted loading recommended.",
    "Substructure movement detected; detailed NDT and load assessment required.",
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def safe_filename(value: str) -> str:
    """Return a filesystem-safe string without removing Unicode letters."""

    allowed = "-_."
    return "".join(
        character if character.isalnum() or character in allowed else "_"
        for character in str(value)
    )


def _coerce_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _read_frame_any(path: Path) -> pd.DataFrame:
    """Read a Parquet file with JSON/CSV fallbacks."""

    candidates = (path, path.with_suffix(".json"), path.with_suffix(".csv"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            return pd.read_parquet(candidate)
        if candidate.suffix == ".json":
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return pd.DataFrame(payload)
        return pd.read_csv(candidate)
    raise FileNotFoundError(f"none of {candidates} exists")


def _write_frame_any(frame: pd.DataFrame, path: Path) -> Path:
    """Persist a frame as Parquet, falling back to JSON then CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        pass
    try:
        path.with_suffix(".json").write_text(
            frame.to_json(orient="records", date_format="iso"),
            encoding="utf-8",
        )
        return path.with_suffix(".json")
    except Exception:
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        return csv_path


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").upper()


def display_feature_name(feature: str) -> str:
    """Convert one-hot / transformed feature names into physical labels."""

    text = str(feature)
    for prefix, label in (
        ("STRUCTURE_KIND_MATERIAL_", "Material"),
        ("STRUCTURE_TYPE_DESIGN_", "Design"),
        ("DESIGN_LOAD_", "Design Load"),
        ("STATE_", "State"),
        ("DISTRICT_", "District"),
        ("OWNER_", "Owner"),
        ("missingindicator_", "Missing flag — "),
    ):
        if text.startswith(prefix):
            value = text[len(prefix):].replace("___MISSING__", "not reported")
            return f"{label}: {value}"
    return text


# ---------------------------------------------------------------------------
# Live meteorological ingestion
# ---------------------------------------------------------------------------


def _strip_timezone(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_localize(None)
    return series


def _extract_json_strings(value: Any, output: list[str], depth: int = 0) -> None:
    if depth > 5 or len(output) > 200:
        return
    if isinstance(value, str):
        if 12 <= len(value) <= 800:
            output.append(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _extract_json_strings(item, output, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value[:80]:
            _extract_json_strings(item, output, depth + 1)


def _imd_level(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("red alert", "extreme", "very heavy", "extremely heavy")):
        return "RED"
    if any(token in lowered for token in ("orange alert", "severe", "heavy rainfall", "heavy rain")):
        return "ORANGE"
    if any(token in lowered for token in ("yellow alert", "heavy", "thunderstorm", "warning", "flood")):
        return "YELLOW"
    return "INFO"


def fetch_imd_bulletins(
    states: Sequence[str],
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Fetch and parse public IMD warning bulletins.

    IMD occasionally changes its endpoint or blocks automated clients.  Every
    failure is captured and returned rather than raised, so the portal remains
    fully operational in air-gapped or firewalled environments.
    """

    state_lookup = {str(state).strip().lower(): str(state) for state in states}
    bulletins: list[dict[str, Any]] = []
    errors: list[str] = []
    for url in IMD_ENDPOINTS:
        try:
            response = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
            if response.status_code >= 400:
                errors.append(f"{url}: HTTP {response.status_code}")
                continue
            text = response.text or ""
            candidates: list[str] = []
            stripped = text.lstrip()
            if stripped.startswith(("{", "[")):
                try:
                    payload = json.loads(stripped)
                    _extract_json_strings(payload, candidates)
                except Exception:
                    candidates.append(stripped[:1000])
            else:
                # ``html.parser`` is always available and parses RSS/XML text
                # well enough for warning headlines without requiring lxml.
                soup = BeautifulSoup(text, "html.parser")
                nodes = soup.find_all(["item", "title", "description", "warning", "p", "li"])
                if not nodes:
                    soup = BeautifulSoup(text, "html.parser")
                    nodes = soup.find_all(["p", "li", "div", "span", "h1", "h2", "h3"])
                for node in nodes[:200]:
                    candidate = " ".join(node.get_text(" ", strip=True).split())
                    if candidate:
                        candidates.append(candidate)

            seen: set[str] = set()
            for candidate in candidates:
                lowered = candidate.lower()
                if candidate in seen:
                    continue
                if not any(
                    token in lowered
                    for token in ("rain", "flood", "warning", "alert", "thunder", "cyclone", "scour")
                ):
                    continue
                matched_state = next(
                    (full for key, full in state_lookup.items() if key in lowered),
                    "",
                )
                if not matched_state:
                    continue
                seen.add(candidate)
                bulletins.append(
                    {
                        "headline": candidate[:420],
                        "state": matched_state,
                        "level": _imd_level(candidate),
                        "source": "IMD",
                        "endpoint": url,
                    }
                )
            if bulletins:
                break
        except requests.RequestException as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # parser guard: a malformed feed must never crash
            errors.append(f"{url}: parser {type(exc).__name__}: {exc}")
    return {
        "ok": bool(bulletins),
        "bulletins": bulletins[:80],
        "errors": errors,
    }


WEATHER_CODE_LABELS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm with hail",
}


def fetch_open_meteo(
    latitude: float,
    longitude: float,
    *,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Return live 24 h / 72 h rainfall and weather-code severity."""

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
        "hourly": "precipitation",
        "daily": "precipitation_sum,weather_code",
        "forecast_days": 4,
        "timezone": "auto",
    }
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            headers=HTTP_HEADERS,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {}) or {}
        hourly = payload.get("hourly", {}) or {}
        daily = payload.get("daily", {}) or {}

        precip_series = pd.Series(hourly.get("precipitation", []), dtype="float64")
        precip_series = pd.to_numeric(precip_series, errors="coerce").fillna(0.0)
        times = pd.to_datetime(pd.Series(hourly.get("time", [])), errors="coerce")
        times = _strip_timezone(times)

        start = 0
        current_time = pd.to_datetime(current.get("time"), errors="coerce")
        if pd.notna(current_time) and not times.empty:
            current_time = pd.Timestamp(current_time)
            if current_time.tzinfo is not None:
                current_time = current_time.tz_localize(None)
            start = int(np.searchsorted(times.to_numpy(), current_time.to_datetime64()))
            start = int(np.clip(start, 0, max(len(times) - 1, 0)))

        rainfall_24h = float(precip_series.iloc[start:start + 24].sum())
        rainfall_72h = float(precip_series.iloc[start:start + 72].sum())
        code = int(_coerce_float(current.get("weather_code"), 0.0))
        if code == 0 and daily.get("weather_code"):
            code = int(_coerce_float(pd.Series(daily["weather_code"]).max(), 0.0))
        return {
            "ok": True,
            "rainfall_24h_mm": rainfall_24h,
            "rainfall_72h_mm": rainfall_72h,
            "weather_code": code,
            "weather_label": WEATHER_CODE_LABELS.get(code, f"WMO code {code}"),
            "temperature_c": _coerce_float(current.get("temperature_2m")),
            "wind_speed_kmh": _coerce_float(current.get("wind_speed_10m")),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": "",
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "rainfall_24h_mm": np.nan,
            "rainfall_72h_mm": np.nan,
            "weather_code": 0,
            "weather_label": "Unavailable",
            "temperature_c": np.nan,
            "wind_speed_kmh": np.nan,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "rainfall_24h_mm": np.nan,
            "rainfall_72h_mm": np.nan,
            "weather_code": 0,
            "weather_label": "Parser error",
            "temperature_c": np.nan,
            "wind_speed_kmh": np.nan,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": f"parser {type(exc).__name__}: {exc}",
        }


def district_coordinate_table(bridges: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bridge coordinates to one row per state/district."""

    required = {"BRIDGE_ID", "STATE", "DISTRICT", "LATITUDE", "LONGITUDE"}
    if bridges is None or bridges.empty or not required.issubset(bridges.columns):
        return pd.DataFrame(columns=["STATE", "DISTRICT", "LATITUDE", "LONGITUDE", "FLOOD_PRONENESS"])
    work = bridges.loc[:, [column for column in (
        "STATE", "DISTRICT", "LATITUDE", "LONGITUDE", "FLOOD_PRONENESS"
    ) if column in bridges.columns]].copy()
    work["STATE"] = work["STATE"].astype(str)
    work["DISTRICT"] = work["DISTRICT"].astype(str)
    for column in ("LATITUDE", "LONGITUDE", "FLOOD_PRONENESS"):
        if column not in work.columns:
            work[column] = 0.5 if column == "FLOOD_PRONENESS" else np.nan
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = (
        work.groupby(["STATE", "DISTRICT"], as_index=False, observed=True)
        .agg(
            LATITUDE=("LATITUDE", "mean"),
            LONGITUDE=("LONGITUDE", "mean"),
            FLOOD_PRONENESS=("FLOOD_PRONENESS", "mean"),
            N_BRIDGES=("DISTRICT", "size"),
        )
        .sort_values(["STATE", "DISTRICT"], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["LATITUDE"] = grouped["LATITUDE"].fillna(22.0)
    grouped["LONGITUDE"] = grouped["LONGITUDE"].fillna(78.0)
    grouped["FLOOD_PRONENESS"] = grouped["FLOOD_PRONENESS"].fillna(0.5)
    return grouped


def _classify_weather_shock(
    rainfall_24h: float,
    rainfall_72h: float,
    weather_code: int,
    imd_level: str,
    flood_prone: float,
) -> dict[str, Any]:
    """Map rainfall, WMO codes and IMD alerts into Module 4 shock parameters."""

    r24 = _coerce_float(rainfall_24h, 0.0)
    r72 = _coerce_float(rainfall_72h, 0.0)
    code = int(_coerce_float(weather_code, 0.0))
    level = str(imd_level or "INFO").upper()
    flood_prone = _coerce_float(flood_prone, 0.5)

    if (
        r24 >= 204.4
        or r72 >= 300.0
        or code >= 96
        or level == "RED"
    ):
        return {
            "ALERT_LEVEL": "EXTREME",
            "EXCEEDS_THRESHOLD": True,
            "SEVERITY": 1.85,
            "MONSOON_ANOMALY": 3.5,
            "RAINFALL_MULTIPLIER": 2.15,
            "DISCHARGE_MULTIPLIER": 2.35,
            "VELOCITY_MULTIPLIER": 1.80,
            "SCOUR_MULTIPLIER": 2.60,
            "FLOOD_EXCEEDANCE": 2,
            "CUMULATIVE_SCOUR_INCREMENT": 20.0,
            "EVENT_TYPE": "COMPOUND",
        }
    if (
        r24 >= 115.6
        or r72 >= 200.0
        or code >= 95
        or level == "ORANGE"
    ):
        return {
            "ALERT_LEVEL": "SEVERE",
            "EXCEEDS_THRESHOLD": True,
            "SEVERITY": 1.55,
            "MONSOON_ANOMALY": 2.7,
            "RAINFALL_MULTIPLIER": 1.90,
            "DISCHARGE_MULTIPLIER": 2.15,
            "VELOCITY_MULTIPLIER": 1.60,
            "SCOUR_MULTIPLIER": 2.35,
            "FLOOD_EXCEEDANCE": 2,
            "CUMULATIVE_SCOUR_INCREMENT": 16.0,
            "EVENT_TYPE": "COMPOUND",
        }
    if (
        r24 >= 64.5
        or r72 >= 120.0
        or code in {65, 67, 82}
        or level == "YELLOW"
    ):
        return {
            "ALERT_LEVEL": "HEAVY",
            "EXCEEDS_THRESHOLD": True,
            "SEVERITY": 1.32,
            "MONSOON_ANOMALY": 2.1,
            "RAINFALL_MULTIPLIER": 1.72,
            "DISCHARGE_MULTIPLIER": 1.90,
            "VELOCITY_MULTIPLIER": 1.45,
            "SCOUR_MULTIPLIER": 2.05,
            "FLOOD_EXCEEDANCE": 1,
            "CUMULATIVE_SCOUR_INCREMENT": 12.0,
            "EVENT_TYPE": "FLOOD",
        }
    if (
        r24 >= 35.5
        or r72 >= 70.0
        or code in {63, 80, 81}
        or (flood_prone >= 0.80 and r24 >= 20.0)
    ):
        return {
            "ALERT_LEVEL": "MODERATE",
            "EXCEEDS_THRESHOLD": False,
            "SEVERITY": 1.10,
            "MONSOON_ANOMALY": 1.1,
            "RAINFALL_MULTIPLIER": 1.30,
            "DISCHARGE_MULTIPLIER": 1.35,
            "VELOCITY_MULTIPLIER": 1.20,
            "SCOUR_MULTIPLIER": 1.35,
            "FLOOD_EXCEEDANCE": 0,
            "CUMULATIVE_SCOUR_INCREMENT": 5.0,
            "EVENT_TYPE": "FLOOD",
        }
    return {
        "ALERT_LEVEL": "NORMAL",
        "EXCEEDS_THRESHOLD": False,
        "SEVERITY": 1.0,
        "MONSOON_ANOMALY": 0.0,
        "RAINFALL_MULTIPLIER": 1.0,
        "DISCHARGE_MULTIPLIER": 1.0,
        "VELOCITY_MULTIPLIER": 1.0,
        "SCOUR_MULTIPLIER": 1.0,
        "FLOOD_EXCEEDANCE": 0,
        "CUMULATIVE_SCOUR_INCREMENT": 0.0,
        "EVENT_TYPE": "FLOOD",
    }


def fetch_live_weather(
    coordinates: pd.DataFrame,
    *,
    timeout: float = 5.0,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Fetch live district meteorology and IMD flash-flood bulletins.

    Open-Meteo calls are parallelised because the portfolio spans many
    districts.  Each call has a strict five-second timeout.  A complete offline
    fallback is returned instead of raising.
    """

    synced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if coordinates is None or coordinates.empty:
        return {
            "districts": pd.DataFrame(),
            "bulletins": [],
            "synced_at": synced_at,
            "offline": True,
            "errors": ["No district coordinates were available from bridges.parquet."],
            "n_ok": 0,
            "n_districts": 0,
        }

    states = sorted(coordinates["STATE"].astype(str).unique().tolist())
    imd = fetch_imd_bulletins(states, timeout=timeout)

    records: list[dict[str, Any]] = []
    work_items = list(coordinates.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_map = {
            executor.submit(
                fetch_open_meteo,
                float(getattr(item, "LATITUDE")),
                float(getattr(item, "LONGITUDE")),
                timeout=timeout,
            ): item
            for item in work_items
        }
        for future in as_completed(future_map):
            item = future_map[future]
            state = str(getattr(item, "STATE"))
            district = str(getattr(item, "DISTRICT"))
            flood_prone = _coerce_float(getattr(item, "FLOOD_PRONENESS", 0.5), 0.5)
            try:
                weather = future.result()
            except Exception as exc:
                weather = {
                    "ok": False,
                    "rainfall_24h_mm": np.nan,
                    "rainfall_72h_mm": np.nan,
                    "weather_code": 0,
                    "weather_label": "Unavailable",
                    "temperature_c": np.nan,
                    "wind_speed_kmh": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            matching = [
                bulletin
                for bulletin in imd.get("bulletins", [])
                if bulletin.get("state") == state
                or district.lower() in str(bulletin.get("headline", "")).lower()
            ]
            imd_level = "INFO"
            if matching:
                rank = {"INFO": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
                imd_level = max(
                    (str(entry.get("level", "INFO")) for entry in matching),
                    key=lambda value: rank.get(value, 0),
                )
            shock = _classify_weather_shock(
                weather.get("rainfall_24h_mm"),
                weather.get("rainfall_72h_mm"),
                int(weather.get("weather_code", 0)),
                imd_level,
                flood_prone,
            )
            headline = matching[0]["headline"] if matching else ""
            records.append(
                {
                    "STATE": state,
                    "DISTRICT": district,
                    "LATITUDE": float(getattr(item, "LATITUDE")),
                    "LONGITUDE": float(getattr(item, "LONGITUDE")),
                    "N_BRIDGES": int(getattr(item, "N_BRIDGES", 0)),
                    "FLOOD_PRONENESS": flood_prone,
                    "RAINFALL_24H_MM": weather.get("rainfall_24h_mm"),
                    "RAINFALL_72H_MM": weather.get("rainfall_72h_mm"),
                    "WEATHER_CODE": int(weather.get("weather_code", 0)),
                    "WEATHER_LABEL": weather.get("weather_label", ""),
                    "TEMPERATURE_C": weather.get("temperature_c"),
                    "WIND_SPEED_KMH": weather.get("wind_speed_kmh"),
                    "IMD_LEVEL": imd_level,
                    "IMD_HEADLINE": headline,
                    "SOURCE": "Open-Meteo + IMD" if weather.get("ok") else "Open-Meteo unavailable",
                    "ERROR": weather.get("error", ""),
                    **shock,
                }
            )

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.sort_values(
            ["EXCEEDS_THRESHOLD", "SEVERITY", "RAINFALL_24H_MM"],
            ascending=[False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)
    errors = list(imd.get("errors", []))
    if not frame.empty:
        errors.extend(
            f"{row.STATE}/{row.DISTRICT}: {row.ERROR}"
            for row in frame.loc[frame["ERROR"].astype(str).str.len() > 0].itertuples()
        )
    n_ok = int(frame["ERROR"].astype(str).str.len().eq(0).sum()) if not frame.empty else 0
    return {
        "districts": frame,
        "bulletins": imd.get("bulletins", []),
        "synced_at": synced_at,
        "offline": bool(frame.empty or n_ok == 0),
        "errors": errors[:20],
        "n_ok": n_ok,
        "n_districts": int(len(frame)),
        "imd_ok": bool(imd.get("ok", False)),
    }


def shocks_to_events(
    shocks: pd.DataFrame,
    *,
    max_events: int | None = None,
) -> list[Any]:
    """Convert exceeded weather rows into Module 4 ``DisasterEvent`` objects."""

    from bridge_engine.triage_and_xai import DisasterEvent

    if shocks is None or shocks.empty:
        return []
    work = shocks.copy()
    if "EXCEEDS_THRESHOLD" in work.columns:
        work = work.loc[work["EXCEEDS_THRESHOLD"].astype(bool)]
    if work.empty:
        return []
    work = work.sort_values("SEVERITY", ascending=False, kind="mergesort")
    if max_events is not None:
        work = work.head(int(max_events))
    events: list[Any] = []
    for row in work.to_dict("records"):
        district = str(row.get("DISTRICT", ""))
        state = str(row.get("STATE", ""))
        event_id = (
            f"LIVE-{_slug(state)}-{_slug(district)}-"
            f"{str(row.get('ALERT_LEVEL', 'HEAVY')).upper()}"
        )
        events.append(
            DisasterEvent(
                event_id=event_id,
                event_type=str(row.get("EVENT_TYPE", "FLOOD")),
                district=district or None,
                state=state or None,
                severity=_coerce_float(row.get("SEVERITY"), 1.3),
                monsoon_anomaly_increment=_coerce_float(row.get("MONSOON_ANOMALY"), 2.0),
                rainfall_multiplier=_coerce_float(row.get("RAINFALL_MULTIPLIER"), 1.7),
                discharge_multiplier=_coerce_float(row.get("DISCHARGE_MULTIPLIER"), 1.9),
                velocity_multiplier=_coerce_float(row.get("VELOCITY_MULTIPLIER"), 1.45),
                scour_multiplier=_coerce_float(row.get("SCOUR_MULTIPLIER"), 2.05),
                flood_exceedance_increment=int(_coerce_float(row.get("FLOOD_EXCEEDANCE"), 1)),
                cumulative_scour_increment=_coerce_float(
                    row.get("CUMULATIVE_SCOUR_INCREMENT"), 12.0
                ),
                event_year=AS_OF_YEAR,
            )
        )
    return events


class WeatherShockMonitor:
    """Automated IMD + Open-Meteo district shock monitor.

    The UI wraps :meth:`monitor` with ``st.cache_data(ttl=900)``.  The class is
    deliberately stateless so it is safe to cache the serialised result.
    """

    def __init__(self, *, timeout: float = 5.0, max_workers: int = 8) -> None:
        self.timeout = float(timeout)
        self.max_workers = int(max_workers)

    def fetch_imd_bulletins(self, states: Sequence[str]) -> dict[str, Any]:
        return fetch_imd_bulletins(states, timeout=self.timeout)

    def fetch_open_meteo(self, latitude: float, longitude: float) -> dict[str, Any]:
        return fetch_open_meteo(latitude, longitude, timeout=self.timeout)

    def monitor(self, coordinates: pd.DataFrame) -> dict[str, Any]:
        return fetch_live_weather(
            coordinates,
            timeout=self.timeout,
            max_workers=self.max_workers,
        )

    @staticmethod
    def classify(
        rainfall_24h: float,
        rainfall_72h: float,
        weather_code: int,
        imd_level: str,
        flood_prone: float,
    ) -> dict[str, Any]:
        return _classify_weather_shock(
            rainfall_24h,
            rainfall_72h,
            weather_code,
            imd_level,
            flood_prone,
        )


def build_forced_event(
    *,
    state: str,
    district: str,
    severity: float = 1.65,
    monsoon_anomaly: float = 2.8,
    scour_multiplier: float = 2.4,
    flood_exceedance: int = 2,
    rainfall_multiplier: float = 1.9,
) -> Any:
    from bridge_engine.triage_and_xai import DisasterEvent

    return DisasterEvent(
        event_id=f"FORCED-{_slug(state)}-{_slug(district)}",
        event_type="COMPOUND",
        district=str(district),
        state=str(state) if state else None,
        severity=float(severity),
        monsoon_anomaly_increment=float(monsoon_anomaly),
        rainfall_multiplier=float(rainfall_multiplier),
        discharge_multiplier=1.0 + (float(rainfall_multiplier) - 1.0) * 1.2,
        velocity_multiplier=1.0 + (float(scour_multiplier) - 1.0) * 0.55,
        scour_multiplier=float(scour_multiplier),
        flood_exceedance_increment=max(1, int(flood_exceedance)),
        cumulative_scour_increment=float(10.0 + 4.0 * max(0.0, severity - 1.0)),
        event_year=AS_OF_YEAR,
    )


# ---------------------------------------------------------------------------
# Deterministic inspector ledger
# ---------------------------------------------------------------------------


def _pick(values: Sequence[str], indexes: np.ndarray) -> np.ndarray:
    pool = np.asarray(values, dtype=object)
    return pool[np.mod(indexes.astype(np.uint64), np.uint64(len(pool))).astype(int)]


def _seeded_ledger(bridges: pd.DataFrame, inspections: pd.DataFrame | None) -> pd.DataFrame:
    """Build a deterministic Indian engineering governance record per bridge."""

    inventory = bridges.copy()
    inventory["BRIDGE_ID"] = inventory["BRIDGE_ID"].astype(str)
    inventory = inventory.drop_duplicates("BRIDGE_ID").sort_values("BRIDGE_ID").reset_index(drop=True)
    ids = inventory["BRIDGE_ID"].astype(str)
    n = len(inventory)
    h = pd.util.hash_pandas_object(ids, index=False).to_numpy(np.uint64)
    h_cur = pd.util.hash_pandas_object(ids + "|current", index=False).to_numpy(np.uint64)
    h_prev = pd.util.hash_pandas_object(ids + "|previous", index=False).to_numpy(np.uint64)
    h_date = pd.util.hash_pandas_object(ids + "|date", index=False).to_numpy(np.uint64)

    current_first = _pick(FIRST_NAMES, h_cur)
    current_last = _pick(LAST_NAMES, h_cur >> np.uint64(7))
    previous_first = _pick(FIRST_NAMES, h_prev)
    previous_last = _pick(LAST_NAMES, h_prev >> np.uint64(5))

    owner = inventory.get("OWNER", pd.Series("", index=inventory.index)).fillna("").astype(str).str.upper()
    owner_agency = np.select(
        [
            owner.str.contains("NHAI", regex=False),
            owner.str.contains("PWD", regex=False),
            owner.str.contains("BRO", regex=False),
            owner.str.contains("MUNICIPAL", regex=False),
        ],
        ["NHAI", "State PWD", "BRO", "Municipal Corporation"],
        default=None,
    )
    seeded_agency = _pick(MANAGING_AGENCIES, h >> np.uint64(11))
    owner_agency = np.where(pd.isna(owner_agency), "", owner_agency)
    agency = np.where(owner_agency.astype(str) == "", seeded_agency, owner_agency).astype(str)
    agency_code = np.vectorize(AGENCY_CODES.get, otypes=[object])(agency)
    domains = np.vectorize(AGENCY_DOMAINS.get, otypes=[object])(agency)

    latest_year = pd.to_numeric(
        inventory.get("LAST_INSPECTION_YEAR", pd.Series(AS_OF_YEAR, index=inventory.index)),
        errors="coerce",
    ).fillna(float(AS_OF_YEAR)).to_numpy()
    if inspections is not None and not inspections.empty and "BRIDGE_ID" in inspections.columns:
        inspections = inspections.copy()
        inspections["BRIDGE_ID"] = inspections["BRIDGE_ID"].astype(str)
        year_column = "INSPECTION_YEAR" if "INSPECTION_YEAR" in inspections.columns else None
        if year_column is not None:
            latest = (
                inspections.sort_values([year_column], kind="mergesort")
                .groupby("BRIDGE_ID", observed=True, sort=False)
                .tail(1)
                .set_index("BRIDGE_ID")
            )
            mapped = pd.to_numeric(
                latest[year_column].reindex(ids.to_numpy()), errors="coerce"
            ).to_numpy()
            latest_year = np.where(np.isfinite(mapped), mapped, latest_year)

    years = np.clip(latest_year.astype(int), 1990, AS_OF_YEAR)
    months = (h_date % np.uint64(12)).astype(int) + 1
    days = ((h_date >> np.uint64(9)) % np.uint64(28)).astype(int) + 1
    dates = pd.to_datetime(
        pd.DataFrame({"year": years, "month": months, "day": days})
    ).dt.strftime("%Y-%m-%d").to_numpy()

    phone_current = [f"+91-98{int(value) % 100_000_000:08d}" for value in h_cur]
    phone_previous = [f"+91-98{int(value) % 100_000_000:08d}" for value in h_prev]
    current_id = [
        f"INS-{code}-{int(value) % 90_000 + 10_000:05d}"
        for code, value in zip(agency_code, h_cur, strict=False)
    ]
    previous_id = [
        f"INS-{code}-{int(value) % 90_000 + 10_000:05d}"
        for code, value in zip(agency_code, h_prev, strict=False)
    ]
    current_local = [
        f"{first.lower()}.{last.lower()}.{int(value) % 9000 + 1000}"
        for first, last, value in zip(current_first, current_last, h_cur, strict=False)
    ]
    previous_local = [
        f"{first.lower()}.{last.lower()}.{int(value) % 9000 + 1000}"
        for first, last, value in zip(previous_first, previous_last, h_prev, strict=False)
    ]

    reported = pd.to_numeric(
        inventory.get("REPORTED_LAST_MIN_COND", pd.Series(6, index=inventory.index)),
        errors="coerce",
    ).fillna(6).clip(0, 9).astype(int).to_numpy()

    remarks = []
    for condition, digest in zip(reported, h, strict=False):
        pool = REMARKS_HIGH if condition >= 7 else REMARKS_MEDIUM if condition >= 4 else REMARKS_LOW
        remarks.append(pool[int(digest) % len(pool)])

    ledger = pd.DataFrame(
        {
            "BRIDGE_ID": ids.to_numpy(),
            "MANAGING_AGENCY": agency,
            "LAST_INSPECTION_DATE": dates,
            "REPORTED_LAST_MIN_COND": reported,
            "REPORTED_LAST_DECK_COND": reported,
            "REPORTED_LAST_SUPERSTRUCTURE_COND": reported,
            "REPORTED_LAST_SUBSTRUCTURE_COND": reported,
            "LAST_INSPECTION_METHOD": np.where(h % 100 < 62, "Visual Only", "Physical NDT Performed"),
            "CURRENT_INSPECTOR_NAME": [
                f"{first} {last}" for first, last in zip(current_first, current_last, strict=False)
            ],
            "CURRENT_INSPECTOR_ID": current_id,
            "CURRENT_INSPECTOR_PHONE": phone_current,
            "CURRENT_INSPECTOR_EMAIL": [
                f"{local}@{domain}" for local, domain in zip(current_local, domains, strict=False)
            ],
            "PREVIOUS_INSPECTOR_NAME": [
                f"{first} {last}" for first, last in zip(previous_first, previous_last, strict=False)
            ],
            "PREVIOUS_INSPECTOR_ID": previous_id,
            "PREVIOUS_INSPECTOR_PHONE": phone_previous,
            "PREVIOUS_INSPECTOR_EMAIL": [
                f"{local}@{domain}" for local, domain in zip(previous_local, domains, strict=False)
            ],
            "INSPECTOR_REMARKS": remarks,
            "ATTACHMENT_PATH": "",
            "SUBMISSION_COUNT": np.zeros(n, dtype=np.int32),
            "LAST_SUBMISSION_AT": "",
            "RECEIPT_SHA256": "",
        }
    )
    return ledger.reset_index(drop=True)


def load_or_create_ledger(
    bridges: pd.DataFrame,
    inspections: pd.DataFrame | None,
    *,
    path: Path = LEDGER_PATH,
) -> pd.DataFrame:
    """Load the persistent ledger, creating or extending it deterministically."""

    existing: pd.DataFrame | None = None
    try:
        existing = _read_frame_any(path)
    except Exception:
        existing = None
    seeded = _seeded_ledger(bridges, inspections)
    if existing is None or existing.empty:
        _write_frame_any(seeded, path)
        return seeded

    ledger = existing.copy()
    ledger["BRIDGE_ID"] = ledger["BRIDGE_ID"].astype(str)
    missing_columns = [column for column in seeded.columns if column not in ledger.columns]
    for column in missing_columns:
        ledger[column] = seeded.set_index("BRIDGE_ID")[column].reindex(ledger["BRIDGE_ID"]).to_numpy()
    missing_ids = set(seeded["BRIDGE_ID"]).difference(ledger["BRIDGE_ID"])
    if missing_ids:
        additions = seeded.loc[seeded["BRIDGE_ID"].isin(missing_ids)]
        ledger = pd.concat([ledger, additions], ignore_index=True)
    ledger = ledger.drop_duplicates("BRIDGE_ID", keep="first").reset_index(drop=True)
    if missing_ids or missing_columns:
        _write_frame_any(ledger, path)
    return ledger


def ledger_signature(path: Path = LEDGER_PATH) -> str:
    try:
        stat = path.stat()
        return f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    except OSError:
        overlay = path.with_suffix(".json")
        try:
            stat = overlay.stat()
            return f"{int(stat.st_mtime_ns)}-{stat.st_size}"
        except OSError:
            return "missing"


def record_inspection_submission(
    ledger: pd.DataFrame,
    *,
    bridge_id: str,
    reported_condition: int,
    deck_condition: int,
    superstructure_condition: int,
    substructure_condition: int,
    inspection_method: str,
    remarks: str,
    upload_name: str | None,
    upload_bytes: bytes | None,
    root: Path = ROOT,
    path: Path = LEDGER_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Persist a blind field submission, file bytes and ledger update."""

    output = ledger.copy()
    output["BRIDGE_ID"] = output["BRIDGE_ID"].astype(str)
    mask = output["BRIDGE_ID"].eq(str(bridge_id))
    if not mask.any():
        raise ValueError(f"bridge {bridge_id!r} is absent from the inspector ledger")
    index = output.index[mask][0]
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    raw_attachment = output.at[index, "ATTACHMENT_PATH"]
    attachment_path = "" if raw_attachment is None or pd.isna(raw_attachment) else str(raw_attachment)
    file_digest = ""
    if upload_bytes and upload_name:
        bridge_dir = Path(root) / "artifacts" / "uploads" / safe_filename(str(bridge_id))
        bridge_dir.mkdir(parents=True, exist_ok=True)
        timestamp = now.strftime("%Y%m%dT%H%M%S")
        target = bridge_dir / f"{timestamp}_{safe_filename(upload_name)}"
        target.write_bytes(upload_bytes)
        attachment_path = str(target)
        file_digest = hashlib.sha256(upload_bytes).hexdigest()

    previous = {
        "PREVIOUS_INSPECTOR_NAME": output.at[index, "CURRENT_INSPECTOR_NAME"],
        "PREVIOUS_INSPECTOR_ID": output.at[index, "CURRENT_INSPECTOR_ID"],
        "PREVIOUS_INSPECTOR_PHONE": output.at[index, "CURRENT_INSPECTOR_PHONE"],
        "PREVIOUS_INSPECTOR_EMAIL": output.at[index, "CURRENT_INSPECTOR_EMAIL"],
    }
    for column, value in previous.items():
        output.at[index, column] = value

    output.at[index, "REPORTED_LAST_MIN_COND"] = int(reported_condition)
    output.at[index, "REPORTED_LAST_DECK_COND"] = int(deck_condition)
    output.at[index, "REPORTED_LAST_SUPERSTRUCTURE_COND"] = int(superstructure_condition)
    output.at[index, "REPORTED_LAST_SUBSTRUCTURE_COND"] = int(substructure_condition)
    output.at[index, "LAST_INSPECTION_DATE"] = today
    output.at[index, "LAST_INSPECTION_METHOD"] = str(inspection_method)
    output.at[index, "INSPECTOR_REMARKS"] = str(remarks).strip()
    output.at[index, "ATTACHMENT_PATH"] = attachment_path
    output.at[index, "SUBMISSION_COUNT"] = int(_coerce_float(output.at[index, "SUBMISSION_COUNT"], 0.0)) + 1
    output.at[index, "LAST_SUBMISSION_AT"] = now.isoformat(timespec="seconds")

    receipt_source = "|".join(
        [
            str(bridge_id),
            str(int(reported_condition)),
            str(inspection_method),
            str(remarks).strip(),
            today,
            attachment_path,
            file_digest,
            output.at[index, "CURRENT_INSPECTOR_ID"],
            now.isoformat(timespec="seconds"),
        ]
    )
    receipt = hashlib.sha256(receipt_source.encode("utf-8")).hexdigest()
    output.at[index, "RECEIPT_SHA256"] = receipt
    _write_frame_any(output, path)
    record = {
        "BRIDGE_ID": str(bridge_id),
        "REPORTED_LAST_MIN_COND": int(reported_condition),
        "LAST_INSPECTION_DATE": today,
        "ATTACHMENT_PATH": attachment_path,
        "RECEIPT_SHA256": receipt,
        "SUBMITTED_AT": now.isoformat(timespec="seconds"),
        "INSPECTOR_NAME": str(output.at[index, "CURRENT_INSPECTOR_NAME"]),
    }
    return output, record


# ---------------------------------------------------------------------------
# Runtime artefact loading
# ---------------------------------------------------------------------------


def load_runtime_bundle(root: Path = ROOT) -> dict[str, Any]:
    """Load persisted models, data and reports once for the Streamlit process."""

    from bridge_engine.survival_models import load_processed_datasets
    from bridge_engine.triage_and_xai import load_engine_result

    root = Path(root)
    artifacts = root / "artifacts"
    data_dir = artifacts / "data"
    module2 = artifacts / "module2"
    module3 = artifacts / "module3"
    module4 = artifacts / "module4"

    runtime: dict[str, Any] = {
        "root": root,
        "artifacts": artifacts,
        "data_dir": data_dir,
        "module2_dir": module2,
        "module3_dir": module3,
        "module4_dir": module4,
        "errors": [],
        "portfolio": None,
        "portfolio_error": "",
    }

    def attempt(name: str, loader: Any) -> Any:
        try:
            return loader()
        except Exception as exc:
            runtime["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    runtime["bridges"] = attempt("bridges.parquet", lambda: pd.read_parquet(data_dir / "bridges.parquet"))
    runtime["climate"] = attempt("climate.parquet", lambda: pd.read_parquet(data_dir / "climate.parquet"))
    runtime["inspections"] = attempt(
        "inspections.parquet", lambda: pd.read_parquet(data_dir / "inspections.parquet")
    )
    runtime["engine_bundle"] = attempt(
        "survival_engine.joblib", lambda: joblib.load(module2 / "survival_engine.joblib")
    )
    runtime["engine_result"] = attempt(
        "SurvivalEngineResult", lambda: load_engine_result(module2)
    )
    runtime["datasets"] = attempt(
        "processed datasets", lambda: load_processed_datasets(data_dir)
    )
    runtime["audit_bundle"] = attempt(
        "audit_models.joblib", lambda: joblib.load(module3 / "audit_models.joblib")
    )
    runtime["tv_preprocessor"] = attempt(
        "time_varying_preprocessor.joblib",
        lambda: joblib.load(data_dir / "time_varying_preprocessor.joblib"),
    )
    runtime["static_preprocessor"] = attempt(
        "static_preprocessor.joblib",
        lambda: joblib.load(data_dir / "static_preprocessor.joblib"),
    )
    runtime["audit_findings"] = attempt(
        "audit_findings.parquet", lambda: pd.read_parquet(module3 / "audit_findings.parquet")
    )
    runtime["triage_findings"] = attempt(
        "triage_findings.parquet", lambda: pd.read_parquet(module4 / "triage_findings.parquet")
    )
    runtime["xai_contributions"] = attempt(
        "xai_contributions.parquet", lambda: pd.read_parquet(module4 / "xai_contributions.parquet")
    )
    runtime["summary"] = attempt(
        "end_to_end_summary.json", lambda: _json_load(artifacts / "end_to_end_summary.json")
    )
    runtime["survival_report"] = attempt(
        "survival_report.json", lambda: _json_load(module2 / "survival_report.json")
    )
    runtime["archetypes"] = attempt(
        "archetype_validation.json", lambda: _json_load(module4 / "archetype_validation.json")
    )
    runtime["audit_report"] = attempt(
        "audit_report.json", lambda: _json_load(module3 / "audit_report.json")
    )
    if runtime["bridges"] is None:
        runtime["errors"].append("bridges.parquet is required for the portal.")
    return runtime


def selected_model_bundle(runtime: Mapping[str, Any]) -> tuple[str, Any, tuple[str, ...]]:
    bundle = runtime.get("engine_bundle") or {}
    models = bundle.get("models", {}) if isinstance(bundle, Mapping) else {}
    model_name = str(bundle.get("selected_model", "cox_time_varying")) if isinstance(bundle, Mapping) else "cox_time_varying"
    if model_name not in models and "cox_time_varying" in models:
        model_name = "cox_time_varying"
    model = models.get(model_name) if isinstance(models, Mapping) else None
    if model_name == "cox_time_varying":
        features = tuple(bundle.get("time_varying_feature_names", ())) if isinstance(bundle, Mapping) else ()
    else:
        features = tuple(bundle.get("static_feature_names", ())) if isinstance(bundle, Mapping) else ()
    return model_name, model, features


def _portfolio_from_test_split(runtime: Mapping[str, Any]) -> dict[str, Any] | None:
    from bridge_engine.triage_and_xai import conditional_survival_curve, compute_rul_from_survival

    datasets = runtime.get("datasets")
    model_name, model, features = selected_model_bundle(runtime)
    if datasets is None or model is None:
        return None
    if model_name == "cox_time_varying":
        frame = getattr(datasets, "counting_test", pd.DataFrame())
        if frame.empty:
            return None
        frame = (
            frame.sort_values(["id", "start"], kind="mergesort")
            .groupby("id", observed=True, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        id_col, time_col = "id", "stop"
    else:
        frame = getattr(datasets, "static_test", pd.DataFrame())
        if frame.empty:
            return None
        frame = frame.copy()
        id_col, time_col = "bridge_id", "duration"
    time_at_score = frame[time_col].to_numpy(np.float64)
    curve = conditional_survival_curve(
        model_name=model_name,
        model=model,
        frame=frame,
        feature_names=features,
        time_at_event=time_at_score,
        horizons=CURVE_GRID,
        survival_floor=SURVIVAL_FLOOR,
    )
    rul, censored = compute_rul_from_survival(
        curve, CURVE_GRID, threshold=SURVIVAL_THRESHOLD, censor_at=RUL_HORIZON
    )
    bridge_ids = frame[id_col].astype(str).to_numpy()
    baseline = pd.DataFrame(
        {
            "BRIDGE_ID": bridge_ids,
            "TIME_AT_SCORE_YEARS": time_at_score.astype(np.float32),
            "BASELINE_SURVIVAL_5Y": curve[:, int(DECISION_HORIZON)].astype(np.float32),
            "BASELINE_RUL_YEARS": rul.astype(np.float32),
            "BASELINE_RUL_CENSORED": censored,
        }
    )
    return {
        "frame": frame,
        "id_col": id_col,
        "time_col": time_col,
        "bridge_ids": bridge_ids,
        "curve": curve,
        "rul": rul,
        "baseline": baseline,
        "model_name": model_name,
        "model": model,
        "features": features,
        "source": "held-out test split fallback",
    }


def build_portfolio(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Build and score the full persisted bridge inventory.

    Raw inventory rows are rebuilt into the same counting-process design used in
    training and transformed with the persisted preprocessor.  This keeps the
    master queue representative of every bridge rather than only the test split.
    """

    from bridge_engine.data_pipeline import (
        AndersenGillSplitter,
        attach_dynamic_covariates,
        build_counting_process_frame,
        build_static_survival_frame,
    )
    from bridge_engine.triage_and_xai import conditional_survival_curve, compute_rul_from_survival

    bridges = runtime.get("bridges")
    climate = runtime.get("climate")
    tv_preprocessor = runtime.get("tv_preprocessor")
    static_preprocessor = runtime.get("static_preprocessor")
    model_name, model, features = selected_model_bundle(runtime)
    if bridges is None or bridges.empty or model is None:
        fallback = _portfolio_from_test_split(runtime)
        if fallback is not None:
            return fallback
        raise ValueError("survival model or bridge inventory is unavailable")

    try:
        roster = bridges.copy()
        roster["BRIDGE_ID"] = roster["BRIDGE_ID"].astype(str)
        roster["ENTRY_YEAR"] = pd.to_numeric(roster["ENTRY_YEAR"], errors="coerce").fillna(AS_OF_YEAR - 20).astype(int)
        roster["DURATION_YEARS"] = pd.to_numeric(
            roster["DURATION_YEARS"], errors="coerce"
        ).clip(lower=1).astype(float)
        roster["EVENT"] = pd.to_numeric(roster["EVENT"], errors="coerce").fillna(0).astype(int)
        if model_name == "cox_time_varying":
            if tv_preprocessor is None or climate is None:
                raise ValueError("time-varying preprocessor or climate panel is unavailable")
            intervals = AndersenGillSplitter(1.0).split(roster)
            intervals = attach_dynamic_covariates(intervals, climate, interval_width=1.0)
            counting_raw = build_counting_process_frame(intervals)
            frame = (
                counting_raw.sort_values(["id", "start"], kind="mergesort")
                .groupby("id", observed=True, sort=False)
                .tail(1)
                .reset_index(drop=True)
            )
            id_col, time_col = "id", "stop"
            transformed = tv_preprocessor.transform(frame)
            frame = pd.concat(
                [frame.loc[:, [id_col, time_col]].reset_index(drop=True), transformed.reset_index(drop=True)],
                axis=1,
            )
        else:
            if static_preprocessor is None or climate is None:
                raise ValueError("static preprocessor or climate panel is unavailable")
            static_raw = build_static_survival_frame(roster, climate)
            frame = static_raw.rename(
                columns={
                    "BRIDGE_ID": "bridge_id",
                    "DURATION_YEARS": "duration",
                    "EVENT": "event",
                }
            )
            id_col, time_col = "bridge_id", "duration"
            transformed = static_preprocessor.transform(frame)
            frame = pd.concat(
                [frame.loc[:, [id_col, time_col]].reset_index(drop=True), transformed.reset_index(drop=True)],
                axis=1,
            )
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"transformed portfolio frame is missing features: {missing[:6]}")
        time_at_score = frame[time_col].to_numpy(np.float64)
        curve = conditional_survival_curve(
            model_name=model_name,
            model=model,
            frame=frame,
            feature_names=features,
            time_at_event=time_at_score,
            horizons=CURVE_GRID,
            survival_floor=SURVIVAL_FLOOR,
        )
        rul, censored = compute_rul_from_survival(
            curve, CURVE_GRID, threshold=SURVIVAL_THRESHOLD, censor_at=RUL_HORIZON
        )
        bridge_ids = frame[id_col].astype(str).to_numpy()
        baseline = pd.DataFrame(
            {
                "BRIDGE_ID": bridge_ids,
                "TIME_AT_SCORE_YEARS": time_at_score.astype(np.float32),
                "BASELINE_SURVIVAL_5Y": curve[:, int(DECISION_HORIZON)].astype(np.float32),
                "BASELINE_RUL_YEARS": rul.astype(np.float32),
                "BASELINE_RUL_CENSORED": censored,
            }
        )
        return {
            "frame": frame,
            "id_col": id_col,
            "time_col": time_col,
            "bridge_ids": bridge_ids,
            "curve": curve,
            "rul": rul,
            "baseline": baseline,
            "model_name": model_name,
            "model": model,
            "features": features,
            "source": "full persisted inventory",
        }
    except Exception as exc:
        fallback = _portfolio_from_test_split(runtime)
        if fallback is not None:
            fallback["source"] = f"held-out test split fallback ({type(exc).__name__})"
            return fallback
        raise


def get_portfolio(runtime: Mapping[str, Any]) -> dict[str, Any]:
    cached = runtime.get("portfolio")
    if cached is not None:
        return cached
    try:
        portfolio = build_portfolio(runtime)
    except Exception as exc:
        if isinstance(runtime, dict):
            runtime["portfolio_error"] = f"{type(exc).__name__}: {exc}"
            runtime["portfolio"] = {}
        return {}
    if isinstance(runtime, dict):
        runtime["portfolio"] = portfolio
    return portfolio


# ---------------------------------------------------------------------------
# Module 4 regional triage
# ---------------------------------------------------------------------------


def _event_bridge_ids(bridges: pd.DataFrame, event: Any) -> list[str]:
    if bridges is None or bridges.empty:
        return []
    ids = bridges["BRIDGE_ID"].astype(str)
    if getattr(event, "bridge_ids", ()):
        return ids[ids.isin(set(event.bridge_ids))].tolist()
    mask = pd.Series(True, index=bridges.index)
    if getattr(event, "district", None) is not None:
        mask &= bridges["DISTRICT"].astype(str).eq(str(event.district))
    if getattr(event, "state", None) is not None:
        mask &= bridges["STATE"].astype(str).eq(str(event.state))
    return ids[mask].tolist()


def run_module4_regions(
    runtime: Mapping[str, Any],
    events: Sequence[Any],
    *,
    max_regions: int = 3,
    top_n_explanations: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """Run Module 4 for the highest-severity affected districts.

    The portfolio-wide queue additionally applies the same Module 4
    ``PostDisasterUpdater`` shock to every affected bridge, so ranking coverage
    is not limited by the held-out split used for SHAP explanations.
    """

    from bridge_engine.triage_and_xai import TriageConfig, TriageEngine

    datasets = runtime.get("datasets")
    engine_result = runtime.get("engine_result")
    bridges = runtime.get("bridges")
    if datasets is None or engine_result is None or bridges is None:
        return pd.DataFrame(), ["Module 4 runtime artefacts are unavailable."]

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    config = TriageConfig(
        model_name="cox_time_varying",
        explain_model_name="ensemble",
        split="test",
        top_n_explanations=int(max(0, top_n_explanations)),
        save_force_plots=True,
        shap_background_size=80,
        shap_max_evals=384,
    )
    try:
        engine = TriageEngine(config)
    except Exception as exc:
        return pd.DataFrame(), [f"TriageEngine initialisation: {type(exc).__name__}: {exc}"]
    output_dir = Path(runtime.get("module4_dir", MODULE4_DIR)) / "live"
    output_dir.mkdir(parents=True, exist_ok=True)
    for event in list(events)[: max(1, int(max_regions))]:
        try:
            result = engine.run_event(
                datasets=datasets,
                engine_result=engine_result,
                bridges=bridges,
                event=event,
                audit_result=None,
                output_dir=output_dir,
            )
            table = result.triage_table.copy()
            table["LIVE_EVENT_ID"] = str(event.event_id)
            frames.append(table)
        except Exception as exc:
            errors.append(f"{getattr(event, 'event_id', 'event')}: {type(exc).__name__}: {exc}")
    if not frames:
        return pd.DataFrame(), errors
    combined = pd.concat(frames, ignore_index=True)
    combined["BRIDGE_ID"] = combined["BRIDGE_ID"].astype(str)
    combined = combined.sort_values(
        ["TRIAGE_PRIORITY_SCORE", "BRIDGE_ID"], ascending=[False, True], kind="mergesort"
    )
    combined = combined.drop_duplicates("BRIDGE_ID", keep="first").reset_index(drop=True)
    _write_frame_any(combined, output_dir / "live_triage_findings.parquet")
    return combined, errors


def read_live_triage() -> pd.DataFrame:
    try:
        return _read_frame_any(LIVE_TRIAGE_PATH)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Incremental Module 3 audit scoring
# ---------------------------------------------------------------------------


def read_audit_overlay() -> pd.DataFrame:
    try:
        overlay = _read_frame_any(AUDIT_OVERLAY_PATH)
    except Exception:
        return pd.DataFrame()
    if "BRIDGE_ID" in overlay.columns:
        overlay["BRIDGE_ID"] = overlay["BRIDGE_ID"].astype(str)
    return overlay


def audit_overlay_signature(path: Path = AUDIT_OVERLAY_PATH) -> str:
    try:
        stat = path.stat()
        return f"{int(stat.st_mtime_ns)}-{stat.st_size}"
    except OSError:
        fallback = path.with_suffix(".json")
        try:
            stat = fallback.stat()
            return f"{int(stat.st_mtime_ns)}-{stat.st_size}"
        except OSError:
            return "missing"


def _write_audit_overlay(row: Mapping[str, Any]) -> None:
    overlay = read_audit_overlay()
    if not overlay.empty and "BRIDGE_ID" in overlay.columns:
        overlay = overlay.loc[overlay["BRIDGE_ID"].astype(str).ne(str(row.get("BRIDGE_ID")))]
    overlay = pd.concat([overlay, pd.DataFrame([dict(row)])], ignore_index=True)
    _write_frame_any(overlay, AUDIT_OVERLAY_PATH)


def _audit_config() -> Any:
    from bridge_engine.audit_verifier import AuditVerifierConfig

    return AuditVerifierConfig()


def _audit_model_parts(runtime: Mapping[str, Any]) -> tuple[Any, Any, Any, tuple[str, ...]] | None:
    from bridge_engine.audit_verifier import AuditCalibration

    bundle = runtime.get("audit_bundle")
    if not isinstance(bundle, Mapping):
        return None
    try:
        calibration = AuditCalibration(**bundle["calibration"])
        return (
            calibration,
            bundle["feature_pipeline"],
            bundle["isolation_forest"],
            tuple(bundle["audit_feature_columns"]),
        )
    except Exception:
        return None


def rescore_submitted_bridge(
    runtime: Mapping[str, Any],
    *,
    bridge_id: str,
    reported_condition: int,
    inspection_method: str | None = None,
    receipt: str = "",
) -> dict[str, Any]:
    """Re-run Module 3 audit scoring for one newly submitted inspection.

    The function is intentionally deterministic and returns only machine
    fields.  The inspector portal never renders any part of this result.
    """

    from bridge_engine.audit_verifier import _apply_audit_scores, _prioritise, build_audit_metadata
    from bridge_engine.triage_and_xai import conditional_survival_curve

    config = _audit_config()
    bridges = runtime.get("bridges")
    climate = runtime.get("climate")
    if bridges is None or climate is None:
        raise ValueError("bridge and climate artefacts are unavailable")
    bridge_mask = bridges["BRIDGE_ID"].astype(str).eq(str(bridge_id))
    if not bridge_mask.any():
        raise ValueError(f"bridge {bridge_id!r} is absent from bridges.parquet")

    updated_bridges = bridges.copy()
    updated_bridges.loc[bridge_mask, "REPORTED_LAST_MIN_COND"] = int(reported_condition)
    if inspection_method is not None and "LAST_INSPECTION_WAS_NDT" in updated_bridges.columns:
        updated_bridges.loc[bridge_mask, "LAST_INSPECTION_WAS_NDT"] = int(
            str(inspection_method).lower().startswith("physical")
        )
    metadata = build_audit_metadata(updated_bridges, climate)
    metadata = metadata.loc[metadata["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
    if metadata.empty:
        raise ValueError("audit metadata could not be constructed for this bridge")

    model_name, model, features = selected_model_bundle(runtime)
    expected_survival = np.nan
    time_at_audit = float("nan")
    split_name = "portfolio"
    datasets = runtime.get("datasets")
    if model_name == "cox_time_varying" and datasets is not None:
        for split in ("test", "validation", "train"):
            frame = getattr(datasets, f"counting_{split}", pd.DataFrame())
            if frame is None or frame.empty:
                continue
            subset = frame.loc[frame["id"].astype(str).eq(str(bridge_id))]
            if subset.empty:
                continue
            latest = subset.sort_values(["start"], kind="mergesort").tail(1).reset_index(drop=True)
            time_at_audit = float(latest["stop"].iloc[0])
            curve = conditional_survival_curve(
                model_name=model_name,
                model=model,
                frame=latest,
                feature_names=features,
                time_at_event=np.asarray([time_at_audit], dtype=np.float64),
                horizons=np.asarray([0.0, DECISION_HORIZON], dtype=np.float64),
                survival_floor=SURVIVAL_FLOOR,
            )
            expected_survival = float(curve[0, 1])
            split_name = split
            break

    if not np.isfinite(expected_survival):
        portfolio = get_portfolio(runtime)
        baseline = portfolio.get("baseline", pd.DataFrame()) if portfolio else pd.DataFrame()
        if baseline is not None and not baseline.empty:
            match = baseline.loc[baseline["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
            if not match.empty:
                expected_survival = float(match.iloc[0]["BASELINE_SURVIVAL_5Y"])
                time_at_audit = float(match.iloc[0]["TIME_AT_SCORE_YEARS"])
    if not np.isfinite(expected_survival):
        persisted = runtime.get("audit_findings")
        if persisted is not None and not persisted.empty:
            match = persisted.loc[persisted["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
            if not match.empty:
                expected_survival = float(_coerce_float(match.iloc[0].get("EXPECTED_SURVIVAL"), 0.5))
    if not np.isfinite(expected_survival):
        expected_survival = 0.5

    expected_survival = float(np.clip(expected_survival, 0.0, 1.0))
    expected_condition = float(np.clip(9.0 * expected_survival, 0.0, 9.0))
    condition_gap = float(int(reported_condition) - expected_condition)
    table = metadata.copy()
    table["DATA_SPLIT"] = split_name
    table["TIME_AT_AUDIT"] = time_at_audit
    table["SURVIVAL_AT_AUDIT"] = 1.0
    table["SURVIVAL_AT_HORIZON"] = expected_survival
    table["EXPECTED_SURVIVAL"] = expected_survival
    table["EXPECTED_DEGRADATION_HAZARD"] = -math.log(max(expected_survival, SURVIVAL_FLOOR)) / DECISION_HORIZON
    table["AUDIT_HORIZON_YEARS"] = DECISION_HORIZON
    table["PHYSICS_EXPECTED_CONDITION"] = expected_condition
    table["CONDITION_GAP"] = condition_gap
    table["SURVIVAL_DEFICIT"] = max(0.0, config.critical_survival_threshold - expected_survival)
    table["REPORTED_HEALTHY"] = (
        int(config.healthy_condition_min) <= int(reported_condition) <= int(config.healthy_condition_max)
    )
    table["PAPER_INSPECTION_SUSPECT"] = bool(
        table["REPORTED_HEALTHY"].iloc[0]
        and expected_survival < config.critical_survival_threshold
    )

    model_parts = _audit_model_parts(runtime)
    if model_parts is not None:
        calibration, pipeline, forest, feature_columns = model_parts
        missing = [column for column in feature_columns if column not in table.columns]
        if missing:
            raise ValueError(f"audit feature construction failed: {missing}")
        table = _apply_audit_scores(
            table,
            calibration=calibration,
            feature_pipeline=pipeline,
            isolation_forest=forest,
            feature_columns=feature_columns,
            config=config,
        )
        table = _prioritise(table, config)
    else:
        table["AUDIT_Z_SCORE"] = 0.0
        table["ISOLATION_ANOMALY_SCORE"] = 0.0
        table["ISOLATION_FLAG"] = False
        table["AUDIT_ALERT"] = table["PAPER_INSPECTION_SUSPECT"]
        table["AUDIT_PRIORITY_SCORE"] = 0.0
        table["AUDIT_RANK"] = 1
        table["AUDIT_PRIORITY"] = np.where(table["AUDIT_ALERT"], "HIGH", "LOW")

    row = table.iloc[0].to_dict()
    row["BRIDGE_ID"] = str(bridge_id)
    row["REPORTED_LAST_MIN_COND"] = int(reported_condition)
    row["LIVE_SUBMISSION"] = True
    row["RESCORED_AT"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["SUBMISSION_RECEIPT_SHA256"] = str(receipt)
    _write_audit_overlay(row)
    return row


def build_audit_table(runtime: Mapping[str, Any], ledger: pd.DataFrame) -> pd.DataFrame:
    """Merge persisted audit rows, live rescores and full-inventory fallbacks."""

    from bridge_engine.audit_verifier import _apply_audit_scores, _prioritise, build_audit_metadata

    bridges = runtime.get("bridges")
    climate = runtime.get("climate")
    if bridges is None or bridges.empty:
        return pd.DataFrame()

    persisted = runtime.get("audit_findings")
    overlay = read_audit_overlay()
    parts: list[pd.DataFrame] = []
    if overlay is not None and not overlay.empty:
        parts.append(overlay.copy())
    if persisted is not None and not persisted.empty:
        parts.append(persisted.copy())

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not combined.empty:
        combined["BRIDGE_ID"] = combined["BRIDGE_ID"].astype(str)
        combined = combined.drop_duplicates("BRIDGE_ID", keep="first")

    portfolio = get_portfolio(runtime)
    baseline = portfolio.get("baseline", pd.DataFrame()) if portfolio else pd.DataFrame()
    if baseline is not None and not baseline.empty:
        try:
            inventory = bridges.copy()
            inventory["BRIDGE_ID"] = inventory["BRIDGE_ID"].astype(str)
            ledger_conditions = (
                ledger.set_index("BRIDGE_ID")["REPORTED_LAST_MIN_COND"]
                .astype(float)
                .to_dict()
            )
            if ledger_conditions:
                inventory["REPORTED_LAST_MIN_COND"] = (
                    inventory["BRIDGE_ID"].map(ledger_conditions)
                    .fillna(inventory["REPORTED_LAST_MIN_COND"])
                    .astype(int)
                )
            metadata = build_audit_metadata(inventory, climate)
            expected = baseline.loc[:, ["BRIDGE_ID", "BASELINE_SURVIVAL_5Y", "TIME_AT_SCORE_YEARS"]].copy()
            expected["BRIDGE_ID"] = expected["BRIDGE_ID"].astype(str)
            expected = expected.rename(
                columns={
                    "BASELINE_SURVIVAL_5Y": "EXPECTED_SURVIVAL",
                    "TIME_AT_SCORE_YEARS": "TIME_AT_AUDIT",
                }
            )
            expected["EXPECTED_SURVIVAL"] = pd.to_numeric(
                expected["EXPECTED_SURVIVAL"], errors="coerce"
            ).clip(0.0, 1.0)
            physics = expected.merge(metadata, on="BRIDGE_ID", how="inner", validate="one_to_one")
            physics["DATA_SPLIT"] = "portfolio"
            physics["SURVIVAL_AT_AUDIT"] = 1.0
            physics["SURVIVAL_AT_HORIZON"] = physics["EXPECTED_SURVIVAL"]
            physics["EXPECTED_DEGRADATION_HAZARD"] = (
                -np.log(np.maximum(physics["EXPECTED_SURVIVAL"], SURVIVAL_FLOOR)) / DECISION_HORIZON
            )
            physics["AUDIT_HORIZON_YEARS"] = DECISION_HORIZON
            physics["PHYSICS_EXPECTED_CONDITION"] = np.clip(
                9.0 * physics["EXPECTED_SURVIVAL"].to_numpy(np.float64), 0.0, 9.0
            )
            physics["CONDITION_GAP"] = (
                physics["REPORTED_LAST_MIN_COND"].to_numpy(np.float64)
                - physics["PHYSICS_EXPECTED_CONDITION"].to_numpy(np.float64)
            )
            physics["SURVIVAL_DEFICIT"] = np.maximum(
                0.0, SURVIVAL_THRESHOLD - physics["EXPECTED_SURVIVAL"].to_numpy(np.float64)
            )
            physics["REPORTED_HEALTHY"] = physics["REPORTED_LAST_MIN_COND"].between(7, 9)
            physics["PAPER_INSPECTION_SUSPECT"] = (
                physics["REPORTED_HEALTHY"].to_numpy(bool)
                & (physics["EXPECTED_SURVIVAL"].to_numpy(np.float64) < SURVIVAL_THRESHOLD)
            )
            model_parts = _audit_model_parts(runtime)
            if model_parts is None:
                physics["AUDIT_Z_SCORE"] = 0.0
                physics["ISOLATION_ANOMALY_SCORE"] = 0.0
                physics["ISOLATION_FLAG"] = False
                physics["AUDIT_ALERT"] = physics["PAPER_INSPECTION_SUSPECT"]
                physics["AUDIT_PRIORITY_SCORE"] = 0.0
                physics["AUDIT_RANK"] = np.arange(1, len(physics) + 1)
                physics["AUDIT_PRIORITY"] = np.where(physics["AUDIT_ALERT"], "HIGH", "LOW")
            else:
                calibration, pipeline, forest, feature_columns = model_parts
                missing = [column for column in feature_columns if column not in physics.columns]
                if missing:
                    raise ValueError(f"audit feature construction failed: {missing}")
                physics = _apply_audit_scores(
                    physics,
                    calibration=calibration,
                    feature_pipeline=pipeline,
                    isolation_forest=forest,
                    feature_columns=feature_columns,
                    config=_audit_config(),
                )
                physics = _prioritise(physics, _audit_config())
            if combined.empty:
                combined = physics
            else:
                combined = pd.concat([combined, physics], ignore_index=True)
                combined["BRIDGE_ID"] = combined["BRIDGE_ID"].astype(str)
                combined = combined.drop_duplicates("BRIDGE_ID", keep="first")
        except Exception as exc:
            message = f"full-inventory audit fallback: {type(exc).__name__}: {exc}"
            errors = runtime.setdefault("errors", []) if isinstance(runtime, dict) else []
            if message not in errors:
                errors.append(message)

    if combined.empty:
        return combined

    combined = combined.copy()
    combined["BRIDGE_ID"] = combined["BRIDGE_ID"].astype(str)

    ledger_columns = [
        column
        for column in (
            "BRIDGE_ID",
            "MANAGING_AGENCY",
            "LAST_INSPECTION_DATE",
            "REPORTED_LAST_MIN_COND",
            "INSPECTOR_REMARKS",
            "ATTACHMENT_PATH",
            "CURRENT_INSPECTOR_NAME",
            "CURRENT_INSPECTOR_ID",
            "CURRENT_INSPECTOR_PHONE",
            "CURRENT_INSPECTOR_EMAIL",
            "PREVIOUS_INSPECTOR_NAME",
            "PREVIOUS_INSPECTOR_ID",
            "PREVIOUS_INSPECTOR_PHONE",
            "PREVIOUS_INSPECTOR_EMAIL",
            "SUBMISSION_COUNT",
            "LAST_SUBMISSION_AT",
            "RECEIPT_SHA256",
        )
        if column in ledger.columns
    ]
    ledger_view = ledger.loc[:, ledger_columns].copy()
    ledger_view["BRIDGE_ID"] = ledger_view["BRIDGE_ID"].astype(str)
    combined = combined.drop(
        columns=[column for column in ledger_view.columns if column != "BRIDGE_ID" and column in combined.columns],
        errors="ignore",
    )
    combined = combined.merge(ledger_view, on="BRIDGE_ID", how="left", validate="one_to_one")

    location_columns = [
        column for column in ("BRIDGE_ID", "STATE", "DISTRICT") if column in bridges.columns
    ]
    if len(location_columns) == 3:
        locations = bridges.loc[:, location_columns].copy()
        locations["BRIDGE_ID"] = locations["BRIDGE_ID"].astype(str)
        locations = locations.drop_duplicates("BRIDGE_ID")
        combined = combined.drop(
            columns=[column for column in ("STATE", "DISTRICT") if column in combined.columns],
            errors="ignore",
        )
        combined = combined.merge(locations, on="BRIDGE_ID", how="left", validate="one_to_one")

    def boolean_column(name: str) -> pd.Series:
        values = combined.get(name, pd.Series(False, index=combined.index))
        return values.map(lambda value: _coerce_bool(value, False))

    def numeric_column(name: str, default: float = 0.0) -> pd.Series:
        values = combined.get(name, pd.Series(default, index=combined.index))
        return pd.to_numeric(values, errors="coerce").fillna(default)

    combined["PAPER_INSPECTION_SUSPECT"] = boolean_column("PAPER_INSPECTION_SUSPECT")
    combined["AUDIT_ALERT"] = boolean_column("AUDIT_ALERT")
    combined["ISOLATION_FLAG"] = boolean_column("ISOLATION_FLAG")
    combined["AUDIT_Z_SCORE"] = numeric_column("AUDIT_Z_SCORE")
    combined["AUDIT_PRIORITY_SCORE"] = numeric_column("AUDIT_PRIORITY_SCORE")
    combined["EXPECTED_SURVIVAL"] = numeric_column("EXPECTED_SURVIVAL")
    combined["PHYSICS_EXPECTED_CONDITION"] = numeric_column("PHYSICS_EXPECTED_CONDITION")
    combined["CONDITION_GAP"] = numeric_column("CONDITION_GAP")
    combined["REPORTED_LAST_MIN_COND"] = numeric_column("REPORTED_LAST_MIN_COND").astype(int)
    combined = combined.sort_values(
        ["AUDIT_Z_SCORE", "AUDIT_PRIORITY_SCORE", "BRIDGE_ID"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    combined = combined.drop(columns=["AUDIT_QUEUE_RANK"], errors="ignore")
    combined.insert(0, "AUDIT_QUEUE_RANK", np.arange(1, len(combined) + 1, dtype=np.int32))
    return combined


# ---------------------------------------------------------------------------
# Revised repair queue and bridge curves
# ---------------------------------------------------------------------------


def _apply_events_to_frame(
    frame: pd.DataFrame,
    runtime: Mapping[str, Any],
    events: Sequence[Any],
    *,
    id_col: str,
) -> pd.DataFrame:
    from bridge_engine.triage_and_xai import PostDisasterUpdater

    shocked = frame.copy()
    bridges = runtime.get("bridges")
    if bridges is None or not events:
        return shocked
    updater = PostDisasterUpdater()
    for event in events:
        ids = _event_bridge_ids(bridges, event)
        available = set(shocked[id_col].astype(str))
        target_ids = [value for value in ids if value in available]
        if not target_ids:
            continue
        shocked = updater.apply(shocked, event, affected_ids=target_ids, id_col=id_col)
    return shocked


def build_master_queue(
    runtime: Mapping[str, Any],
    ledger: pd.DataFrame,
    audit_table: pd.DataFrame,
    *,
    events: Sequence[Any] | None = None,
    live_triage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank every bridge by post-event triage priority and net RUL loss."""

    from bridge_engine.triage_and_xai import conditional_survival_curve, compute_rul_from_survival

    portfolio = get_portfolio(runtime)
    if not portfolio:
        return pd.DataFrame()
    frame = portfolio["frame"]
    id_col = portfolio["id_col"]
    time_col = portfolio["time_col"]
    model_name = portfolio["model_name"]
    model = portfolio["model"]
    features = portfolio["features"]
    baseline_curve = portfolio["curve"]
    baseline_rul = portfolio["rul"]
    baseline_survival = baseline_curve[:, int(DECISION_HORIZON)]
    assets = portfolio["baseline"].copy()
    assets["BRIDGE_ID"] = assets["BRIDGE_ID"].astype(str)

    active_events = list(events or [])
    if active_events:
        shocked = _apply_events_to_frame(frame, runtime, active_events, id_col=id_col)
        time_at_score = shocked[time_col].to_numpy(np.float64)
        post_curve = conditional_survival_curve(
            model_name=model_name,
            model=model,
            frame=shocked,
            feature_names=features,
            time_at_event=time_at_score,
            horizons=CURVE_GRID,
            survival_floor=SURVIVAL_FLOOR,
        )
        post_rul, post_censored = compute_rul_from_survival(
            post_curve, CURVE_GRID, threshold=SURVIVAL_THRESHOLD, censor_at=RUL_HORIZON
        )
    else:
        post_curve = baseline_curve
        post_rul = baseline_rul
        post_censored = assets["BASELINE_RUL_CENSORED"].to_numpy(bool)

    post_survival = post_curve[:, int(DECISION_HORIZON)]
    rul_loss = np.maximum(baseline_rul - post_rul, 0.0)
    risk_post = np.clip(1.0 - post_survival, 0.0, 1.0)
    risk_increase = np.clip(baseline_survival - post_survival, 0.0, 1.0)
    rul_loss_ratio = np.clip(rul_loss / np.maximum(baseline_rul, 1.0), 0.0, 1.0)
    priority_score = 0.50 * risk_post + 0.30 * risk_increase + 0.20 * rul_loss_ratio

    queue = pd.DataFrame(
        {
            "BRIDGE_ID": assets["BRIDGE_ID"].to_numpy(),
            "TIME_AT_EVENT_YEARS": assets["TIME_AT_SCORE_YEARS"].to_numpy(),
            "SURVIVAL_5Y": post_survival.astype(np.float32),
            "BASELINE_SURVIVAL_5Y": baseline_survival.astype(np.float32),
            "POST_EVENT_RUL_YEARS": post_rul.astype(np.float32),
            "BASELINE_RUL_YEARS": baseline_rul.astype(np.float32),
            "RUL_LOSS_YEARS": rul_loss.astype(np.float32),
            "POST_EVENT_RUL_CENSORED": post_censored,
            "TRIAGE_PRIORITY_SCORE": priority_score.astype(np.float32),
        }
    )

    audit = audit_table.copy() if audit_table is not None else pd.DataFrame()
    if not audit.empty and "BRIDGE_ID" in audit.columns:
        audit = audit.copy()
        audit["BRIDGE_ID"] = audit["BRIDGE_ID"].astype(str)
        rename = {
            "EXPECTED_SURVIVAL": "AUDIT_EXPECTED_SURVIVAL",
            "PHYSICS_EXPECTED_CONDITION": "PHYSICS_EXPECTED_CONDITION",
            "CONDITION_GAP": "CONDITION_GAP",
            "AUDIT_Z_SCORE": "AUDIT_Z_SCORE",
            "PAPER_INSPECTION_SUSPECT": "PAPER_INSPECTION_SUSPECT",
            "AUDIT_ALERT": "AUDIT_ALERT",
            "ISOLATION_FLAG": "ISOLATION_FLAG",
            "AUDIT_PRIORITY": "AUDIT_PRIORITY",
            "REPORTED_LAST_MIN_COND": "CLAIMED_RATING",
        }
        available = [column for column in rename if column in audit.columns]
        audit_view = audit.loc[:, ["BRIDGE_ID", *available]].rename(columns=rename)
        queue = queue.merge(audit_view, on="BRIDGE_ID", how="left", validate="one_to_one")
    for column, default in (
        ("AUDIT_EXPECTED_SURVIVAL", np.nan),
        ("PHYSICS_EXPECTED_CONDITION", np.nan),
        ("CONDITION_GAP", np.nan),
        ("AUDIT_Z_SCORE", 0.0),
        ("PAPER_INSPECTION_SUSPECT", False),
        ("AUDIT_ALERT", False),
        ("ISOLATION_FLAG", False),
        ("AUDIT_PRIORITY", "NOT_AUDITED"),
        ("CLAIMED_RATING", np.nan),
    ):
        if column not in queue.columns:
            queue[column] = default

    locations = runtime.get("bridges")
    if locations is not None and not locations.empty:
        loc = locations.loc[:, ["BRIDGE_ID", "STATE", "DISTRICT"]].copy()
        loc["BRIDGE_ID"] = loc["BRIDGE_ID"].astype(str)
        queue = queue.merge(loc.drop_duplicates("BRIDGE_ID"), on="BRIDGE_ID", how="left", validate="one_to_one")
    ledger_view = ledger.copy()
    ledger_view["BRIDGE_ID"] = ledger_view["BRIDGE_ID"].astype(str)
    ledger_columns = [
        column
        for column in (
            "BRIDGE_ID",
            "MANAGING_AGENCY",
            "LAST_INSPECTION_DATE",
            "CURRENT_INSPECTOR_NAME",
            "CURRENT_INSPECTOR_ID",
            "CURRENT_INSPECTOR_PHONE",
            "PREVIOUS_INSPECTOR_NAME",
            "PREVIOUS_INSPECTOR_PHONE",
            "ATTACHMENT_PATH",
            "SUBMISSION_COUNT",
        )
        if column in ledger_view.columns
    ]
    queue = queue.merge(
        ledger_view.loc[:, ledger_columns].drop_duplicates("BRIDGE_ID"),
        on="BRIDGE_ID",
        how="left",
        validate="one_to_one",
    )

    a_suspect = queue["PAPER_INSPECTION_SUSPECT"].map(lambda value: _coerce_bool(value, False)).to_numpy(bool)
    a_alert = queue["AUDIT_ALERT"].map(lambda value: _coerce_bool(value, False)).to_numpy(bool)
    critical = (post_survival < SURVIVAL_THRESHOLD) | (post_rul <= 5.0)
    high = (~critical) & ((rul_loss >= 5.0) | a_suspect | (risk_increase >= 0.10))
    medium = (~critical) & (~high) & ((risk_increase > 0.02) | a_alert)
    queue["TRIAGE_PRIORITY"] = np.select(
        [critical, high, medium],
        ["CRITICAL", "HIGH", "MEDIUM"],
        default="LOW",
    )
    queue["IMMEDIATE_NDT_DEPLOYMENT"] = critical | high | a_suspect
    queue["EVENT_APPLIED"] = bool(active_events)

    if live_triage is not None and not live_triage.empty:
        live = live_triage.copy()
        live["BRIDGE_ID"] = live["BRIDGE_ID"].astype(str)
        live = live.sort_values("TRIAGE_PRIORITY_SCORE", ascending=False, kind="mergesort")
        live = live.drop_duplicates("BRIDGE_ID", keep="first")
        live_columns = {
            "POST_EVENT_SURVIVAL_AT_DECISION": "SURVIVAL_5Y",
            "POST_EVENT_RUL_YEARS": "POST_EVENT_RUL_YEARS",
            "RUL_LOSS_YEARS": "RUL_LOSS_YEARS",
            "TRIAGE_PRIORITY": "TRIAGE_PRIORITY",
            "IMMEDIATE_NDT_DEPLOYMENT": "IMMEDIATE_NDT_DEPLOYMENT",
            "TRIAGE_PRIORITY_SCORE": "TRIAGE_PRIORITY_SCORE",
            "LIVE_EVENT_ID": "LIVE_EVENT_ID",
        }
        available = {source: target for source, target in live_columns.items() if source in live.columns}
        live_view = live.loc[:, ["BRIDGE_ID", *available]].rename(columns=available)
        queue = queue.drop(
            columns=[column for column in live_view.columns if column != "BRIDGE_ID" and column in queue.columns],
            errors="ignore",
        )
        queue = queue.merge(live_view, on="BRIDGE_ID", how="left", validate="one_to_one", suffixes=("", "_LIVE"))
        queue["LIVE_TRIAGE_APPLIED"] = queue.get("LIVE_EVENT_ID", pd.Series(pd.NA, index=queue.index)).notna()
    else:
        queue["LIVE_EVENT_ID"] = ""
        queue["LIVE_TRIAGE_APPLIED"] = False

    queue = queue.sort_values(
        ["TRIAGE_PRIORITY_SCORE", "RUL_LOSS_YEARS", "BRIDGE_ID"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    queue = queue.drop(columns=["RANK"], errors="ignore")
    queue.insert(0, "RANK", np.arange(1, len(queue) + 1, dtype=np.int32))
    return queue


def curves_for_bridge(
    runtime: Mapping[str, Any],
    bridge_id: str,
    *,
    events: Sequence[Any] | None = None,
    scenario_event: Any | None = None,
) -> dict[str, Any] | None:
    """Return baseline and post-shock 0–60-year curves for one bridge."""

    from bridge_engine.triage_and_xai import conditional_survival_curve, compute_rul_from_survival

    portfolio = get_portfolio(runtime)
    if not portfolio:
        return None
    frame = portfolio["frame"]
    id_col = portfolio["id_col"]
    time_col = portfolio["time_col"]
    match = np.flatnonzero(frame[id_col].astype(str).to_numpy() == str(bridge_id))
    if match.size == 0:
        return None
    index = int(match[0])
    t0 = float(frame[time_col].iloc[index])
    baseline_curve = portfolio["curve"][index]
    active = list(events or [])
    bridges = runtime.get("bridges")
    applied_event = None
    if bridges is not None and active:
        row = bridges.loc[bridges["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
        if not row.empty:
            state = str(row.iloc[0]["STATE"])
            district = str(row.iloc[0]["DISTRICT"])
            for event in active:
                if (
                    (getattr(event, "district", None) is None or str(getattr(event, "district")) == district)
                    and (getattr(event, "state", None) is None or str(getattr(event, "state")) == state)
                ):
                    applied_event = event
                    break
    if applied_event is None and scenario_event is not None:
        applied_event = scenario_event

    if applied_event is None:
        post_curve = baseline_curve
        event_label = "Baseline only — no active regional shock"
    else:
        shocked = _apply_events_to_frame(
            frame.iloc[[index]].reset_index(drop=True),
            runtime,
            [applied_event],
            id_col=id_col,
        )
        post_curve = conditional_survival_curve(
            model_name=portfolio["model_name"],
            model=portfolio["model"],
            frame=shocked,
            feature_names=portfolio["features"],
            time_at_event=np.asarray([t0], dtype=np.float64),
            horizons=CURVE_GRID,
            survival_floor=SURVIVAL_FLOOR,
        )[0]
        event_label = str(getattr(applied_event, "event_id", "Scenario shock"))

    baseline_rul, baseline_censored = compute_rul_from_survival(
        baseline_curve[None, :], CURVE_GRID, threshold=SURVIVAL_THRESHOLD, censor_at=RUL_HORIZON
    )
    post_rul, post_censored = compute_rul_from_survival(
        post_curve[None, :], CURVE_GRID, threshold=SURVIVAL_THRESHOLD, censor_at=RUL_HORIZON
    )
    return {
        "bridge_id": str(bridge_id),
        "t0": t0,
        "baseline_curve": baseline_curve,
        "post_curve": post_curve,
        "baseline_survival_5y": float(baseline_curve[int(DECISION_HORIZON)]),
        "post_survival_5y": float(post_curve[int(DECISION_HORIZON)]),
        "baseline_rul": float(baseline_rul[0]),
        "post_rul": float(post_rul[0]),
        "rul_loss": float(max(baseline_rul[0] - post_rul[0], 0.0)),
        "baseline_censored": bool(baseline_censored[0]),
        "post_censored": bool(post_censored[0]),
        "event_label": event_label,
    }


# ---------------------------------------------------------------------------
# SHAP and attachments
# ---------------------------------------------------------------------------


def shap_contributions(
    runtime: Mapping[str, Any],
    bridge_id: str,
    *,
    events: Sequence[Any] | None = None,
    scenario_event: Any | None = None,
    use_persisted: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return top local risk contributions and a provenance note."""

    persisted = runtime.get("xai_contributions")
    if (
        use_persisted
        and not (events or scenario_event)
        and persisted is not None
        and not persisted.empty
        and "BRIDGE_ID" in persisted.columns
    ):
        subset = persisted.loc[persisted["BRIDGE_ID"].astype(str).eq(str(bridge_id))].copy()
        if not subset.empty:
            subset["FEATURE"] = subset["FEATURE"].astype(str).map(display_feature_name)
            return (
                subset.sort_values("ABS_SHAP_VALUE", ascending=False).head(12).reset_index(drop=True),
                "Persisted SHAP contributions from `artifacts/module4/xai_contributions.parquet`.",
            )

    portfolio = get_portfolio(runtime)
    if not portfolio:
        return pd.DataFrame(), "The survival portfolio is unavailable."
    frame = portfolio["frame"]
    id_col = portfolio["id_col"]
    match = np.flatnonzero(frame[id_col].astype(str).to_numpy() == str(bridge_id))
    if match.size == 0:
        return pd.DataFrame(), "No transformed feature row exists for this bridge."
    index = int(match[0])
    row_frame = frame.iloc[[index]].reset_index(drop=True)
    applied = None
    bridges = runtime.get("bridges")
    if bridges is not None and events:
        bridge_row = bridges.loc[bridges["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
        if not bridge_row.empty:
            state = str(bridge_row.iloc[0]["STATE"])
            district = str(bridge_row.iloc[0]["DISTRICT"])
            for event in events:
                if (
                    (getattr(event, "district", None) is None or str(getattr(event, "district")) == district)
                    and (getattr(event, "state", None) is None or str(getattr(event, "state")) == state)
                ):
                    applied = event
                    break
    if applied is None:
        applied = scenario_event
    if applied is not None:
        row_frame = _apply_events_to_frame(row_frame, runtime, [applied], id_col=id_col)

    model = portfolio["model"]
    if not hasattr(model, "hazard_ratios"):
        return pd.DataFrame(), "Local linear SHAP is only available for the Cox survival model."
    background = frame
    records: list[dict[str, Any]] = []
    for record in model.hazard_ratios():
        feature = str(getattr(record, "covariate", ""))
        if feature not in row_frame.columns or feature not in background.columns:
            continue
        value = _coerce_float(row_frame.iloc[0][feature])
        baseline = _coerce_float(pd.to_numeric(background[feature], errors="coerce").mean(), 0.0)
        contribution = float(getattr(record, "coef", 0.0)) * (value - baseline)
        records.append(
            {
                "FEATURE": display_feature_name(feature),
                "FEATURE_CODE": feature,
                "FEATURE_VALUE": value,
                "SHAP_VALUE": contribution,
                "ABS_SHAP_VALUE": abs(contribution),
                "DIRECTION": "INCREASES_RISK" if contribution >= 0.0 else "REDUCES_RISK",
            }
        )
    if not records:
        return pd.DataFrame(), "No hazard-ratio records were available for local attribution."
    contributions = pd.DataFrame.from_records(records)
    contributions = contributions.sort_values("ABS_SHAP_VALUE", ascending=False).head(12)
    return (
        contributions.reset_index(drop=True),
        "Exact local linear SHAP for the fitted Cox predictor, centred on the "
        "full-inventory mean and evaluated on the live post-shock feature row.",
    )


def _write_default_receipt(
    bridge_row: Mapping[str, Any],
    ledger_row: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> Path:
    bridge_id = str(bridge_row.get("BRIDGE_ID", ledger_row.get("BRIDGE_ID", "UNKNOWN")))
    directory = Path(root) / "artifacts" / "uploads" / safe_filename(bridge_id)
    directory.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    target = directory / f"AUTO_INSPECTION_RECEIPT_{safe_filename(bridge_id)}_{today}.pdf"
    if target.exists():
        return target
    lines = [
        "NATIONAL HIGHWAYS AUTHORITY OF INDIA / STATE PWD",
        "ZERO-TRUST BRIDGE INSPECTION RECEIPT",
        "",
        f"Bridge ID: {bridge_id}",
        f"State / District: {bridge_row.get('STATE', 'Unknown')} / {bridge_row.get('DISTRICT', 'Unknown')}",
        f"Managing Agency: {ledger_row.get('MANAGING_AGENCY', 'Unknown')}",
        f"Last Inspection Date: {ledger_row.get('LAST_INSPECTION_DATE', 'Not recorded')}",
        f"Reported Minimum Condition: {ledger_row.get('REPORTED_LAST_MIN_COND', 'Not recorded')} / 9",
        "",
        "CURRENT INSPECTOR",
        f"Name: {ledger_row.get('CURRENT_INSPECTOR_NAME', 'Not assigned')}",
        f"ID: {ledger_row.get('CURRENT_INSPECTOR_ID', 'Not assigned')}",
        f"Phone: {ledger_row.get('CURRENT_INSPECTOR_PHONE', 'Not assigned')}",
        f"Email: {ledger_row.get('CURRENT_INSPECTOR_EMAIL', 'Not assigned')}",
        "",
        "PREVIOUS INSPECTOR",
        f"Name: {ledger_row.get('PREVIOUS_INSPECTOR_NAME', 'Not recorded')}",
        f"ID: {ledger_row.get('PREVIOUS_INSPECTOR_ID', 'Not recorded')}",
        f"Phone: {ledger_row.get('PREVIOUS_INSPECTOR_PHONE', 'Not recorded')}",
        f"Email: {ledger_row.get('PREVIOUS_INSPECTOR_EMAIL', 'Not recorded')}",
        "",
        "INSPECTOR REMARKS",
        str(ledger_row.get("INSPECTOR_REMARKS", "No remarks recorded."))[:900],
        "",
        f"Receipt generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "This controlled copy is generated for central NHAI / State PWD verification.",
    ]
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        document = canvas.Canvas(str(target), pagesize=A4)
        width, height = A4
        y = height - 56
        for index, line in enumerate(lines):
            if y < 52:
                document.showPage()
                y = height - 56
            document.setFont("Helvetica-Bold" if index < 2 else "Helvetica", 12 if index < 2 else 9)
            document.drawString(52, y, line[:115])
            y -= 18 if line else 10
        document.save()
    except Exception:
        fallback = target.with_suffix(".txt")
        fallback.write_text("\n".join(lines), encoding="utf-8")
        return fallback
    return target


def attachment_for_bridge(
    runtime: Mapping[str, Any],
    ledger: pd.DataFrame,
    bridge_id: str,
    *,
    root: Path = ROOT,
) -> tuple[Path | None, bool]:
    """Return the uploaded attachment or a generated default receipt."""

    ledger_row = ledger.loc[ledger["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
    ledger_record: dict[str, Any] = ledger_row.iloc[0].to_dict() if not ledger_row.empty else {}
    attachment = str(ledger_record.get("ATTACHMENT_PATH", "") or "")
    if attachment:
        path = Path(attachment)
        if path.exists():
            return path, False
    bridges = runtime.get("bridges")
    if bridges is None or bridges.empty:
        return None, False
    bridge_row = bridges.loc[bridges["BRIDGE_ID"].astype(str).eq(str(bridge_id))]
    if bridge_row.empty:
        return None, False
    path = _write_default_receipt(bridge_row.iloc[0].to_dict(), ledger_record, root=root)
    return path, True


# ---------------------------------------------------------------------------
# Miscellaneous presentation helpers
# ---------------------------------------------------------------------------


def hazard_ratio_payload(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Extract +1 SD / +2 SD climate hazard ratios from the survival report."""

    report = runtime.get("survival_report") or {}
    acceleration = {}
    if isinstance(report, Mapping):
        for key in ("cox_time_varying", "monsoon_acceleration"):
            candidate = report.get(key)
            if isinstance(candidate, Mapping):
                nested = candidate.get("monsoon_acceleration", candidate)
                if isinstance(nested, Mapping) and nested:
                    acceleration = nested
                    break
    per_1sd = acceleration.get("hazard_ratio_per_1sd", {}) if isinstance(acceleration, Mapping) else {}
    per_2sd = acceleration.get("hazard_ratio_per_2sd", {}) if isinstance(acceleration, Mapping) else {}
    if not per_2sd and per_1sd:
        per_2sd = {
            str(key): float(value) ** 2
            for key, value in per_1sd.items()
            if isinstance(value, (int, float))
        }
    return {
        "acceleration": acceleration,
        "per_1sd": per_1sd,
        "per_2sd": per_2sd,
        "combined_1sd": _coerce_float(
            acceleration.get("combined_1sd_shock_hazard_ratio"), np.nan
        ) if isinstance(acceleration, Mapping) else np.nan,
        "combined_2sd": _coerce_float(
            acceleration.get("combined_extreme_shock_hazard_ratio"), np.nan
        ) if isinstance(acceleration, Mapping) else np.nan,
    }


def scenario_event_for_bridge(bridge_id: str, state: str, district: str) -> Any:
    """Return a deterministic illustrative flood-shock for deep dives."""

    from bridge_engine.triage_and_xai import DisasterEvent

    return DisasterEvent(
        event_id=f"SCENARIO-FLOOD-{_slug(district)}",
        event_type="FLOOD",
        district=str(district) if district else None,
        state=str(state) if state else None,
        severity=1.35,
        monsoon_anomaly_increment=3.0,
        rainfall_multiplier=1.9,
        discharge_multiplier=2.2,
        velocity_multiplier=1.6,
        scour_multiplier=2.4,
        flood_exceedance_increment=1,
        cumulative_scour_increment=15.0,
        event_year=AS_OF_YEAR,
    )


def event_to_dict(event: Any) -> dict[str, Any]:
    try:
        return asdict(event)
    except Exception:
        return {}


def events_summary(events: Sequence[Any]) -> str:
    return " • ".join(str(getattr(event, "event_id", "event")) for event in events)
