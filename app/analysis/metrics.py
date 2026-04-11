
"""
Session and task metrics derived from the uploaded CSVs.
The module keeps metric computation lightweight so values can be reused in API responses and storage.
It also centralizes socio-demographic extraction and normalization rules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from app.parsing.maptrack_csv import ParsedSession, TaskStream
from app.normalization.nationality import normalize_nationality

SOC_DEMO_KEYS = [
    "age",
    "gender",
    "occupation",
    "education",
    "nationality",
    "device",
    "confidence",
    "paper_maps",
    "computer_maps",
    "mobile_maps",
    "screenResolution",
    "viewportSize",
    "browser",
    "browserVersion",
    "os",
    "ip",
]

def _first_nonempty(raw_row: Optional[Dict[str, Any]], key: str) -> Optional[str]:
    """Return a trimmed string value if present, otherwise None."""
    if not raw_row:
        return None
    v = raw_row.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

def extract_soc_demo(
    *,
    session: ParsedSession,
    raw_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Resolve socio-demographics from parsed session data or fallback CSV row."""
    out: Dict[str, Optional[str]] = {}

    # parser-level fields are preferred so future parser improvements are picked up automatically
    soc_from_session = getattr(session, "soc_demo", None)
    if isinstance(soc_from_session, dict):
        for k in SOC_DEMO_KEYS:
            v = soc_from_session.get(k)
            if k == "nationality":
                out[k] = normalize_nationality(v)
                continue
            if v is None:
                out[k] = None
            else:
                s = str(v).strip()
                out[k] = s if s else None
        return out

    for k in SOC_DEMO_KEYS:
        if k == "nationality":
            out[k] = normalize_nationality(_first_nonempty(raw_row, k))
        else:
            out[k] = _first_nonempty(raw_row, k)

    return out

def compute_session_metrics(
    *,
    session: ParsedSession,
    raw_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute per-session metrics used by API responses and persistence."""
    events = session.events
    event_count = len(events)

    if event_count:
        ts = [e.timestamp_ms for e in events if isinstance(e.timestamp_ms, int)]
        if ts:
            time_min = min(ts)
            time_max = max(ts)
            duration_ms = time_max - time_min
        else:
            time_min = None
            time_max = None
            duration_ms = None
    else:
        time_min = None
        time_max = None
        duration_ms = None

    task_ids = list(session.tasks.keys())
    tasks_count = len(task_ids)

    soc_demo = extract_soc_demo(session=session, raw_row=raw_row)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "tasks_count": tasks_count,
        "events_total": event_count,
        "time_min_ms": time_min,
        "time_max_ms": time_max,
        "duration_ms": duration_ms,
        "soc_demo": soc_demo,
    }

def compute_task_metrics(task: TaskStream) -> Dict[str, Any]:
    """Compute duration and event counts for a single task stream."""
    events = task.events
    event_count = len(events)

    if event_count:
        ts = [e.timestamp_ms for e in events if isinstance(e.timestamp_ms, int)]
        if ts:
            tmin = min(ts)
            tmax = max(ts)
            duration_ms = tmax - tmin
        else:
            tmin = None
            tmax = None
            duration_ms = None
    else:
        tmin = None
        tmax = None
        duration_ms = None

    return {
        "task_id": task.task_id,
        "events_total": event_count,
        "time_min_ms": tmin,
        "time_max_ms": tmax,
        "duration_ms": duration_ms,
    }

def compute_all_task_metrics(session: ParsedSession) -> Dict[str, Dict[str, Any]]:
    """Return metrics keyed by task_id."""
    out: Dict[str, Dict[str, Any]] = {}
    for task_id, task_stream in session.tasks.items():
        out[task_id] = compute_task_metrics(task_stream)
    return out

def aggregate_sessions(
    sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate session metric payloads for experiment-level summaries."""
    if not sessions:
        return {
            "sessions_count": 0,
            "avg_duration_ms": None,
            "avg_events_total": None,
        }

    durations = [s.get("duration_ms") for s in sessions if isinstance(s.get("duration_ms"), int)]
    events = [s.get("events_total") for s in sessions if isinstance(s.get("events_total"), int)]

    avg_duration = int(sum(durations) / len(durations)) if durations else None
    avg_events = int(sum(events) / len(events)) if events else None

    return {
        "sessions_count": len(sessions),
        "avg_duration_ms": avg_duration,
        "avg_events_total": avg_events,
    }