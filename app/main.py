from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Body

from pathlib import Path
import re
import unicodedata
from collections import Counter
import shutil
from typing import Any, Dict, Optional, List
from uuid import uuid4

import pandas as pd

from app.storage import STORE, SessionData, ensure_upload_dir
from app.storage import get_test_answers, set_test_answer
from app.storage import list_groups, upsert_group
from app.parsing.maptrack_csv import (
    parse_session,
    parse_session_df,
    list_task_ids,
    ParsedSession,
    get_user_id_column,
    infer_session_id_from_filename,
    validate_maptrack_df,
    build_spatial_trace_for_user,
)
from app.parsing.column_aliases import SOC_DEMO_COLUMN_ALIASES, resolve_column_aliases, resolve_single_column
from app.analysis.metrics import (
    compute_session_metrics,
    compute_all_task_metrics,
    SOC_DEMO_KEYS,
)


app = FastAPI(title="MapTrack Analytics (MVP)")

# --- Robust absolute paths (prevents slow/buggy relative FS issues) ---
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"

# Static UI
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


# =========================
# Helpers
# =========================

def _read_soc_demo_row(csv_path: Path) -> Dict[str, Any]:
    """
    Read just first row + relevant columns for soc-demo.
    Fast and robust even for big CSVs.
    """
    try:
        df0 = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return {}

    if df0.empty:
        return {}

    row = df0.iloc[0].to_dict()
    resolved = resolve_column_aliases(df0.columns, SOC_DEMO_COLUMN_ALIASES)
    out: Dict[str, Any] = {}
    for k in SOC_DEMO_KEYS:
        source_col = resolved.get(k)
        if source_col:
            out[k] = row.get(source_col)
    return out

def _normalize_user_id(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s else None


def _read_soc_demo_rows_by_user(df: pd.DataFrame, user_col: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    resolved = resolve_column_aliases(df.columns, SOC_DEMO_COLUMN_ALIASES)

    for _, row in df.iterrows():
        user_id = _normalize_user_id(row.get(user_col))
        if not user_id or user_id in out:
            continue

        row_dict = row.to_dict()
        soc: Dict[str, Any] = {}
        for k in SOC_DEMO_KEYS:
            source_col = resolved.get(k)
            if source_col:
                soc[k] = row_dict.get(source_col)
        out[user_id] = soc
    return out


def _sanitize_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return cleaned or "user"

def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Tries common delimiters (comma/tab/auto) to support slightly different CSV exports.
    """
    attempts = [
        {"kwargs": {"low_memory": False}},
        {"kwargs": {"sep": "	", "low_memory": False}},
        {"kwargs": {"sep": None, "engine": "python", "low_memory": False}},
    ]

    for attempt in attempts:
        try:
            df = pd.read_csv(path, **attempt["kwargs"])
        except Exception:
            continue
        if {"timestamp", "event_name"}.issubset(set(df.columns)):
            return df

    # last resort: return default read result (will fail later with clearer message if invalid)
    return pd.read_csv(path, low_memory=False)

def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_answers_by_task_from_df(df: pd.DataFrame) -> Dict[str, str]:
    if df.empty or "event_name" not in df.columns:
        return {}
    
    answer_event_names = {"answer selected", "polygon selected"}
    confirm_event_names = {"setting task", "completed"}

    if "task" not in df.columns:
        return {}

    rows = list(df.to_dict("records"))
    finalized: Dict[str, str] = {}
    pending: Optional[tuple[str, str]] = None

    for row in rows:
        event_name_raw = row.get("event_name")
        if event_name_raw is None or (isinstance(event_name_raw, float) and pd.isna(event_name_raw)):
            continue

        event_name = str(event_name_raw).strip().lower()
        if event_name in answer_event_names:
            task_raw = row.get("task")
            if task_raw is None or (isinstance(task_raw, float) and pd.isna(task_raw)):
                continue

            task_id = str(task_raw).strip()
            if not task_id:
                continue

            event_detail_raw = row.get("event_detail") if "event_detail" in df.columns else None
            answer_text = "" if event_detail_raw is None or (isinstance(event_detail_raw, float) and pd.isna(event_detail_raw)) else str(event_detail_raw).strip()
            if not answer_text:
                continue

            # keep latest candidate answer until task switch/completion confirms it
            pending = (task_id, answer_text)
            continue

        if event_name in confirm_event_names and pending is not None:
            task_id, answer_text = pending
            finalized[task_id] = answer_text
            pending = None

    return finalized


def _evaluate_answer(correct_answer: str, user_answer: str) -> bool:
    if not correct_answer or not user_answer:
        return False
    return _normalize_text(correct_answer) == _normalize_text(user_answer)


def _build_group_answers_payload(group: Dict[str, Any]) -> Dict[str, Any]:
    sessions = group.get("sessions", []) if isinstance(group.get("sessions"), list) else []
    test_id = str(group.get("test_id") or "TEST")
    answer_key = get_test_answers(test_id)

    by_task: Dict[str, Dict[str, Any]] = {}
    for session in sessions:
        user_id = str(session.get("user_id") or "").strip()
        stats = session.get("stats") if isinstance(session.get("stats"), dict) else {}
        answers_map = stats.get("answers_by_task") if isinstance(stats.get("answers_by_task"), dict) else {}

        for task_id, answer_text in answers_map.items():
            task = str(task_id).strip()
            answer = str(answer_text).strip()
            if not task or not answer:
                continue

            correct = answer_key.get(task)
            record = by_task.setdefault(task, {
                "task_id": task,
                "answers": [],
                "correct_answer": correct,
                "correct_count": 0,
                "total_count": 0,
            })
            is_correct = _evaluate_answer(correct or "", answer) if isinstance(correct, str) else False
            record["answers"].append({
                "user_id": user_id or None,
                "answer": answer,
                "is_correct": is_correct,
            })
            record["total_count"] += 1
            if is_correct:
                record["correct_count"] += 1

    for record in by_task.values():
        total = record.get("total_count") or 0
        correct_count = record.get("correct_count") or 0
        record["accuracy"] = (correct_count / total) if total else None

    return {
        "group_id": group.get("id"),
        "test_id": test_id,
        "tasks": by_task,
    }


def _tokenize_for_wordcloud(value: str) -> List[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    parts = [p for p in normalized.split() if len(p) >= 2]
    stopwords = {"a", "i", "v", "ve", "s", "z", "na", "do", "se", "ze", "to", "je", "jsou", "the", "and", "or", "of"}
    return [p for p in parts if p not in stopwords]


def _build_wordcloud_from_group_payload(payload: Dict[str, Any], task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    tasks = payload.get("tasks", {}) if isinstance(payload.get("tasks"), dict) else {}
    counter: Counter[str] = Counter()

    for t_id, record in tasks.items():
        if task_id and t_id != task_id:
            continue
        answers = record.get("answers", []) if isinstance(record.get("answers"), list) else []
        for item in answers:
            answer = str(item.get("answer") or "").strip()
            for token in _tokenize_for_wordcloud(answer):
                counter[token] += 1

    return [{"text": text, "count": count} for text, count in counter.most_common(80)]


# =========================
# API
# =========================

@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    test_id: str = Form("TEST"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Nahraj prosím CSV soubor.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOAD_DIR / file.filename

    # save file to disk
    try:
        with dst.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uložení souboru selhalo: {e}")

    # parse + metrics
    try:
        parsed_session = parse_session(str(dst), file.filename)

        # tasks list (stable order)
        tasks: List[str] = list_task_ids(parsed_session)
        primary_task: Optional[str] = tasks[0] if tasks else None

        # soc-demo from first row (optional columns)
        soc_row = _read_soc_demo_row(dst)

        # session metrics (+ soc-demo inside)
        session_metrics = compute_session_metrics(session=parsed_session, raw_row=soc_row)

        # task metrics (duration + event_count per task)
        task_metrics = compute_all_task_metrics(parsed_session)

        # store all stats in one structure
        source_df = _read_csv_flexible(dst)
        answers_by_task = _extract_answers_by_task_from_df(source_df)

        stats: Dict[str, Any] = {
            "session": session_metrics,
            "tasks": task_metrics,
            "answers_by_task": answers_by_task,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Zpracování CSV selhalo: {e}")

    # store metadata (MVP: in-memory)
    session_meta = SessionData(
        session_id=parsed_session.session_id,
        test_id=test_id or "TEST",
        file_path=str(dst),
        user_id=parsed_session.user_id,
        task=primary_task,   # legacy (keep for now)
        stats=stats,
    )
    STORE.upsert(session_meta)

    return {
        "session_id": parsed_session.session_id,
        "user_id": parsed_session.user_id,
        "test_id": test_id or "TEST",
        "task": primary_task,  # legacy
        "tasks": tasks,        # list of tasks for UI
        "stats": stats,        # { session:..., tasks:{...} }
    }

@app.post("/api/upload/bulk")
async def upload_bulk_csv(
    file: UploadFile = File(...),
    test_id: str = Form("TEST"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Nahraj prosím CSV soubor.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOAD_DIR / file.filename

    try:
        with dst.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uložení souboru selhalo: {e}")

    try:
        df = pd.read_csv(dst, low_memory=False)
        age_col = resolve_single_column(df.columns, "age", SOC_DEMO_COLUMN_ALIASES["age"])
        if age_col:
            df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
        validate_maptrack_df(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Zpracování CSV selhalo: {e}")

    user_col = get_user_id_column(df)
    if not user_col:
        raise HTTPException(status_code=400, detail="CSV neobsahuje povinný sloupec 'userid'.")

    df["_user_id_norm"] = df[user_col].apply(_normalize_user_id)
    df = df[df["_user_id_norm"].notna()]
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV neobsahuje žádné platné hodnoty ve sloupci 'userid'.")

    soc_rows = _read_soc_demo_rows_by_user(df, "_user_id_norm")
    base_session_id = infer_session_id_from_filename(file.filename)

    sessions_out: List[Dict[str, Any]] = []

    try:
        for user_id, df_user in df.groupby("_user_id_norm", sort=False):
            df_user = df_user.drop(columns=["_user_id_norm"])

            user_suffix = _sanitize_filename_component(str(user_id))
            user_filename = f"{dst.stem}__{user_suffix}.csv"
            user_path = UPLOAD_DIR / user_filename
            df_user.to_csv(user_path, index=False)

            session_id = f"{base_session_id}__{user_suffix}"
            parsed_session = parse_session_df(
                df_user,
                user_filename,
                user_id_override=str(user_id),
                session_id_override=session_id,
            )

            tasks: List[str] = list_task_ids(parsed_session)
            primary_task: Optional[str] = tasks[0] if tasks else None

            soc_row = soc_rows.get(str(user_id), {})
            session_metrics = compute_session_metrics(session=parsed_session, raw_row=soc_row)
            task_metrics = compute_all_task_metrics(parsed_session)
            answers_by_task = _extract_answers_by_task_from_df(df_user)

            stats: Dict[str, Any] = {
                "session": session_metrics,
                "tasks": task_metrics,
                "answers": answers_by_task,
            }

            session_meta = SessionData(
                session_id=parsed_session.session_id,
                test_id=test_id or "TEST",
                file_path=str(user_path),
                user_id=parsed_session.user_id,
                task=primary_task,
                stats=stats,
            )
            STORE.upsert(session_meta)

            sessions_out.append({
                "session_id": parsed_session.session_id,
                "test_id": test_id or "TEST",
                "user_id": parsed_session.user_id,
                "task": primary_task,
                "tasks": tasks,
                "stats": stats,
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Zpracování hromadného CSV selhalo: {e}")

    return {
        "count": len(sessions_out),
        "sessions": sessions_out,
    }


@app.get("/api/sessions")
def list_sessions():
    sessions = STORE.list_sessions()

    out = []
    for s in sessions.values():
        stats = s.stats if isinstance(s.stats, dict) else {}
        session_stats = stats.get("session", {}) if isinstance(stats.get("session"), dict) else {}

        task_metrics = stats.get("tasks", {}) if isinstance(stats.get("tasks"), dict) else {}
        tasks = list(task_metrics.keys())

        out.append({
            "session_id": s.session_id,
            "test_id": getattr(s, "test_id", "TEST") or "TEST",
            "user_id": s.user_id,
            "task": s.task,      # legacy
            "tasks": tasks,      # new
            "stats": stats,
            "session_stats": session_stats,
        })

    return {"sessions": out}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = STORE.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session nenalezena.")

    stats = s.stats if isinstance(s.stats, dict) else {}
    task_metrics = stats.get("tasks", {}) if isinstance(stats.get("tasks"), dict) else {}
    tasks = list(task_metrics.keys())

    return {
        "session_id": s.session_id,
        "test_id": getattr(s, "test_id", "TEST") or "TEST",
        "user_id": s.user_id,
        "task": s.task,      # legacy
        "tasks": tasks,      # new
        "stats": stats,
        "file_path": s.file_path,
    }


@app.get("/api/sessions/{session_id}/tasks/{task_id}/metrics")
def get_task_metrics(session_id: str, task_id: str):
    s = STORE.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session nenalezena.")

    stats = s.stats if isinstance(s.stats, dict) else {}
    task_metrics = stats.get("tasks", {}) if isinstance(stats.get("tasks"), dict) else {}

    m = task_metrics.get(task_id)
    if not isinstance(m, dict):
        raise HTTPException(status_code=404, detail="Task nenalezen v session.")

    return m


# ===== NEW: raw events for timeline =====
@app.get("/api/sessions/{session_id}/events")
def get_session_events(session_id: str):
    s = STORE.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session nenalezena.")

    csv_path = Path(s.file_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV soubor pro session nenalezen.")

    # read only required columns
    usecols = ["timestamp", "event_name", "event_detail", "task"]
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nelze načíst eventy z CSV: {e}")

    # normalize + drop rows without essentials
    if "timestamp" not in df.columns or "event_name" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV neobsahuje required sloupce (timestamp, event_name).")

    # keep order as in file
    out = []
    for _, row in df.iterrows():
        ts = row.get("timestamp")
        name = row.get("event_name")
        if pd.isna(ts) or pd.isna(name):
            continue
        detail = row.get("event_detail") if "event_detail" in df.columns else None
        task = row.get("task") if "task" in df.columns else None

        out.append({
            "timestamp": int(ts),
            "event_name": str(name),
            "event_detail": None if pd.isna(detail) else str(detail),
            "task": None if pd.isna(task) else str(task),
        })

    return {
        "session_id": s.session_id,
        "user_id": s.user_id,
        "events": out,
    }


@app.get("/api/sessions/{session_id}/spatial-trace")
def get_session_spatial_trace(session_id: str, task_id: Optional[str] = None):
    s = STORE.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session nenalezena.")

    csv_path = Path(s.file_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV soubor pro session nenalezen.")

    usecols = ["timestamp", "event_name", "event_detail", "task", "userId", "userid", "user_id"]
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in usecols)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nelze načíst spatial data z CSV: {e}")

    user_id = s.user_id
    user_col = get_user_id_column(df)
    if not user_id and user_col and not df.empty:
        first_uid = df.iloc[0].get(user_col)
        if first_uid is not None and not (isinstance(first_uid, float) and pd.isna(first_uid)):
            user_id = str(first_uid).strip()

    try:
        trace = build_spatial_trace_for_user(df, user_id=user_id, task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "session_id": s.session_id,
        "user_id": s.user_id,
        "spatial": trace,
    }


@app.get("/api/tests/{test_id}/answers")
def api_get_test_answers(test_id: str):
    return {"test_id": test_id, "answers": get_test_answers(test_id)}


@app.put("/api/tests/{test_id}/answers/{task_id}")
def api_put_test_answer(
    test_id: str,
    task_id: str,
    payload: dict = Body(...),
):
    if "answer" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'answer' in body.")

    answer = payload["answer"]
    if answer is None:
        updated = set_test_answer(test_id, task_id, None)
        return {"test_id": test_id, "answers": updated}

    if not isinstance(answer, str):
        raise HTTPException(status_code=400, detail="'answer' must be a string or null.")

    updated = set_test_answer(test_id, task_id, answer)
    return {"test_id": test_id, "answers": updated}


@app.get("/api/groups")
def api_list_groups(test_id: Optional[str] = None):
    groups = list_groups(test_id=test_id)

    out = []
    for g in groups:
        session_ids = g.get("session_ids", []) if isinstance(g.get("session_ids"), list) else []
        sessions_out = []

        for sid in session_ids:
            session = STORE.get(sid)
            if not session:
                continue
            sessions_out.append({
                "session_id": session.session_id,
                "user_id": session.user_id,
                "test_id": getattr(session, "test_id", "TEST") or "TEST",
                "task": session.task,
                "tasks": list((session.stats or {}).get("tasks", {}).keys()),
                "stats": session.stats if isinstance(session.stats, dict) else {},
            })

        out.append({
            "id": g.get("id"),
            "test_id": g.get("test_id"),
            "name": g.get("name"),
            "session_ids": session_ids,
            "sessions": sessions_out,
        })

    return {"groups": out}

@app.get("/api/groups/{group_id}/answers")
def api_group_answers(group_id: str):
    group = next((g for g in list_groups() if g.get("id") == group_id), None)
    if not group:
        raise HTTPException(status_code=404, detail="Skupina nenalezena.")

    session_ids = group.get("session_ids", []) if isinstance(group.get("session_ids"), list) else []
    sessions_out = []
    for sid in session_ids:
        session = STORE.get(sid)
        if not session:
            continue

        stats = session.stats if isinstance(session.stats, dict) else {}
        answers_by_task = stats.get("answers_by_task") if isinstance(stats.get("answers_by_task"), dict) else {}

        if not answers_by_task:
            # Backfill for sessions uploaded before answers extraction existed.
            csv_path = Path(session.file_path)
            if csv_path.exists():
                try:
                    df_session = _read_csv_flexible(csv_path)
                    answers_by_task = _extract_answers_by_task_from_df(df_session)
                except Exception:
                    answers_by_task = {}
            stats = {**stats, "answers_by_task": answers_by_task}
            session.stats = stats

        sessions_out.append({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "test_id": getattr(session, "test_id", "TEST") or "TEST",
            "task": session.task,
            "tasks": list((stats or {}).get("tasks", {}).keys()),
            "stats": stats,
        })

    payload = _build_group_answers_payload({**group, "sessions": sessions_out})
    return payload


@app.get("/api/groups/{group_id}/wordcloud")
def api_group_wordcloud(group_id: str, task_id: Optional[str] = None):
    answers_payload = api_group_answers(group_id)
    words = _build_wordcloud_from_group_payload(answers_payload, task_id=task_id)
    return {
        "group_id": group_id,
        "task_id": task_id,
        "words": words,
    }


@app.post("/api/groups/compare/wordcloud")
def api_compare_wordcloud(payload: dict = Body(...)):
    group_ids = payload.get("group_ids", [])
    task_id = payload.get("task_id")
    if not isinstance(group_ids, list) or not group_ids:
        raise HTTPException(status_code=400, detail="group_ids must be non-empty list")

    out_groups = []
    for gid in group_ids:
        gid_str = str(gid).strip()
        if not gid_str:
            continue
        try:
            answers_payload = api_group_answers(gid_str)
        except HTTPException:
            continue
        words = _build_wordcloud_from_group_payload(answers_payload, task_id=str(task_id).strip() if isinstance(task_id, str) and task_id.strip() else None)
        out_groups.append({
            "group_id": gid_str,
            "words": words,
        })

    return {
        "task_id": task_id if isinstance(task_id, str) and task_id.strip() else None,
        "groups": out_groups,
    }


@app.post("/api/groups")
def api_create_group(payload: dict = Body(...)):
    name = str(payload.get("name", "")).strip()
    test_id = str(payload.get("test_id", "TEST")).strip() or "TEST"
    session_ids = payload.get("session_ids", [])

    if not name:
        raise HTTPException(status_code=400, detail="Název skupiny je povinný.")
    if not isinstance(session_ids, list) or not session_ids:
        raise HTTPException(status_code=400, detail="Vyber alespoň jednu session.")

    group_id = f"grp_{uuid4().hex[:12]}"
    try:
        group = upsert_group(group_id=group_id, test_id=test_id, name=name, session_ids=session_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"group": group}


@app.put("/api/groups/{group_id}")
def api_update_group(group_id: str, payload: dict = Body(...)):
    existing = next((g for g in list_groups() if g.get("id") == group_id), None)
    if not existing:
        raise HTTPException(status_code=404, detail="Skupina nenalezena.")

    name = str(payload.get("name", existing.get("name", ""))).strip()
    test_id = str(payload.get("test_id", existing.get("test_id", "TEST"))).strip() or "TEST"
    session_ids = payload.get("session_ids", existing.get("session_ids", []))
    if not isinstance(session_ids, list):
        raise HTTPException(status_code=400, detail="session_ids musí být pole.")

    try:
        group = upsert_group(group_id=group_id, test_id=test_id, name=name, session_ids=session_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"group": group}