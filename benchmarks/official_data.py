"""
benchmarks/official_data.py — paths + download helpers for official corpora.

Datasets
--------
LoCoMo       : snap-research/locomo  → benchmarks/data/locomo10.json
LongMemEval  : xiaowu0162/longmemeval-cleaned → longmemeval_{s,m}_cleaned.json
ConvoMem     : Salesforce/ConvoMem → benchmarks/data/convomem/...
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

LOCOMO_PATH = DATA_DIR / "locomo10.json"
LONGMEM_S = DATA_DIR / "longmemeval_s_cleaned.json"
LONGMEM_M = DATA_DIR / "longmemeval_m_cleaned.json"
CONVOMEM_DIR = DATA_DIR / "convomem"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _hf_download(repo: str, *patterns: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(local_dir),
    ]
    for pat in patterns:
        cmd.extend(["--include", pat])
    logger.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd)


def download_locomo() -> Path:
    ensure_data_dir()
    if LOCOMO_PATH.exists():
        return LOCOMO_PATH
    # Prefer GitHub raw (small JSON)
    import urllib.request

    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    logger.info("Downloading LoCoMo from %s", url)
    urllib.request.urlretrieve(url, LOCOMO_PATH)
    return LOCOMO_PATH


def download_longmemeval(variant: str = "s") -> Path:
    ensure_data_dir()
    target = LONGMEM_S if variant == "s" else LONGMEM_M
    filename = target.name
    if target.exists():
        return target
    _hf_download(
        "xiaowu0162/longmemeval-cleaned",
        filename,
        local_dir=DATA_DIR,
    )
    if not target.exists():
        raise FileNotFoundError(f"Expected {target} after download")
    return target


def download_convomem_sample() -> Path:
    """Download a manageable ConvoMem slice (user + changing evidence)."""
    ensure_data_dir()
    marker = CONVOMEM_DIR / "core_benchmark" / "evidence_questions"
    if marker.exists() and any(marker.rglob("*.json")):
        return CONVOMEM_DIR
    _hf_download(
        "Salesforce/ConvoMem",
        "core_benchmark/evidence_questions/user_evidence/1_evidence/*.json",
        "core_benchmark/evidence_questions/changing_evidence/2_evidence/*.json",
        "core_benchmark/evidence_questions/preference_evidence/1_evidence/*.json",
        local_dir=CONVOMEM_DIR,
    )
    return CONVOMEM_DIR


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_locomo_cases(path: Path = LOCOMO_PATH, *, seed: Optional[int] = None):
    """Yield (sample_id, turns, qa_items) for each conversation."""
    data = list(load_json(path))
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(data)
    for sample in data:
        conv = sample["conversation"]
        session_keys = sorted(
            (
                k
                for k in conv
                if k.startswith("session_")
                and not k.endswith("date_time")
                and "observation" not in k
                and "summary" not in k
            ),
            key=lambda k: int(k.split("_")[1]),
        )
        turns: List[dict] = []
        for sk in session_keys:
            for turn in conv[sk]:
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                speaker = turn.get("speaker") or "Speaker"
                turns.append(
                    {
                        "dia_id": turn.get("dia_id"),
                        "text": f"{speaker}: {text}",
                        "raw": text,
                    }
                )
        dia_map = {
            t["dia_id"]: (t.get("raw") or t["text"])
            for t in turns
            if t.get("dia_id")
        }
        qa = []
        for q in sample.get("qa") or []:
            if q.get("answer") is None or str(q.get("answer")).strip() == "":
                continue
            evid_ids = q.get("evidence") or []
            evidence_texts = [
                dia_map[eid] for eid in evid_ids if isinstance(eid, str) and eid in dia_map
            ]
            enriched = dict(q)
            enriched["evidence_texts"] = evidence_texts
            qa.append(enriched)
        if rng is not None:
            rng.shuffle(qa)
        yield sample.get("sample_id", "unknown"), turns, qa


def iter_longmemeval_cases(
    path: Path,
    limit: Optional[int] = None,
    *,
    seed: Optional[int] = None,
):
    data = list(load_json(path))
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(data)
    for i, item in enumerate(data):
        if limit is not None and i >= limit:
            break
        sessions = item.get("haystack_sessions") or []
        turns: List[str] = []
        evidence_texts: List[str] = []
        for session in sessions:
            for turn in session:
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                role = turn.get("role") or "user"
                line = f"{role}: {content}"
                turns.append(line)
                if turn.get("has_answer"):
                    evidence_texts.append(content)
        yield {
            "question_id": item.get("question_id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "question_type": item.get("question_type"),
            "turns": turns,
            "evidence_texts": evidence_texts,
            "n_sessions": len(sessions),
        }


def _flatten_convomem_messages(obj: Any) -> List[str]:
    """Best-effort extraction of chat messages from ConvoMem JSON variants."""
    messages: List[str] = []

    def walk(node: Any):
        if isinstance(node, dict):
            if "role" in node and "content" in node:
                content = str(node.get("content") or "").strip()
                if content:
                    messages.append(f"{node.get('role')}: {content}")
                return
            if "messages" in node and isinstance(node["messages"], list):
                for m in node["messages"]:
                    walk(m)
                return
            if "conversation" in node:
                walk(node["conversation"])
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(obj)
    return messages


def _convomem_turns_from_item(item: dict) -> List[str]:
    turns: List[str] = []
    for conv in item.get("conversations") or []:
        for msg in conv.get("messages") or []:
            text = (msg.get("text") or msg.get("content") or "").strip()
            if not text:
                continue
            speaker = msg.get("speaker") or msg.get("role") or "User"
            turns.append(f"{speaker}: {text}")
    if turns:
        return turns
    # Fallback: evidence messages only
    for msg in item.get("message_evidences") or []:
        text = (msg.get("text") or "").strip()
        if text:
            turns.append(f"{msg.get('speaker') or 'User'}: {text}")
    return turns


def iter_convomem_cases(
    root: Path = CONVOMEM_DIR,
    limit: Optional[int] = None,
    *,
    seed: Optional[int] = None,
) -> Iterable[dict]:
    files = sorted(
        f
        for f in root.rglob("*.json")
        if "evidence_questions" in str(f) and ".cache" not in str(f)
    )
    cases: List[dict] = []
    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip %s: %s", path, exc)
            continue
        items = data.get("evidence_items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            items = [data] if isinstance(data, dict) else []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            question = item.get("question")
            answer = item.get("answer")
            if not question or answer is None:
                continue
            turns = _convomem_turns_from_item(item)
            if not turns:
                continue
            evidence_texts: List[str] = []
            for msg in item.get("message_evidences") or []:
                text = (msg.get("text") or msg.get("content") or "").strip()
                if text:
                    evidence_texts.append(text)
            cases.append(
                {
                    "id": item.get("id")
                    or item.get("question_id")
                    or f"{path.name}:{idx}",
                    "question": str(question),
                    "answer": str(answer),
                    "turns": turns,
                    "evidence_texts": evidence_texts,
                    "source": str(path.relative_to(root)),
                }
            )
    rng = random.Random(seed) if seed is not None else None
    if rng is not None:
        rng.shuffle(cases)
    if limit is not None:
        cases = cases[:limit]
    yield from cases
