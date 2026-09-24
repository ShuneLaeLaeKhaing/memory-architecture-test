"""Jev calls for classification, prioritization, and confidence. Not for grounding."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from loguru import logger


def jev_ready() -> bool:
    return bool(os.getenv("TYPESAFE_API_KEY", "").strip())


def ask(state: str, questions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        return None
    base = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    model = os.getenv("TYPESAFE_MODEL", "jev-latest")
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/systemone",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning(f"[jev] request failed: {exc}")
        return None
    answers = payload.get("answers") if isinstance(payload, dict) else None
    return answers if isinstance(answers, dict) else None


def choice(state: str, name: str, instructions: str, criteria: Dict[str, str]) -> Optional[str]:
    answers = ask(state, {
        name: {"type": "choice", "instructions": instructions, "criteria": criteria},
    })
    if not answers:
        return None
    value = _scalar(answers.get(name))
    if value in criteria:
        logger.info(f"[jev] choice {name}={value}")
        return value
    logger.warning(f"[jev] choice {name} rejected: {value}")
    return None


def noul(
    state: str,
    name: str,
    instructions: str,
    *,
    yes: str,
    no: str,
    threshold: float = 0.5,
) -> Optional[bool]:
    """Noul returns one probability that the answer is yes, from 0 to 1."""
    answers = ask(state, {
        name: {
            "type": "noul",
            "instructions": instructions,
            "criteria": {"true": yes, "false": no},
        },
    })
    if not answers:
        return None
    score = _noul_score(answers.get(name))
    if score is None:
        logger.warning(f"[jev] noul {name} rejected: {answers.get(name)}")
        return None
    decision = score >= threshold
    logger.info(f"[jev] noul {name}={score:.2f} yes={decision}")
    return decision


def _scalar(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("value", "answer", "choice", "label"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    return None


def _noul_score(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, dict)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = value.get("noul")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    return None
