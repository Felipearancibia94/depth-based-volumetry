"""Structured VLM answer: JSON schema (for constrained decoding) and strict validation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from perception_lab.events.model import ACTIVITIES

CERTAINTY = ("low", "medium", "high")
MAX_TEXT_CHARS = 600
MAX_ITEMS = 10


class InterpretationError(ValueError):
    """Raised when a model answer does not match the schema."""


@dataclass(frozen=True)
class Interpretation:
    activity: str
    description: str
    visual_observations: tuple[str, ...]
    uncertainty: tuple[str, ...]
    certainty: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["visual_observations"] = list(self.visual_observations)
        data["uncertainty"] = list(self.uncertainty)
        return data


JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "activity": {"type": "string", "enum": list(ACTIVITIES)},
        "description": {"type": "string"},
        "visual_observations": {"type": "array", "items": {"type": "string"}},
        "uncertainty": {"type": "array", "items": {"type": "string"}},
        "certainty": {"type": "string", "enum": list(CERTAINTY)},
    },
    "required": ["activity", "description", "visual_observations", "uncertainty", "certainty"],
    "additionalProperties": False,
}

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_interpretation(text: str) -> Interpretation:
    """Parse and validate a model answer. Extra keys and wrong types are rejected."""
    match = _FENCE.match(text)
    body = match.group(1) if match else text
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InterpretationError(f"answer is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise InterpretationError("answer must be a JSON object")
    missing = [k for k in JSON_SCHEMA["required"] if k not in data]
    if missing:
        raise InterpretationError(f"missing field(s): {', '.join(missing)}")
    extra = sorted(set(data) - set(JSON_SCHEMA["properties"]))
    if extra:
        raise InterpretationError(f"unexpected field(s): {', '.join(extra)}")
    if data["activity"] not in ACTIVITIES:
        raise InterpretationError(f"activity {data['activity']!r} is not in the allowed list")
    if data["certainty"] not in CERTAINTY:
        raise InterpretationError(f"certainty {data['certainty']!r} is not low/medium/high")
    description = _text(data["description"], "description")
    observations = _items(data["visual_observations"], "visual_observations")
    uncertainty = _items(data["uncertainty"], "uncertainty")
    return Interpretation(
        data["activity"], description, observations, uncertainty, data["certainty"]
    )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise InterpretationError(f"{name} must be a string")
    if len(value) > MAX_TEXT_CHARS:
        raise InterpretationError(f"{name} is longer than {MAX_TEXT_CHARS} characters")
    return value.strip()


def _items(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InterpretationError(f"{name} must be a list of strings")
    if len(value) > MAX_ITEMS:
        raise InterpretationError(f"{name} has more than {MAX_ITEMS} items")
    return tuple(_text(v, name) for v in value)
