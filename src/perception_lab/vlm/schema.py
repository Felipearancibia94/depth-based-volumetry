"""Structured VLM answer: JSON schema (for constrained decoding) and strict validation."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from perception_lab.core.config import UNKNOWN_PET, PetConfig
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
    pet: str | None = None  # a configured pet name or "unknown"; None = not asked

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.pet is None:
            del data["pet"]
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


def build_json_schema(pets: Sequence[PetConfig] = ()) -> dict[str, Any]:
    """The answer schema; with household pets it also requires a short ``pet`` value."""
    if not pets:
        return JSON_SCHEMA
    schema = {**JSON_SCHEMA, "properties": dict(JSON_SCHEMA["properties"])}
    schema["properties"]["pet"] = {"type": "string", "enum": [p.name for p in pets] + [UNKNOWN_PET]}
    schema["required"] = [*JSON_SCHEMA["required"], "pet"]
    return schema


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_interpretation(text: str, pets: Sequence[PetConfig] = ()) -> Interpretation:
    """Parse and validate a model answer. Extra keys and wrong types are rejected."""
    schema = build_json_schema(pets)
    match = _FENCE.match(text)
    body = match.group(1) if match else text
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise InterpretationError(f"answer is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise InterpretationError("answer must be a JSON object")
    missing = [k for k in schema["required"] if k not in data]
    if missing:
        raise InterpretationError(f"missing field(s): {', '.join(missing)}")
    extra = sorted(set(data) - set(schema["properties"]))
    if extra:
        raise InterpretationError(f"unexpected field(s): {', '.join(extra)}")
    if data["activity"] not in ACTIVITIES:
        raise InterpretationError(f"activity {data['activity']!r} is not in the allowed list")
    if data["certainty"] not in CERTAINTY:
        raise InterpretationError(f"certainty {data['certainty']!r} is not low/medium/high")
    pet = None
    if pets:
        pet = data["pet"]
        if pet not in schema["properties"]["pet"]["enum"]:
            raise InterpretationError(f"pet {pet!r} is not a configured pet or {UNKNOWN_PET!r}")
    description = _text(data["description"], "description")
    observations = _items(data["visual_observations"], "visual_observations")
    uncertainty = _items(data["uncertainty"], "uncertainty")
    return Interpretation(
        data["activity"], description, observations, uncertainty, data["certainty"], pet
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
