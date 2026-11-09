"""Versioned prompts. Change the text only together with ``PROMPT_VERSION``."""

from __future__ import annotations

from collections.abc import Sequence

from perception_lab.core.config import PetConfig
from perception_lab.events.model import ACTIVITIES, PetEvent

PROMPT_VERSION = "pets_v1"
PROMPT_VERSION_WITH_PETS = "pets_v1+pet"  # the text differs when household pets are listed


def prompt_version(pets: Sequence[PetConfig] = ()) -> str:
    return PROMPT_VERSION_WITH_PETS if pets else PROMPT_VERSION


_TEMPLATE = """You are looking at {count} from a home camera, in time order ({times}).
{layout}They show {animals}.

Measured by sensors (treat as reliable):
{facts}

The detection rule that selected these frames suggests "{hypothesis}". This is only a
hypothesis and may be wrong.
{household}
Describe only what is visible in the images. Choose "activity" from: {activities}.
If the images do not show enough to decide, answer "unknown". Do not guess.

Answer with a JSON object with exactly these fields:
- "activity": one of the allowed values.
- "description": one or two sentences about what is visible.
- "visual_observations": short facts you can directly see in the images.
- "uncertainty": what cannot be determined from the images.
- "certainty": "low", "medium" or "high".{pet_field}
"""

_HOUSEHOLD = """
The household has these pets:
{pets}
Judge only by visible features such as species and coat colour.
"""

_PET_FIELD = """
- "pet": the name of the listed pet that is clearly shown, or "unknown" if none is clearly shown
  or you cannot tell."""

RETRY_SUFFIX = (
    "\nYour previous answer was not valid ({error}). Reply with the JSON object only, "
    "using exactly the fields and allowed values listed above."
)


def build_prompt(
    event: PetEvent,
    frame_times_s: Sequence[float],
    *,
    montage: bool,
    pets: Sequence[PetConfig] = (),
) -> str:
    t0 = frame_times_s[0] if frame_times_s else event.timestamp_start_s
    times = ", ".join(f"t+{t - t0:.1f}s" for t in frame_times_s) or "single frame"
    count = "1 image" if montage or len(frame_times_s) == 1 else f"{len(frame_times_s)} images"
    layout = (
        "The frames are arranged in one image as a grid, left to right, top to bottom. "
        if montage
        else ""
    )
    animals = " and ".join(
        f"a {label} (track {tid})"
        for label, tid in zip(event.animal_classes, event.track_ids, strict=False)
    )
    facts = "\n".join(f"- {key}: {value}" for key, value in event.observed_facts.items())
    household = (
        _HOUSEHOLD.format(
            pets="\n".join(f"- {p.name} ({p.species}): {p.description}" for p in pets)
        )
        if pets
        else ""
    )
    return _TEMPLATE.format(
        household=household,
        pet_field=_PET_FIELD if pets else "",
        count=count,
        times=times,
        layout=layout,
        animals=animals or "an animal",
        facts=facts or "- none",
        hypothesis=event.hypothesis,
        activities=", ".join(ACTIVITIES),
    )
