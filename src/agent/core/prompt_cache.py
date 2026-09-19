from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal


SectionKind = Literal[
    "system",
    "tools",
    "stable_context",
    "dynamic",
]

_SECTION_ORDER: dict[SectionKind, int] = {
    "system": 0,
    "tools": 1,
    "stable_context": 2,
    "dynamic": 3,
}


_CACHEABLE_KINDS: set[SectionKind] = {
    "system",
    "tools",
    "stable_context",
}


class PromptCachePlanError(ValueError):
    """Raised when prompt sections cannot form a safe cacheable prefix."""


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One named section of the prompt sent to an LLM."""

    kind: SectionKind
    content: str

    @property
    def is_cacheable(self) -> bool:
        return self.kind in _CACHEABLE_KINDS


@dataclass(frozen=True, slots=True)
class PromptCachePlan:
    sections : tuple[PromptSection, ...]

    def __post_init__(self) -> None:
        validate_cache_plan(self)

    @property
    def stable_sections(self) -> tuple[PromptSection, ...]:
        return tuple(
            section
            for section in self.sections
            if section.is_cacheable
        )

    @property
    def dynamic_sections(self) -> tuple[PromptSection, ...]:
        return tuple(
            section
            for section in self.sections
            if not section.is_cacheable
        )

    @property
    def stable_fingerprint(self) -> str:
        return stable_prefix_fingerprint(self)



def validate_cache_plan(plan : PromptCachePlan) -> None:
    if not plan.sections:
        raise PromptCachePlanError(
            "prompt cache plan must contain at least one section"
        )

    previous_order = -1
    seen_stable_kinds: set[SectionKind] = set()


    for section in plan.sections:
        if section.kind not in _SECTION_ORDER:
            raise PromptCachePlanError(
                f"unknown prompt section kind: {section.kind!r}"
            )

        if not isinstance(section.content, str):
            raise PromptCachePlanError(
                f"{section.kind} content must be a string"
            )

        current_order = _SECTION_ORDER[section.kind]

        if current_order < previous_order:
            raise PromptCachePlanError(
                "prompt sections must be ordered as: "
                "system -> tools -> stable_context -> dynamic"
            )

        if (
            section.is_cacheable
            and section.kind in seen_stable_kinds
        ):
            raise PromptCachePlanError(
                f"cacheable section appears more than once: {section.kind}"
            )

        if section.is_cacheable:
            seen_stable_kinds.add(section.kind)

        previous_order = current_order

    if not plan.stable_sections:
        raise PromptCachePlanError(
            "prompt cache plan needs at least one cacheable section"
        )


def stable_prefix_fingerprint(plan : PromptCachePlan) -> str:
    stable_payload = [
        {
            "kind" : section.kind,
            "content" : section.content
        }
        for section in plan.stable_sections
    ]

    encoded = json.dumps(
        stable_payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()