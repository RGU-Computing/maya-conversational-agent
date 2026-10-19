"""MAYA's auxiliary disambiguation policy, separate from the partner ontology."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from maya.framework.ontology.loader import load_ontology


def load_disambiguation_policy() -> dict[str, Any]:
    return load_ontology(Path(__file__).with_name("disambiguation_ontology.json"))


def render_disambiguation_block(
    ontology: dict[str, Any],
    *,
    limit: int | None = 4,
    rules_per_entity: int | None = 2,
) -> str:
    entities = ontology.get("entities") or []
    lines = [f"Ontology: {ontology.get('ontology_name', 'unknown')}"]
    selected_entities = entities if limit is None else entities[:limit]
    for item in selected_entities:
        label = item.get("label", item.get("concept_id", "unknown"))
        rule_lines = item.get("prompt_rules") or []
        lines.append(f"- {item.get('concept_id', 'unknown')}: {label}")
        selected_rules = (
            rule_lines
            if rules_per_entity is None
            else rule_lines[:rules_per_entity]
        )
        for index, rule in enumerate(selected_rules, 1):
            lines.append(f"  {index}. {rule}")
    return "\n".join(lines)
