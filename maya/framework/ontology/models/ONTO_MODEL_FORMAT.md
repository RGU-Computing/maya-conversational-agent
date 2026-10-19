# OntoModel Source Format

This folder contains draft ontology-model translations extracted from screenshots.
These files are the source of truth for runtime ontology selection, prompt
rendering, and reviewed long-term-memory bindings.

## Design Rules

- Use stable snake_case IDs for nodes, relations, procedures, and memory paths.
- Preserve the visual label from the screenshot in `label`.
- Do not encode label changes as new concepts unless the meaning changed.
- If a future screenshot renames a concept, keep the stable ID and add an alias.
- Mark unclear arrows, directions, and taxonomy groups with `confidence`.
- Keep memory bindings as draft placeholders until the memory layer is designed.
- Activate a memory binding only after its capture fields and update policy are
  reviewed. Runtime code must ignore draft bindings.
- Treat strict workflows as `procedure_contracts`, not just prompt guidance.

## Top-Level Fields

- `model_id`: Stable model identifier.
- `title`: Human-readable model name.
- `version`: Draft version of this translation.
- `status`: Current translation status.
- `source`: Screenshot provenance.
- `model_kinds`: Intended future uses, such as `data_schema`,
  `guidance_policy`, or `procedure_candidate`.
- `nodes`: Concepts/classes/instances visible in the screenshot.
- `relations`: Directed object-property style relations.
- `taxonomy_groups`: Subclass or instance lists shown as grouped boxes or
  inheritance arrows.
- `agent_contracts`: Draft notes on which agents may need this model.
- `procedure_contracts`: Strict or semi-strict process candidates for future
  LangGraph enforcement.
- `memory_bindings`: Draft future memory slots tied to ontology paths.
- `aliases`: Reserved for model evolution and migration.
- `review_notes`: Human review items and uncertainty notes.

## Interpretation

These files intentionally separate ontology structure from runtime enforcement:

1. The source model captures the ontology.
2. The runtime renderer builds prompt context from selected nodes/relations.
3. The memory adapter writes user facts only to active `memory_bindings`.
4. Strict procedure enforcement still requires deterministic
   LangGraph nodes, conditional edges, interrupts, and validators.

## Memory Binding Contract

An active memory binding adds a `capture` object:

- `properties_schema`: Standard JSON Schema for the record properties. This is
  the single runtime source for allowed fields, types, required values, and
  value constraints.
- `relation_mappings`: Property-to-relation mappings for fields represented by
  the ontology.
- `extension_fields`: Explicit application-memory properties not yet represented
  by screenshot relations.
- `identity_fields`: Properties used to build a stable upsert key. Leave this
  empty for event bindings that require a source-message-specific key.
- `update_mode`: `upsert`, `append_event`, or `supersede`.

Bindings with `status: "draft"` are prompt guidance only and cannot receive
runtime writes. This lets ontology translations evolve without silently
accepting unreviewed memory shapes.

Semantic save/no-save decisions use configured Azure LLM structured output.
Proposed records receive a second, independent LLM verification pass before
being validated and written to the LangGraph Store. Extraction instructions,
verification checks, and their meanings live in `manifest.json` under
`runtime.memory_policy`; binding-specific field semantics live in each model's
JSON Schema descriptions. Runtime Python performs generic check aggregation,
model selection, schema validation, provenance checking, identity construction,
and persistence. It contains no language- or domain-specific phrase lists.
