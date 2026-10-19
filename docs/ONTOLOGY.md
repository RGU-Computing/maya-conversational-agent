# Main ontology and auxiliary clarification policy

These are separate sources with different responsibilities. The main ontology is
the partner ontology used across MAYA. The disambiguation ontology is MAYA's
application-level clarification policy; it is not another model in the partner
ontology.

| Source | Location | Used by |
| --- | --- | --- |
| Main partner ontology | `maya/framework/ontology/models/` | Supervisor model selection, HealthRAG, exercise decisions and ontology memory |
| Auxiliary disambiguation policy | `maya/framework/conversation/disambiguation_ontology.json` | HealthRAG clarification assessment and the supervisor's specialist-route guard |
| Legacy RDF/JSON exports | `maya/data/legacy_ontologies/` | Reference only; not loaded by the application |

## Main ontology

`framework/ontology/models/` is the former `framework/OntoModels/` directory.
Include the whole directory in delivery: `manifest.json`, all ten model folders,
`ONTO_MODEL_FORMAT.md` and `TRANSLATION_REVIEW.md`. Their contents are unchanged.
The source images remain in `maya/models_screenshots/`, as referenced by the
models. The existing translation status and review notes remain applicable.

`framework/ontology/registry.py` reads only models declared in the manifest. It
validates references and memory schemas, resolves agent defaults and dependencies,
and renders the selected concepts, relationships, procedures and participant
records. It does not load the auxiliary disambiguation policy or legacy exports.

The supervisor selects relevant model IDs. The registry then supplies the resolved
context to HealthRAG or the exercise decision stage, or to the supervisor's own
reply. `framework/memory/ontology.py` uses the same registry for extraction,
verification and storage under active memory bindings. Model IDs, binding paths,
versions and participant namespaces have not changed, so this directory move does
not require a memory migration.

## Auxiliary clarification policy

`framework/conversation/clarification.py` owns the single loader and renderer for
`disambiguation_ontology.json`. Both HealthRAG and the supervisor's route guard use
that loader. HealthRAG receives all rules; the route guard retains its existing
compact view. The JSON, rule IDs, numbering and rendered prompt text are unchanged.

HealthRAG combines these clarification rules with relevant main-ontology context,
participant memory, dialogue and available observations. The policy does not
define partner model IDs or memory bindings and is never registered as a main
ontology model. Keep it in the application delivery because those flows depend on
it, while identifying it separately from the partner ontology deliverable.

This is a structural change. It does not promote procedure contracts from prompt
guidance into deterministic graph gates or change which memory bindings are active.

## Legacy reference material

`Users.rdf`, `UsersData.rdf`, `Users.llm.json` and `UsersData.llm.json` are preserved
unchanged under `maya/data/legacy_ontologies/`. They are not part of the active
model registry. The unused `extractOnto.py` converter, which pointed at an obsolete
personal filesystem path, has been retired.

## Imports and verification

Shared code now imports `get_ontology_registry` or `OntologyRegistry` from
`maya.framework.ontology`. Clarification helpers are imported from
`maya.framework.conversation.clarification`. The previous
`maya.framework.ontologies` package has been removed.

Run offline regression tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s maya/tests -p 'test_*.py' -q
```

Explicit end-to-end checks use real models and retrieval with isolated RAM memory
and synthetic tool responses:

```bash
.venv/bin/python -m maya.tests.live_health_config_scenarios
.venv/bin/python -m maya.tests.live_memory_conversation_scenarios
.venv/bin/python -m maya.tests.live_phyxio_session_scenarios
```

These cover clarification and query rewriting, ontology memory across conversations,
participant isolation, exercise routing and provider failures. They do not exercise
live mirror hardware. Generated reports are written to ignored `output/`.

## Memory-aware retrieval

Context loading returns the full ontology prompt and its record-only portion from
the same participant/model-filtered memory read. HealthRAG rewrites the retrieval
query whenever those records are present, independently of the clarification
model's PICOT source labels. Clarification replies, conversation context and
observations also retain their existing rewrite paths. Definitions alone do not
trigger an extra model call.

If rewriting fails or returns an empty query, the fallback search includes the
question and supplied context, including record provenance, without adding the
ontology definitions to the search. Existing record filtering, projection and
limits are shared by both renderings. Non-self questions receive no participant
records; urgent responses still bypass retrieval. Cache fingerprints include the
record context supplied to retrieval, excluding older summaries that omitted it.

The `treatment_memory` live scenario deliberately removes `ontology_profile`
source labels to exercise the original failure. `memory_rewrite_failures` injects
a timeout and an empty rewrite through full supervisor conversations. Both use
real models and literature outside those fault injections. No partner model,
disambiguation rule, clinical requirement or prompt wording changed for this fix.
