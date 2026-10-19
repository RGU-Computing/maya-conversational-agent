# Codebase structure

MAYA separates agent decisions, shared conversation behaviour, external services
and maintenance commands. Configuration lives beside the component that owns it;
deployment overrides use the existing environment variables.

```text
maya/
├── agents/
│   ├── config.py / config.jsonl      Agent prompt configuration
│   ├── health_rag/                  Health reasoning and clarification
│   └── phyxio_exercise_agent/       Exercise decisions and session analysis
├── cli/                            Maintenance and deployment commands
├── framework/
│   ├── conversation/               Session preferences, dialogue and presentation
│   │   ├── history.py              Recent dialogue selection
│   │   ├── clarification.py        Auxiliary disambiguation policy
│   │   ├── personalization.py      Participant session and personality selection
│   │   ├── responses.py            Separate display and speech output
│   │   └── text.py                 Shared text formatting
│   ├── integrations/               External service clients and adapters
│   │   ├── models.py               Shared chat-model client
│   │   ├── speech.py               ASR and TTS clients
│   │   ├── biomarkers/             Observation service and FHIR adapter
│   │   └── phyxio/                 Shared mirror connection and voice transport
│   ├── memory/                     Persistence and ontology-record memory
│   ├── rag/                        Literature loading and retrieval
│   ├── supervisor/                 Routing, orchestration and the main runner
│   └── ontology/                   Main partner ontology runtime
│       └── models/                 Manifest and partner ontology models
├── data/                           Literature and index metadata
├── models_screenshots/             Partner ontology reference material
├── session.example.json            Onboarding payload example
└── tests/                          Offline tests and explicit live scenarios
docs/                               Architecture and mirror handover
output/                             Ignored generated validation reports
```

## Where changes belong

| Change | Owner |
| --- | --- |
| Participant identity, voice options or personality selection | `framework/conversation/personalization.py` |
| Personality wording | `framework/conversation/personalities.json` |
| Dialogue window | `framework/conversation/conversation.json` |
| References, health notices and spoken progress messages | `framework/conversation/response_policy.json` |
| HealthRAG prompts and reasoning settings | `agents/health_rag/config.json` |
| Main partner ontology models and contracts | `framework/ontology/models/` |
| MAYA's auxiliary clarification rules | `framework/conversation/disambiguation_ontology.json` |
| Supervisor/exercise agent instructions | `agents/config.jsonl` and `framework/supervisor/prompts.py` |
| Biomarker provider, catalog and patient mapping | `framework/integrations/biomarkers/config.json` or a private override |
| FHIR transport and normalization | `framework/integrations/biomarkers/fhir.py` |
| Exercise history provider and comparison settings | `agents/phyxio_exercise_agent/session_config.json` or a private override |
| Exercise session normalization and analysis | `agents/phyxio_exercise_agent/phyxio_sessions.py` and `sessions.py` |
| Phyxio connection lifecycle, UI callbacks and audio | `framework/integrations/phyxio/service.py` |
| Reuse of the current Phyxio connection | `framework/integrations/phyxio/bridge.py` |

The exercise agent owns exercise-specific interpretation; the shared Phyxio
integration owns transport. The runner and tools use the same bridge so they do
not create separate connections. Its native dependencies are imported only when
Phyxio is enabled.

The supervisor composes agents and services. Conversation code is independent of
the supervisor and of native Phyxio dependencies. Agents import shared services;
shared services do not import agents. Maintenance commands compose these modules
without becoming dependencies of application code.

See [the ontology guide](ONTOLOGY.md) for the distinction between the main partner
ontology, auxiliary clarification policy and archived RDF/JSON exports.

The runner loads one `SessionConfig` and passes it to the graph and Phyxio bridge.
Tool identity comes from that session, outside model-visible arguments. UCLM must
start a new process/session for each participant; this version does not switch a
running process between participants. See [the handover guide](MIRROR_HANDOVER.md)
for the deployment contract and remaining live checks.

## Commands and migration

Run commands from the repository root. The application command is unchanged:

```bash
.venv/bin/python -m maya.framework.supervisor.run_with_vad
```

Maintenance commands now have one location:

```bash
.venv/bin/python -m maya.cli.check_tools --help
.venv/bin/python -m maya.cli.inspect_memory --help
.venv/bin/python -m maya.cli.build_index --help
.venv/bin/python -m maya.cli.latency_experiment --help
```

These replace `maya.check_tools`, `maya.framework.memory.inspect_memory`,
`maya.framework.rag.build_index` and
`maya.framework.supervisor.latency_experiment`, respectively. Update external
scripts that call the previous maintenance-command paths.

Onboarding imports use `maya.framework.conversation.personalization`. Existing
session JSON fields and environment-variable names are unchanged. If a deployment
override points to a moved bundled file, update the path: conversation JSON files
are now under `maya/framework/conversation/`, and biomarker configurations are
under `maya/framework/integrations/biomarkers/`. Private configuration files stored
outside the repository can remain where they are. Never put credentials or actual
participant mappings in the checked-in examples.

Offline regression tests and live scenarios remain under `maya/tests`; generated
reports belong in ignored `output/`. Partner ontology files and their reference
material are maintained separately and were not modified in this reorganization.
