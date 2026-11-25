# MAYA

See [the codebase structure guide](docs/ARCHITECTURE.md) for module ownership,
configuration locations and maintenance commands.
The [ontology guide](docs/ONTOLOGY.md) distinguishes the main partner ontology
from MAYA's auxiliary disambiguation policy and documents their delivery locations.

## Deployment versioning

Package and deployment version [0.9.0](https://github.com/RGU-Computing/maya-conversational-agent/tree/v0.9.0) include the study retention changes. The earlier `v0.9` tag identifies the historical mirror build.

Check out and verify the tag from a clean working tree.

```bash
git fetch origin tag v0.9.0
git switch --detach refs/tags/v0.9.0
git rev-parse HEAD
```

Record the tag and commit in the mirror installation record.

## Setup and mirror handover

For the complete identity/provider setup, live checks and troubleshooting commands,
follow [the mirror handover guide](docs/MIRROR_HANDOVER.md). Run
`python -m maya.cli.check_tools --help` for the model-free deployment diagnostic.

The supported application entry point is `maya.framework.supervisor.run_with_vad`.
The old Flask service, HTTP clients and alternate voice runners have been retired.

From the repository root, install the locked Python 3.13 environment:

```bash
uv sync --locked --inexact --python 3.13
```

`--inexact` preserves separately installed mirror packages such as ICE and Phyxio.
On the Linux mirror, UCLM's `libphyxio-connect` Debian package and its ICE/system
dependencies must be installed first. Its supplied Python wheel is made available
to the virtual environment through the existing mirror requirements:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Keep the deployment's existing `.env`; for a new installation, use `.env.example`
as the template and supply the actual service credentials. Supply a participant
configuration following `maya/session.example.json`, with the real participant ID.
Local `.env` and `maya/session.json` files are excluded from Git by default.
The reviewed [mirror profile](deployment/mirror/README.md) includes the explicitly
authorized session identity, confirmed FHIR mapping and non-secret settings used
on this mirror. Credentials are supplied separately; active private files remain
unchanged when this profile is committed.

```bash
.venv/bin/python -m unittest discover -s maya/tests -p 'test_*.py' -q
.venv/bin/python -m maya.framework.supervisor.run_with_vad --session-config /absolute/path/to/session.json
```

HealthRAG needs its configured PostgreSQL/pgvector literature collection. The
checked-in corpus and chunk snapshot contain 117 readable sources and 11,169 chunks
after removing three byte-identical PDF copies. For a new database, or a mirror
still using the earlier collection, stop MAYA and rebuild that literature collection:

```bash
.venv/bin/python -m maya.cli.build_index
```

This command replaces the configured literature collection; participant memory and
conversation checkpoints are separate. The empty `pgvector/` Git submodule entry
was removed; PostgreSQL still requires the pgvector extension installed on its server.

The biomarker and exercise-feedback providers remain disabled by default. Their
dummy examples explicitly enable synthetic fixtures; separate `fhir` and `phyxio`
examples select the implemented real adapters. Live deployment still requires
the confirmed FHIR patient mapping, Phyxio connection and UCLM login/logout handoff.
The Phyxio session-history wrapper uses the positional page argument required by
`libphyxio-connect` 26.5.29. Actual ICE, microphone and speaker checks must run on
the mirror. Partner ontology files are retained unchanged.

Model requests share the LangChain privacy middleware in
`maya/framework/integrations/privacy.py`. It replaces known session/profile
identifiers, email addresses and international `+` phone formats with temporary
aliases in system prompts, history and tool data. Each request has its own local
alias map; replies and structured results are restored before local memory
validation or tool use. Direct and typed calls use `model_call` in `models.py`;
HealthRAG also applies the middleware after its dynamic prompt. New model-call
sites must use this boundary. No additional LLM calls or NLP dependencies are used.

This covers known identities and the stated formats, not previously unknown
names or arbitrary personal details in free text. Numeric-only IDs are protected
in labelled identity fields and patient references, rather than replacing the
same number in clinical measurements. Biomarker prompts receive clinical fields;
raw FHIR records, patient lookup, stored conversations and logging remain local
application concerns. This middleware protects model requests; it does not redact
tracing/logging integrations. The privacy boundary tests run as part of
`python system_test.py --suite offline` and capture provider payloads without
network access.

The latest privacy validation results, timings and remaining acceptance issue
are recorded in [system_test_privacy_results.json](system_test_privacy_results.json).

The connector subclass only forwards Phyxio's documented callbacks; MAYA's speech
logic lives in `PhyxioService`, so it cannot override the library's internal state
methods. On the mirror's single avatar control, tapping again while recording or
transcribing cancels that voice input; cancelled text is not sent to the supervisor.
Wait for cleanup before starting a fresh recording. This does not cancel a response
already being generated or spoken. Audio
queue waits check cancellation every `MAYA_VOICE_POLL_SECONDS` (default 0.1 seconds),
and `MAYA_VOICE_SECONDS` bounds capture even when no frames arrive. On failure the
interaction is closed so the user can retry; `MAYA_VOICE_ERROR_TEXT` controls the
message. Select the mirror microphone with `MAYA_MIC_DEVICE` when its default input
is unsuitable. Linux device access still needs checking under the deployed service
account; this recovery does not make an unavailable device usable.
The terminal's `/voice` path also honors `MAYA_MIC_DEVICE` and
`MAYA_VOICE_POLL_SECONDS`, bounds capture when audio callbacks stop, and allows
another input after capture cancellation, device failure or transcription failure.

Voice-bridge validation on 19 October 2026: 246 offline tests passed, including
12 callback/recovery tests. A separate check loaded UCLM's actual 26.5.29 connector
source and exercised a microphone failure, retry, `run_with_vad`, supervisor routing,
session retrieval and speech delivery. Models were real; native clients, audio and
ASR/TTS were simulated. No real ICE connection or Linux audio device was exercised.

Generated outputs belong under ignored `output/`; they are not deployment inputs.
The regression tests, live scenario scripts, memory inspector, index builder and
latency diagnostic remain available for development and deployment checks.

Cleanup validation on 16 October 2026: 209 offline tests passed in a fresh locked
environment, plus five live HealthRAG scenario groups, five live exercise scenario
groups and a typed-input main-runner check. Live checks used real models and the
literature index, isolated participant memory and synthetic tool records; speech
playback and ICE were excluded. All 23 ontology source/configuration files matched
their pre-cleanup hashes. The snapshot and local database were checked against each
other for all 11,169 retained literature chunks.

## Latest exercise-session feedback

The exercise agent's read-only `get_latest_exercise_session` tool retrieves the
current participant's latest recorded session and compares it with up to
`comparison_limit` earlier sessions (configured as five). The supervisor and
exercise decision stage receive recent dialogue so a follow-up can refer to an
exercise without repeating its name. They must clarify when the activity or
subject is unclear; a request to start does not establish that a session finished.

The default provider is disabled. To use the synthetic fixtures:

```bash
MAYA_EXERCISE_SESSION_CONFIG="$PWD/maya/agents/phyxio_exercise_agent/session_config.example.json" \
  .venv/bin/python -m maya.framework.supervisor.run_with_vad --session-config maya/session.json
```

The example records belong to `participant-001`. For another test participant,
provide records with that `user_id` in the configured fixture file. No automatic
fallback to another participant exists. Session identities, dates, routine names,
targets and values are fixture data in `sessions.example.json`. Defaults and
comparison limits live in `session_config.json`; feedback instructions live in
the exercise entry of `maya/agents/config.jsonl`. `MAYA_CONVERSATION_CONFIG` may
override `maya/framework/conversation/conversation.json` to change the dialogue message limit.

Comparison selection and arithmetic run in Python: same routine and provider
`comparison_key`, then matching metric ID, units, target and target kind. Results include
per-metric counts, means and differences; missing values are not zero-filled.
Unfinished/future records are excluded, and ambiguous latest records are reported.
An interrupted session can be reviewed without presenting it as completed.
The response should identify the record's date, acknowledge effort and describe
changes without assuming that more repetitions or longer duration is healthier.

The tool accepts no model-supplied identity. It uses the session's `user_id`, checks
record ownership where supplied (see the native provider's binding limitation below), and exposes results in
`AIMessage.additional_kwargs["exercise_session"]` with `executed_tool` and the
synthetic flag. It neither starts an exercise nor writes synthetic results into
ontology memory.

### Phyxio session-history provider

The real adapter is `maya/agents/phyxio_exercise_agent/phyxio_sessions.py`. It reuses
the existing mirror connection and `getSessionList(page)` interface behind
`routine.get_sessions()`. Copy `session_config.phyxio.example.json` outside the
repository and select it with `MAYA_EXERCISE_SESSION_CONFIG`. Settings include:

- `first_page`: shared by history listing and session review (top-level setting,
  formerly inside `phyxio`). The example starts at page 0, as verified against the
  mirror response. `phyxio.max_pages` and `phyxio.end_of_history_error` bound traversal;
  a first-page error is not treated as empty history.
- `accepted_statuses`: eligible recorded exercise statuses, including partially
  performed routines with `executed`, `skipped` and `not-executed` results.
- `metric_statuses`: statuses whose measurements can contribute to performance
  statistics. Skipped and unperformed results remain in the source records, but
  their placeholder zeros are excluded from metrics and comparisons.
- `timeout_ms`: per-call timeout on a separate ICE proxy, without changing the
  original connection. Exactly one connected UI is required for this read path.
- `metrics`: documented result fields, labels, units, target fields and target kinds.
  Repetition goals and maximum time limits remain distinct. `unset_target_values`
  excludes configured placeholder targets (0 in the supplied Phyxio data), while
  preserving actual zero measurements and the original source fields.

The adapter scans bounded history pages until an empty page or the configured end
error, then sorts by recorded date. It does not assume newest-first pagination.
An unexpected error, repeated record or exceeded page limit returns unavailable
rather than feedback based on incomplete history. Increase `max_pages` for longer
histories after confirming the partner's pagination behaviour.

Native `date` becomes `recorded_at`; it is not labelled as an exact finish time.
Legacy synthetic fixtures using `ended_at` still load, but tool output now uses
`recorded_at`. No session ID or session-wide completion status is invented when
absent upstream. Individual exercise results and reports are preserved, including
their statuses. Only configured statuses are eligible; this does not establish
whole-routine completion. A latest skipped-only record is reported without
performance metrics rather than silently replaced by an older session.
Current routine names are optional, and current routine targets are never substituted
for historical targets. Comparable groups use recorded exercise IDs, repetition
goals and time limits; this does not prove the same prescription version or equipment.

Both history tools use the participant ID from the runner's `SessionConfig`.
The exercise-provider configuration contains no participant ID. The connection
receives the same session object as the runner, and rejects history requests for
a different participant before calling Phyxio. The model cannot supply an ID.

**Deployment boundary:** libphyxio-connect 26.5.29 supplies no participant identity
in its history response or login/logout event in the supplied ICE interface.
UCLM must supply the authenticated participant's startup payload and keep the
connected Phyxio UI logged into that participant. Stop MAYA on logout and launch
it with the next participant's payload on login. MAYA validates its internal
session identity; it cannot independently verify the browser's login from this API.

`python -m maya.tests.live_phyxio_session_scenarios` exercises supervisor routing,
the native adapter, pagination and feedback with simulated Phyxio responses and
real models. It does not prove live ICE connectivity or participant ownership.
The mirror must have UCLM's connector/ICE installed in the runner environment.

Validation on 19 October 2026 includes a replay of a supplied mirror response,
262 offline tests, and 14 native-provider supervisor scenarios. The supervisor
checks cover two onboarding identities, identity mismatches for both history tools,
partial routines, skipped-only records and history listing. A
`run_with_vad.main()` check exercises the supplied connector source, configuration,
history translation, routing, displayed feedback and the speech-output call.
Supervisor and runner checks use real models, synthetic Phyxio responses and
isolated RAM memory; live ICE and audio playback remain mirror-side checks.
UCLM's participant identity lifecycle also remains outstanding.

Offline tests: `python -m unittest discover -s maya/tests -p 'test_*.py' -q`.
Live supervisor-level scenarios:
`python -m maya.tests.live_exercise_session_scenarios`.
The live harness uses actual routing and exercise models, private RAM memory,
synthetic records and simulated mirror transport; it does not test ICE or audio.

## Biomarker tool

The shared observation tool is available to supervisor routing and HealthRAG's
ontology assessment. It is disabled by default. To run the dummy provider from
the repository root:

```bash
MAYA_BIOMARKER_CONFIG="$PWD/maya/framework/integrations/biomarkers/config.example.json" \
  .venv/bin/python -m maya.framework.supervisor.run_with_vad \
  --session-config maya/session.json
```

Try “Show my latest blood pressure and blood glucose measurements” for a direct
readout, or “What does my latest blood pressure measurement mean?” for HealthRAG.
The configured demo participant is the only participant mapped to these fixtures.
The example suppresses the displayed/spoken prefix with `synthetic_notice: ""`;
tool results retain `synthetic: true`. Set a non-empty notice to show and speak it.
Synthetic results are never
stored as reusable health-answer summaries or extracted as participant facts.

Configuration overrides live in `MAYA_BIOMARKER_CONFIG`; omitted fields come from
`maya/framework/integrations/biomarkers/config.json`. The example defines measurement catalog
IDs/descriptions, participant-to-patient mapping and a fixture path (relative to
the configuration file). Values, units, dates and grouped components are fixture
data in `observations.example.json`, not clinical rules embedded in Python.
Accepted statuses and tool instructions are configurable. There are no built-in
clinical thresholds or automatic unit conversions.

The existing routing/assessment model selects relevant catalog IDs using the
ontology and current context. A readout uses the supervisor; interpretation uses
HealthRAG. HealthRAG can also request observations itself, then reassess once
before asking the user for missing information. Lookups and the resulting
assessment are checkpointed across clarification resume. Only unattempted IDs
can be fetched again within that turn; new turns start with a fresh lookup.
Clinical relevance and subject identification remain model judgements; identity
binding, catalog validation, duplicate suppression and cache bypass are code.

Observation context reaches both query rewriting (including the first turn) and
answer generation. Turns with lookup results, including failures, bypass both
health-answer caches. Results retain their original timestamps and provenance;
“latest available” does not imply recent or suitable for every question. Empty,
unavailable, unsupported and unmapped outcomes remain distinguishable. UI clients
can inspect `AIMessage.additional_kwargs["biomarker_results"]`.

Tool results are retained in the graph's existing checkpointer; durable clinical
memory ingestion is not part of this integration.

### UCLM FHIR provider

The same tool supports `provider: "fhir"` through
`maya/framework/integrations/biomarkers/fhir.py`. Install UCLM's supplied
`libfhir-26.10.16-py3-none-any.whl` into the actual runner environment. The adapter
requires this version; the older OS-provided 25.5.14 package is not a substitute.
Use `uv sync --locked --inexact` to preserve separately installed partner packages.

Keep the decrypted credential JSON outside the repository (directory `700`, file
`600`). The documented credential forms are `url`, `token`, `is_secure`, or `url`,
`token: null`, `token_file`, `is_secure`. A relative `token_file` is resolved against
the credential JSON's directory; missing/empty files fail configuration. HTTPS
and certificate verification are required. Never log or commit credentials.

Copy `config.fhir.example.json` to a private location and configure:

- `fhir.client_config_path`: credential JSON path, absolute or relative to the
  biomarker configuration file.
- `patient_map`: confirmed MAYA user IDs mapped to FHIR Patient IDs. Oscar confirms
  the FHIR Patient ID is the Phyxio login username; obtain that username from the
  participant/UCLM rather than inferring it from a browser URL or local account.
- `catalog`, `fhir.resource_types` and `fhir.observation_codes`: measurement labels,
  documented SDK type names and accepted coding system/code pairs. The example
  uses the supplied 16 October 2026 guide's LOINC codes for blood pressure, heart
  rate and body weight. These are configuration, not a private SDK catalogue.
  All three mappings must cover the same measurement IDs. Unsupported SDK types
  cannot be validated through a public catalogue and may produce unavailable reads.

Select the private file with `MAYA_BIOMARKER_CONFIG` and launch the usual
`python -m maya.framework.supervisor.run_with_vad` with the session configuration.
The repository default remains disabled and the example patient map is empty.

The adapter imports `FHIRProxyClient` from `libfhir.client`, calls only
`client.get(resource_type, subject=patient_id)` and serializes each returned record
with `record.dump()`. It has no custom HTTP transport or private SDK dependencies.
It validates the exact patient reference and configured measurement coding, filters
accepted statuses, and retains every field returned by `dump()`, including components,
units and reference ranges. The validated subject is consequently present in the
result context; treat persisted tool results as participant data. No FHIR writes occur.
Malformed quantities (including string/boolean/missing numeric values) also fail
validation. Distinct records tied at the latest timestamp remain distinct in the
readout; a corrected status alone does not establish a replacement relationship.

Limitations verified against the installed 26.10.16 SDK:

- Scalar `valueQuantity` serialization drops units. Numeric quantities without a
  source unit or a coding system/unit code are rejected as `unavailable`. No units
  are inferred from the measurement type. Ordinary scalar heart rate/weight reads
  therefore remain blocked by SDK serialization even when the source had units.
- `get()` can return `[]` for a genuine empty search, network/proxy errors or omitted
  records, and `None` for HTTP errors. These outcomes become `unavailable`, never
  proof that the participant has no records. Exceptions and invalid records also
  fail the whole requested lookup rather than exposing partial results as success.
- The SDK accepts UTC timestamps with second precision (`...Z`); offsets and
  fractional seconds can cause records to be omitted. Absence-only observations
  and open periods are also not fully supported. Partial omissions are not visible
  through the public API. MAYA selects the latest **returned, parseable, accepted**
  records, preserving timestamp ties, and cannot guarantee complete history or the
  newest server record. Fields the SDK itself discards cannot be recovered.
- The public API has no supported request timeout. There is **no bounded network
  timeout** for a FHIR lookup. Remove the obsolete timeout setting when migrating
  private configuration, after backing it up. HTTPS verification remains enabled;
  redirect handling is owned by the SDK/requests and is not overridden by MAYA.

Offline tests characterize the installed SDK with an in-memory HTTP boundary and
exercise the adapter with synthetic public API responses. For real models,
supervisor routing, ontology lookup, clarification, retrieval and failure handling,
run `python -m maya.tests.live_fhir_scenarios`. For the actual typed main loop, run
`python -m maya.tests.live_fhir_runner_scenarios`. Both isolate participant memory
in RAM and simulate service responses; the runner check also simulates audio.
They do not establish live FHIR access or microphone/speaker operation. See
[the mirror handover](docs/MIRROR_HANDOVER.md) for current results and blockers.

The main runner handles model-provider API failures with the configured
`request_unavailable_text` in `maya/framework/conversation/response_policy.json`, then
accepts a fresh request in a new thread for the same participant. It does not
automatically replay a rejected message. Existing participant memory remains;
immediate conversational context in the failed thread is not resumed. Custom
response-policy files must include this field.

Phyxio capture requires both sufficient audio energy and WebRTC voice activity
before transcription. `MAYA_VOICE_MIN_SPEECH_SECONDS` (default `0.2`) controls the
minimum detected speech duration; `MAYA_VOICE_VAD_MODE` (default `2`, range `0–3`)
controls detector aggressiveness. The sample rate must be 8000, 16000, 32000 or
48000 Hz. Existing capture-duration, silence and microphone settings still apply.
These checks reduce noise-triggered transcriptions; they cannot guarantee correct
recognition in every acoustic environment. No language or transcript blacklist is
used, and the participant's ASR-language preference is unchanged.

Tests: `python -m unittest discover -s maya/tests -p 'test_*.py' -q`.
Explicit live model/retrieval smoke test with synthetic fixtures and RAM-only
participant memory: `python -m maya.tests.live_biomarker_scenarios`.
For clarification, verified memory reuse across fresh graphs, failed-save
recovery and user isolation, run
`python -m maya.tests.live_memory_conversation_scenarios`. This uses real models
and retrieval with an isolated RAM store; it does not test PostgreSQL or audio.

To change the demo participant, change `user_id` in the selected session JSON
and add the same ID to `patient_map` in the selected biomarker configuration,
mapped to `synthetic-patient`. Restart the runner. A new ID gets separate memory;
using the previous ID restores access to its saved facts. Changing
`MAYA_RUN_USER_ID` has no effect while a session JSON is selected. The synthetic
patient mapping is only for demonstrations; UCLM must supply real patient mapping.

## Onboarding configuration

`run_with_vad.py` accepts one participant and their preferences at startup:

```bash
python -m maya.framework.supervisor.run_with_vad --session-config /absolute/path/to/session.json
```

The payload is illustrated in [maya/session.example.json](maya/session.example.json):

```json
{
  "user_id": "pilot-participant-123",
  "preferences": {
    "personality_id": "supportive",
    "response_language": "ro",
    "tts_voice": "en-GB-LibbyNeural",
    "asr_language": "auto",
    "emergency_contact": null
  }
}
```

UCLM should resolve the card to a stable participant ID, persist the onboarding
choices in Phyxio, and supply this payload when launching MAYA. A JSON file must
contain a non-empty string `user_id`; an invalid file never falls back to another identity.
Unknown fields and invalid setting values are rejected before the graph starts.
This single ID feeds the supervisor, participant memory and exercise-history
tools. No participant-specific exercise-provider configuration is needed.
FHIR patient mapping remains a separate service requirement.

| Preference | Default | Meaning |
| --- | --- | --- |
| `personality_id` | `supportive` | Warm, patient explanations that acknowledge expressed concerns; alternatives are `concise` and `default` (the original style). |
| `response_language` | Shared configuration, initially `en` | Generated user-facing text and clarification language; supported codes and full names are in `response_languages.json`. |
| `tts_voice` | `en-GB-RyanNeural` | Azure Speech voice name. Both local and mirror audio use it. |
| `asr_language` | `auto` | Whisper automatic detection, or a supported two-letter language code such as `en` or `es`. |
| `emergency_contact` | Unset | Optional pilot-approved phone number for urgent replies. Set it in `session.json` as a string of 3–15 digits, with an optional leading `+` and spaces or hyphens between digits. MAYA inserts it in the reply locally. |

Voice names and recognition languages must be supported by the deployed provider.
The local validator checks their format, not remote availability. Recognition
language is separate from both generated response language and the voice used to
speak replies. `pt-PT` is a response-language code, while explicit Portuguese ASR
uses `pt`; response language is never forwarded to ASR. The two voice examples already used
in MAYA are Ryan and Libby; the shared fallback is now Ryan. Set Libby explicitly
to retain the mirror's previous fallback voice.

The contract, defaults and personality descriptions live in
[personalization.py](maya/framework/conversation/personalization.py). UCLM can obtain a
JSON-serializable schema and option descriptions without importing ICE or starting
MAYA:

```python
from maya.framework.conversation.personalization import SessionConfig, onboarding_options

options = onboarding_options()  # JSON Schema, defaults, personalities and code-to-language names.
session = SessionConfig.model_validate(payload)

# For a Python launcher, once per process:
from maya.framework.supervisor.run_with_vad import main
main(session)
```

An explicit supported `preferences.response_language` in the session takes
precedence over the default in
[`response_languages.json`](maya/framework/conversation/response_languages.json).
Omitting it preserves compatibility with older sessions and currently selects
English. There is no response-language environment override. Invalid codes fail
session validation before model execution; restart the runner when changing its
startup preferences. The shared file also owns the human-editable instruction
template, resolved using the full language name. Personality remains independent.

The instruction reaches supervisor replies, biomarker readouts, HealthRAG answers,
clarifications and urgent guidance, and exercise responses. Machine-only routing,
memory verification and query rewriting keep their existing contracts. HealthRAG's
existing instruction fingerprint separates cached answers by response language;
voice-only changes do not invalidate text answers.

This controls model-generated wording, not complete interface localisation. Fixed
messages, disclaimers, reference headings and some failure fallbacks remain in
their configured language. See [the language implementation and validation guide](docs/RESPONSE_LANGUAGE.md)
for the complete boundary, test commands and results. Select a compatible speech
voice explicitly; response language does not silently replace the chosen voice.

For the CLI, an explicit `--session-config` path takes precedence over
`MAYA_SESSION_CONFIG`. Fields in the file override environment defaults:
`MAYA_PERSONALITY`, `AZURE_SPEECH_VOICE` and `MAYA_ASR_LANGUAGE`. A directly
constructed `SessionConfig` uses its declared defaults, independent of these
environment settings. Speech credentials and recording/device settings remain
deployment environment variables; they are not onboarding fields.

Without a JSON file, the runner requires `MAYA_RUN_USER_ID`; there is no default
participant identity. Missing or blank IDs fail before the graph starts.
Pilot launches should supply the participant payload resolved by Phyxio.
Settings remain fixed for that process. UCLM must stop it on logout and start a
new process with the next participant's payload; editing the file does not switch
an active session. No HTTP endpoint or new preference database is introduced.

Each runner launch creates a fresh conversation in RAM. A thread can resume a clarification within that session. Ending the process releases the dialogue, reply cache and unfinished memory inputs, even if the next launch uses the same participant and thread label. PostgreSQL retains verified general memories, supported profile fields and validated ontology facts, with source identifiers where needed. Source messages and evidence quotes are used during verification in RAM. Run study sessions with provider tracing disabled and without transcript capture. Review any data and backups created by older releases separately.

Personality instructions affect generated replies and clarification wording in
the supervisor, health and exercise agents. They add no model calls and retain the
existing evidence, safety and tool instructions. Fixed confirmations and error
messages remain unchanged. Health answer cache entries carry a personality/version
key; older entries are left in storage but are not reused. Changing only voice or
recognition language does not invalidate the text cache. Personality instructions
are included in the cache fingerprint; after editing `personalities.json`, restart
MAYA so it uses the updated instructions and cache scope.

Run the offline regression suite from the repository root:

```bash
python -m unittest discover -s maya/tests -p 'test_*.py' -q
```

It uses model, speech, database and transport doubles. It verifies configuration
propagation and execution contracts, not live model quality, voice availability
or the Linux mirror connection; those still require deployment smoke tests.

## Health references, speech and notices

Health clarification uses the rules in
[disambiguation_ontology.json](maya/framework/conversation/disambiguation_ontology.json)
alongside selected main ontology models and participant records. Relevant remembered
treatments, including past cancer exposures, can resolve a question without
asking again. Clarification is needed when the relevant treatment remains missing
or ambiguous, such as a newly mentioned treatment absent from memory or a question
about someone else. General educational questions can still be answered directly.
Rendered rules include their identifiers and numbers for attribution.

Completed HealthRAG messages include `additional_kwargs["clarification"]` with
the assessment status, question type, missing fields, selected ontology models
and rule checks. The runner prints a compact clarification trace separately from
the retrieval trace; diagnostic fields are not spoken. Failed assessments,
missing/unattributable rule checks, omitted clarification-policy nodes or rules,
unsatisfied rules without a named missing requirement,
or empty required clarification questions stop the answer before retrieval/cache
lookup. An incomplete assessment gets one structural repair attempt; it must pass
validation before continuing. Unmet ontology requirements take priority over
generic PICOT gaps when choosing the clarification question, including requirements
whose names match PICOT fields. The response text for that failure
is configurable as `clarification_unavailable_text` in the response policy.
These checks make the decision auditable; they do not certify model judgements
or clinical correctness.

Loading ontology context retries pending memory jobs within the running session.
For relevant updates belonging to the same user, context loading waits up to
`recovery_wait_seconds` (supplied value: 2 seconds). If verification is still
pending or failed again, MAYA returns `recovery_unavailable_text` rather than
asking for the same information or answering without checking it. The user can
try the question again after background processing finishes. Unprocessed input
is never injected as clinical context; the existing extractor, verifier and schema
checks govern durable writes. Successful processing, including rejection, removes
the pending job. These settings and `pending_job_limit` are configured in
`maya/framework/ontology/models/manifest.json` under `runtime.memory_policy`. No model calls are added
when there is no pending work, and unrelated users/models do not wait for it.

The assessment is checkpointed as a LangGraph task. Resuming a clarification
reuses that decision rather than running the assessment again; a new question
gets a new assessment. Retrieval receives the original question, the clarification
asked and the user's reply. The configurable `health_output_instruction` tells
the answer model to acknowledge unavailable details and avoid conclusions that
depend on them; this is generation guidance, not a second validation of the reply.

Shared model requests require `MAYA_MODEL_TIMEOUT_SECONDS` and
`MAYA_MODEL_MAX_RETRIES` in the environment (see `.env.example`; supplied values
are 45 seconds and zero retries). These apply to individual requests, not the
whole turn. A clarification assessment request failure returns
`clarification_unavailable_text`. Restart MAYA after changing these settings.

The main runner, `run_with_vad.py`, displays retrieved references after each
completed HealthRAG answer and places the health disclaimer last. TTS receives
the answer and disclaimer only. Clarification questions stay short. The runner
waits for the user and does not display or speak an automatic introduction.

When HealthRAG starts, before clarification assessment, the runner schedules
`health_processing_status` after `health_processing_delay_seconds` (2 seconds in
the supplied response policy). A response or error arriving sooner cancels it.
Processing continues during speech; the answer or clarification question waits for
local acknowledgement playback, or follows it in Phyxio's existing TTS queue.
It is spoken at most once per user turn, including retries and clarification resumes.
Ordinary conversation does not trigger it, and it is not stored in conversation or
health memory. Custom policy files must include both fields; the status field was
renamed from `health_retrieval_status`. UCLM can consume the same
`{"event": "health_processing_started"}` custom stream event to drive its UI.

Presentation wording, labels and the health output-format instruction live in
[response_policy.json](maya/framework/conversation/response_policy.json). Supply a complete
replacement with `MAYA_RESPONSE_POLICY=/absolute/path/to/policy.json`; it is
validated and loaded once per process. Restart to apply changes. Reference and
disclaimer formatting adds no model calls. The supplied English policy can be replaced for a
deployment's language; ASR language selection does not translate it.

### Editing HealthRAG without changing Python

The `health_rag.config_path` entry in [agents/config.jsonl](maya/agents/config.jsonl)
points to [health_rag/config.json](maya/agents/health_rag/config.json). Paths are
relative to the agent registration file, including when replacing that file with
`MAYA_AGENTS_CONFIG=/absolute/path/to/agents.json`. Use valid JSON or JSONL;
comments and trailing commas are not supported. URLs and literal braces in prompt
text are preserved. Restart MAYA after edits.

| Setting | What partners can change |
| --- | --- |
| `prompts.assessment` | How the model applies the supplied ontology and fills the clarification assessment. |
| `prompts.assessment_repair` | Instructions for correcting missing or invalid ontology rule checks. |
| `prompts.clarification` | How to phrase the selected clarification question. |
| `prompts.query_rewrite` | How conversation, records and observations become a retrieval query. |
| `prompts.answer` | Health explanation style, grounding and handling of insufficient information. |
| `prompts.urgent_answer` | Immediate safety wording when the assessment identifies urgency; bypasses retrieval and answer reuse. |
| `retrieval` | Result count, citation count, excerpt size and PGVector similarity threshold. |
| `cache` | Similarity and token-overlap thresholds for reusing prior answers. |
| `messages` | Empty-result and cached-answer wording. |

These are literal instructions, not templates: Python attaches the current
question, ontology, memory, observations and evidence. Required fields are
validated at startup; misspellings and empty prompts fail clearly. The old
HealthRAG registration fields `prompt`, `tools` and `model` are rejected rather
than ignored. Model deployment remains in the shared Azure environment settings.

Clinical requirements remain in the partner ontology files. HealthRAG prompts
apply those requirements; they do not redefine the ontology. Structured output
schemas and consistency checks remain in Python. The supervisor's structured
`question_subject` controls access to the signed-in user's clinical records and
observations. Health questions about another person, general education or an
unresolved subject receive ontology definitions without those personal records.
Personality wording is shared
through [personalities.json](maya/framework/conversation/personalities.json); onboarding still
selects the existing personality IDs. The empty `default` instruction adds no
style override; `supportive` remains the session default. Reference/disclaimer rendering remains in
`response_policy.json`; its `health_output_instruction` covers presentation only.
The previous `assessment_repair_instruction` field has moved to
`prompts.assessment_repair`; remove it from custom response-policy files.

Clarification replies and actual participant records loaded for the selected ontology
models trigger query rewriting, independently of the model's PICOT source labels.
Conversation context and observations also trigger rewriting. If rewriting fails
or returns nothing, the search retains the supplied context and record provenance;
ontology definitions alone do not trigger a rewrite. Empty retrieval results
return `messages.no_evidence` directly, with no medical answer generation. Urgent
safety responses take their own path and do not depend on retrieved documents.
Background memory verification checks coverage for every proposed record and
property. An incomplete result gets one retry for the omitted records using the
same ontology policy; if it remains incomplete, the existing pending job is kept
for another attempt within the session. A complete semantic rejection is not retried or treated as approval.

Answer reuse includes a fingerprint of the HealthRAG configuration, disambiguation
ontology, rendered ontology/profile context, record context supplied to retrieval,
response style and output instruction.
Changing those inputs prevents reuse of answers generated under the old settings.
Reference headings and the disclaimer are rendered from the current policy on
every delivery, including cached answers. This is not a corpus-update invalidation
mechanism; corpus/index changes still require managing the answer cache separately.

Run the offline regression suite with
`.venv/bin/python -m unittest discover -s maya/tests -p 'test_*.py'`.
For live supervisor scenarios, run
`.venv/bin/python -m maya.tests.live_health_config_scenarios`.
The live suite uses the configured models and literature, synthetic observations
and isolated in-memory storage. It does not test microphone, TTS playback or ICE.
Its answers, workflow checks and memory extraction/verification traces are saved
under `output/health_config_live*.json`; inspect those alongside the pass/fail results.

For UCLM's UI, each completed HealthRAG `AIMessage` exposes
`additional_kwargs["presentation"]` with `answer_text`, `references`, `disclaimer`,
`display_text` and `speech_text`. Render the structured fields or `display_text`;
send only `speech_text` to TTS. `AIMessage.content` remains the answer alone for
conversation context. Each reference contains `title`, `file_name` and `page`
(1-based PDF page, or null if unavailable). These identify the actual excerpts
supplied to the model, deduplicated per source/page; they are not a claim that
every generated statement is supported. Titles currently come from corpus
metadata, often filenames. No article titles, URLs or DOIs are guessed, and local
directory paths are excluded from presentation. Fresh and cached answers use the
same renderer; older caches that retained only two citations are bypassed.

Onboarding can read the validated policy without starting models or ICE:

```python
from maya.framework.conversation.responses import load_response_policy, opening_response

policy = load_response_policy().model_dump(mode="json")
opening = opening_response()  # display_text and speech_text
```

`privacy_notice_url` is deliberately null until the partners supply the pilot's
actual notice. If configured, `opening_response()` includes that link in its
display text for onboarding. This helper does not display or speak anything;
the runner does not call it. An external UI must present the AI disclosure and
applicable privacy information before participant input, including after a new
participant logs in; live RFID switching still awaits UCLM's lifecycle contract.

Reviewed on 12 October 2026: the [MAYA website privacy policy](https://maya-horizon.eu/privacy-policy/)
describes website visitors and cookies, and its [contact page](https://maya-horizon.eu/contact/)
lists a general project email. Neither establishes the chatbot pilot's controller,
DPO contact or health-data processing notice. Do not reuse it as the pilot notice
without partner confirmation. The [European Commission's Article 50 guidance](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act)
addresses clear AI disclosure at the first interaction; the [EDPB's transparency guidance](https://www.edpb.europa.eu/sme/be-compliant/respect-individuals-rights_en)
separately requires accessible information about the actual personal-data
processing. The health disclaimer is not a privacy notice or a claim of GDPR/AI
Act compliance. Clinical grounding quality and the Linux ICE/display integration
still need their own validation.

## Inspecting Memory

The memory inspector provides a read-only JSON view of the users, namespaces,
stored values in the configured PostgreSQL database. A separate inspector process cannot see a running session’s RAM checkpoints.

```bash
# Discover user IDs and their memory namespaces.
python -m maya.cli.inspect_memory --list-users

# Show all memory namespaces and values for one user.
python -m maya.cli.inspect_memory --user-id test1

# Focus a demonstration on one or more namespaces.
python -m maya.cli.inspect_memory \
  --user-id test1 \
  --namespace profile \
  --namespace ontology

```

Long strings are shortened to 500 characters by default to keep demonstrations
readable. Use `--full` to show complete content, or `--max-string-length 1000`
to choose another display limit. The script does not write to memory.

## Partner system acceptance suite

The repository-root entry point is **`system_test.py`**. It compares the actual
MAYA application across Azure chat deployments and the nine response languages
declared in `response_languages.json`. No production configuration needs to be
edited to change the model or language under test.

Run it with the Python environment that contains MAYA and the partner SDKs. The
offline tests need the normal application dependencies; the FHIR contract tests
also need the supplied `libfhir` wheel. Missing SDKs are reported as skipped and
make the run **INCOMPLETE**, rather than silently passing.

```bash
# Read the scenario requirements without credentials or network calls.
python system_test.py --list

# Check the test runner itself, then run all existing offline regression tests.
python system_test.py --self-test
python system_test.py --suite offline

# Focus on memory: save/recall, user isolation, no-write inputs, corrections,
# repeated facts, and a mixed recall/update after changing language.
python system_test.py --suite memory \
  --models configured gpt-5.4-mini --languages all --jobs 2

# Check application notices and health/measurement delivery in all languages.
python system_test.py --suite localization \
  --models configured gpt-5.4-mini --languages all --jobs 2

# Compare two deployments in English first.
python system_test.py --suite matrix \
  --models configured gpt-5.4-mini --languages en

# Full comparison: regressions, all language workflows and cloud speech.
python system_test.py --suite all \
  --models configured gpt-5.4-mini --languages all --jobs 2

# Full software acceptance for German and Greek on the active deployment.
python system_test.py --suite all --models configured --languages de el
```

`configured` uses the active Azure settings from the environment or `.env`.
Explicit deployment names override only the deployment name. If a candidate uses
a different Azure resource, provide a private dotenv file containing its
`AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_API_VERSION`, then add:

```bash
--model-env gpt-5.4-mini=/absolute/private/candidate.env
```

The suite never uncomments or rewrites `.env`. Each model/language job uses a
fresh process so cached models and module-level extractors cannot retain the
previous model. Actual deployment selection and returned model identifiers are
recorded in the JSON report. Other LLM providers require support in MAYA's model
factory; passing an Azure deployment name does not validate a different provider.

The live matrix contains 55 functional scenarios per model/language pair. It
combines the established conversation tests with the root-level
`system_test_workflows.py` and the explicit native-language prompts in
`system_test_prompts.json`. Coverage includes:

| Function | Representative acceptance scenarios |
| --- | --- |
| Supervisor and health agent | Ordinary conversation, fact-only clinical updates, native-language health questions, literature retrieval, urgent bypass and missing evidence |
| Clarification | Missing treatment context, resume without refetching, and an unknown-test follow-up |
| Memory | Verified treatment recall, language changes, profile/general-memory recall, corrections, repeated facts, no-write inputs, mixed recall/update, and participant/third-party isolation |
| Cache | Reuse when the effective context is identical; invalidation when language or context changes |
| Biomarkers | Values/units, scoped reads, unavailable data, unitless quantities, unsupported temperature and mixed available/missing readings |
| Phyxio | Every registered exercise tool has a positive native-language request and an explicitly negated request; history errors and safety holds are also checked |
| Localization | All nine response preferences, native input, mixed input, default language, exact medication/dose retention and fixed application messages |
| Voice | Generated native-language speech through the actual runner, VAD, ASR, supervisor and TTS; no reference reading, silent startup and resource cleanup |

Medication spelling checks follow the current application's **verbatim-copy**
policy. A failed spelling check is distinct from a language being unsupported.
Language identification uses one fixed review deployment (`--judge-model`,
defaulting to `configured`) across the comparison. These judgments supplement
structural assertions and do not replace native-speaker or clinical review.
Configured notice translations are also compared against their source wording for
omitted qualifications or added advice. The focused `localization` suite runs six
notice scenarios plus native health and blood-pressure workflows; the full matrix
includes those six notice scenarios too.
Each scenario runs once per selected model/language pair. This is acceptance
coverage, not an estimate of long-run reliability; use separate report directories
for additional repetitions and retain earlier failures for comparison.
The general-memory scenario checks both storage and the recalled experience;
a generic acknowledgement does not count as recall. Its content assertion uses
explicit translation stems for the controlled fixture. Recall must leave both the
original user's records and the other user's empty store unchanged. `--suite memory`
runs the six profile/general-memory scenarios with language review. Its edge cases
use English input across response-language settings; the original save/recall cases
use native-language input. Negated exercise requests
use the same English wording across response-language settings; positive exercise
requests and the principal conversation/measurement cases use native-language input.

Results go into a new `output/system_test/<timestamp>/` directory:

- `report.html`: readable scenario results, requirements and expandable evidence.
- `report.json`: model/language labels, application/configuration/test fingerprints, package versions,
  individual assertions, generated text and model audit information.
- `junit.xml`: CI-compatible failures, errors and skipped tests.
- Each job directory contains detailed evidence and a redacted worker log.

Exit status is `0` for a passing selected scope, `1` for behavioral failures,
`2` for incomplete/error results and `130` for interruption. A hard per-job
deadline (`--timeout`, default 1,800 seconds) prevents hanging SDK calls or memory
workers from leaving the suite running indefinitely. `--output` must name a new
directory. Use `--languages en es`, `--models ...` or `--suite speech` to narrow a
rerun. The runner does not automatically retry failed assertions into a pass.
Only one job per tested Azure deployment runs at a time, even when `--jobs` is
higher. Separate deployments can run concurrently; the language reviewer also
uses its configured deployment quota.

An interrupted run can be resumed with the same suite, models, language list and
model credential files:

```bash
python system_test.py --suite all \
  --models configured gpt-5.4-mini --languages all --jobs 2 \
  --resume output/system_test/YOUR_RUN_DIRECTORY
```

Resume retains completed results, including behavioral failures. Incomplete/error
attempts are archived before being rerun. Earlier failures that change outcome on
an infrastructure rerun are flagged for review. Configuration/dependency changes
and application or test source changes require a new report directory. A report-directory lock prevents two runners
from writing the same report concurrently.

Live model and speech suites make billable provider calls. All participant
records are synthetic, conversational memory is isolated in RAM, and Phyxio
actions are intercepted at the service boundary. Literature retrieval uses the
local corpus with a RAM index and deterministic embeddings for reproducibility;
this is not a production semantic-retrieval benchmark. No microphone is opened
and no audio is played during the cloud speech test.

A passing run covers the selected software scenarios. Deployment review must
also check physical audio/avatar behavior, real mirror connectivity, live FHIR
ingestion, PostgreSQL durability, concurrent users/load and native-language
quality. The report lists those unexercised areas explicitly. Existing application
defects remain visible as failures; this suite does not change application
behavior to make a result pass.
