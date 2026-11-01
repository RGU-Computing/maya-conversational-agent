# Shared response language

MAYA resolves `preferences.response_language` from the same startup `SessionConfig`
as personality, ASR language and speech voice. The authoritative mirror checkout
is `/home/rgu/Documents/MAYA`; its tested interpreter is
`~/.local/share/maya-validation/mirror-20261019T184537Z/venv/bin/python`.
The active service selects its private `session.json` in that directory. Keep its
existing `user_id`, FHIR mapping and remaining preferences when choosing a language.

## Configuration and onboarding

Edit `preferences.response_language` in the selected session JSON, for example:

```json
{
  "preferences": {
    "personality_id": "supportive",
    "response_language": "ro"
  }
}
```

This fragment shows only the preference fields; a complete startup payload still
requires its existing `user_id`. An explicit supported code wins. If omitted, the
default comes from
[`response_languages.json`](../maya/framework/conversation/response_languages.json),
initially `en`. Invalid codes fail session validation before model execution.
There is no additional environment-variable language selection. Existing session
file selection and precedence (`--session-config`, then `MAYA_SESSION_CONFIG`)
remain unchanged. Restart MAYA to load changed startup preferences/configuration.

| Code | Language |
| --- | --- |
| `en` | English |
| `ro` | Romanian |
| `it` | Italian |
| `es` | Spanish |
| `fi` | Finnish |
| `pt-PT` | Portuguese (Portugal) |
| `fr` | French |
| `de` | German |
| `el` | Greek |

The mapping, default and editable instruction template have one configuration
source. `onboarding_options()` exports `languages`, `default_response_language`
and a JSON Schema with the supported code enum. The example session is updated.
Unsupported variants such as `pt-BR`, the bare response code `pt`, or a full name
instead of a code do not silently fall back to another language.

For Romanian the resolved instruction begins:

> Write all user-facing responses and clarification questions in Romanian.

The remaining template makes this preference authoritative over the input/history
language, requires verbatim medication names, identifiers, codes and unit symbols,
preserves numerical values/uncertainty and original evidence quotations, and keeps
machine-only fields canonical. Personality remains a separate field. A participant
can ask in English while receiving Romanian generated wording.

## Active paths and cache scope

The enabled specialists are HealthRAG and Phyxio exercise, alongside the supervisor.
The existing `Preferences.response_style()` instruction path now combines the
independent tone and language instructions. Its consumers already cover:

| Path | Where the instruction is used |
| --- | --- |
| Supervisor direct reply | Final response prompt; excluded from route classification |
| Biomarker result explanation | Readout prompt after the participant-scoped lookup |
| HealthRAG clarification | Wording stage after the machine-only ontology completeness assessment |
| HealthRAG answer and urgent guidance | Dynamic final-answer prompt, including urgent retrieval bypass |
| Exercise clarification, safety hold, out-of-scope response | User-facing fields of the structured decision |
| Exercise history, feedback and tool-failure explanation | Response stage grounded in the executed tool result |

The exercise delegation already forwards the complete preferences object.
HealthRAG already receives the resolved instruction in graph state. Its existing
`cache_scope` hashes that instruction with prompts and ontology context, and both
exact and semantic cache paths use the resulting scope. No second cache key or
language-specific cache implementation was added. Changing language invalidates
reuse of an answer generated under a different instruction while preserving the
participant namespace and existing memory. Changing only voice does not change
the text-answer scope.

Router enums, tool names/arguments, ontology IDs, memory extraction/verification,
query rewriting, evidence quotations and stored history retain their existing
contracts. No ontology, clinical rule, embedding, collection, SDK, FHIR mapping or
exercise binding is changed by this feature. Generated answers use the response
instruction directly; application-owned notices use the shared translation path
described below.

## Speech and application-owned notices

`response_language` controls generated responses and application-owned health and
memory notices. ASR language and the selected TTS voice remain separate settings;
changing the response language does not automatically choose a different voice.

`conversation/response_policy.json` declares the source language of application
message configuration and a generic translation instruction. Source wording stays
partner editable. The shared `responses.localize_messages()` function translates
only user-facing fields through the configured model's LangChain structured-output
API. There are no language-specific translation tables or response keyword rules.

The shared path covers disclosures, health disclaimers, reference headings and
labels, progress acknowledgements, clarification/request errors, HealthRAG empty
and no-evidence messages, cached-answer prefixes, synthetic-data notices and the
ontology-memory pending notice. Memory save/recall replies are generated with the
session's existing response-style instruction. Profile verification receives only
changed factual fields; schema defaults for unknown fields are not proposed facts.
Explicit removals of stored values still require verification.

Only the ordered message strings are sent to the translator; configuration keys
stay in Python. Structured output requires one nonempty translation per message.
Translations are cached in RAM by source wording, source/target instruction and
model deployment/endpoint/API version. Changing a session language cannot reuse
another language's notices. Restart to load edited configuration. All application
message files must use the source language declared in the response policy;
independently mixed source-language files are not the supported configuration.

The runner prepares progress/error messages before accepting input. HealthRAG
prepares its assessment-error wording before assessment. These error paths do not
need another model call during an outage. A cold translation failure propagates;
it does not silently fall back to English. Initial localization therefore requires
model access and adds first-use latency. Translation quality remains model dependent
and needs native review, especially for safety wording and regional variants.

Health answers, original reference titles, source identifiers, numerical values,
privacy URLs, delay settings and machine-only prompts are not passed through the
notice translator. References remain display-only. The same translated disclaimer
is appended once to display and speech. Processing speech keeps its existing delay
and completes before final speech; there is no unsolicited startup speech.

This change does not localize Phyxio's listening/transcribing/cancellation UI,
exercise-service raw failure strings, or developer-facing terminal/device messages.
Those adapter paths are separate from the health/memory notice presentation path.

## German and Greek extension — 2026-11-01

German (`de`) and Greek (`el`) are now available through the same session and
onboarding configuration, bringing the supported response choices to nine.
Both languages have native workflow and speech fixtures. The full software suite
for these languages can be run with:

```bash
.venv/bin/python system_test.py --suite all --models configured --languages de el
```

The first run exposed an ASCII-only tokenizer in the deterministic literature and
memory embedding helpers: a Greek-only query produced a zero vector and failed
cosine search. Both helpers now tokenize Unicode words and normalize equivalent
accent encodings. Regression tests exercise actual vector-store retrieval and
preservation of existing ASCII-only vectors. **Existing persisted deterministic
indexes containing non-ASCII text need consistent re-embedding before relying on
similarity search with the new tokenizer.** No persistent index or participant
data was rewritten here; these runs build fresh RAM indexes. The production
HuggingFace literature-embedding backend is unaffected by this tokenizer change.

The acceptance scorer now recognizes explicit Greek transliterations of its
controlled personal/place names, including in participant-isolation checks.
Medication names, unit symbols and stored profile fields retain their stricter
assertions. All eight affected cases in the first run were reviewed from saved
outputs, including passing cases; three Greek spelling-only failures were removed
in a separate evaluation record. Original reports were retained unchanged.

Validation used the configured `gpt-4.1-mini` deployment, real model/ASR/TTS calls,
synthetic participant services and isolated RAM storage. The corrected application's
full run completed all five jobs with **625 passes and 5 failures**, with no
infrastructure errors. Offline regressions pass **352/352** (including five existing
local benchmark tests); harness self-tests pass **20/20**.

| Language | Functional scenarios | Language/translation reviews | Cloud speech job |
| --- | --- | --- | --- |
| German | 54/55 | 79/79 | PASS |
| Greek | 51/55 | 79/79 | PASS |

The five full-run failures are German experience recall through a write route;
Greek correction recall through a write route; repeated clarification after a
Greek-to-Romanian language change despite available context; a handled assessment
error instead of another-person clarification; and Greek weight output translating
the required `kg` symbol to `κιλά`. Identity checks passed. Unit preservation failed
in both full executions and remains unresolved.

After the full run, the German recall fixture was clarified to remove the ambiguous
phrase “nach meiner Aussage.” A focused German memory run passes **5/6 functional
scenarios and 24/25 total checks**: the clarified native recall case passes, but
correction recall still uses the write route despite returning the correct city.
Application source hashes match the full run; only this test-prompt file changed
before the focused follow-up. No full-run result was replaced by the follow-up.
Acceptance remains failing; these are not deployment or native-speaker sign-offs.

Evidence under `output/system_test/`:

- `20261101T005832Z_0fdeeb/`: initial full run, including the Unicode error;
  `evaluation-review-de-el.json` contains the saved-output scoring review.
- `20261101T011858Z_5b53c0/`: full run after the tokenizer and scoring corrections.
- `20261101T013741Z_9c7665/`: German memory follow-up with the clarified prompt.

Each run retains HTML, JSON and JUnit reports. Physical mirror/audio behavior,
live FHIR ingestion, PostgreSQL durability, and clinical/native-language quality
remain separate deployment checks.

## Measurement and routing follow-up — 2026-10-30

The four requested fixes use the existing request schema, lookup function and
supervisor/HealthRAG paths. Measurement identity is resolved against the configured
catalog before lookup; ambiguous and unsupported requests cannot fetch a substitute.
Each requested measurement keeps its own validated outcome, so an unavailable
measurement cannot discard another valid result. Fact-only clinical updates use the
actual writable ontology catalog and the existing memory finalizer. The router also
receives the actual profile schema. No per-language rules, additional graph nodes or
repair pass were added.

The full suite completed all 29 jobs using GPT-4.1-mini and GPT-5.4-mini across all
seven configured languages. The four priorities passed 70/70 checks: unsupported
measurement selection 14/14, independent partial results 14/14, unidentified-test
clarification 14/14, and fact-only medication/treatment updates 28/28.

The raw run recorded 2,247 passes and 37 failures. Review found that the language
judge sometimes asserted translation fidelity without a source, or ignored a
scenario's target-language override. The harness now applies fidelity only when a
source exists and groups review batches by their actual target language. All 56
instances of the four language-override scenarios were reviewed again from saved
outputs, including previously passing cases. Eight false failures were removed and
three additional language flags were retained. No application turns or functional
results were rerun or overwritten.

The reviewed report records **2,252 passes and 32 failures**: 27 functional checks,
one Finnish speech/intent check, two wrong-language replies, and two Portuguese
regional-language flags requiring native review. Functional failures cover missing
clinical clarification and dependent context-reuse checks (15), medication
preservation/response checks (6), recall routed through the memory-write path despite
correct answers and unchanged stores (3), and handled assessment/provider errors (3).
These are failed checks, not 32 independent defects. Deployment acceptance is still
not fully passing.

The final offline run, including both evaluator regressions, passes 344/344 checks;
the harness self-tests pass 16/16. The full run contains its original 342 offline
checks. Application source hashes still match the full execution. Only the evaluator
and its regression tests changed afterward. Speech pipelines pass 13/14; the failed
Finnish pipeline misrouted the ASR-transcribed question. These tests use real model,
ASR and TTS services with synthetic participant services and isolated RAM storage.
They do not verify live FHIR ingestion, physical devices or database restart recovery.

Evidence under `output/system_test/four-fixes-20261030/`:

- `release/report.html`: immutable full execution and original judgements.
- `release-reviewed/report.html`: reviewed report; `evaluation-review.json` records
  its scope and every changed judgement, including new failures.
- `offline-final-review/report.html`: final regression results.
- `validation-summary.json`: priority totals, model breakdown and remaining findings.
- `application-changes.patch`: application changes relative to this task's starting
  state, preserving earlier uncommitted work.

## Earlier health/memory notice follow-up — 2026-10-30

The final offline run passes 337 tests. Across GPT-4.1-mini and GPT-5.4-mini in all
seven configured languages, 84/84 memory functionality scenarios pass and 112/112
notice/presentation functionality scenarios pass. These runs use synthetic data,
isolated RAM stores and real model calls.

Language quality is not fully passing: notice review passes 264/266 checks and
memory-response review passes 263/266. GPT-5.4-mini produced one mixed-script Finnish
synthetic-data notice (flagged in two outputs) and one English reply under the
Italian preference. Two GPT-4.1-mini Portuguese regional-style flags need native
review. Supplementary routing checks pass 16/18: fact-only medication/treatment
statements can still be routed to HealthRAG unnecessarily. None of these failures
are removed or overridden to make the reports green.

The final reports are under `output/system_test/localization-fix-20261030/`:
`regression-final`, `memory-final`, `notices-verified`, and `validation-summary.json`.
The notice matrix predates the final profile-only verification adjustment; its
translation/presentation implementation is unchanged. The final offline and memory
runs include that adjustment. Earlier development results remain available.
The historical results below predate this follow-up.

## Reproducible validation

Use the mirror's existing tested environment and its private model/retrieval
configuration. Do not start a second native connector alongside the service.

```bash
python -m unittest discover -s maya/tests -p 'test_*.py' -q
python -m maya.tests.live_response_language_scenarios ro --edges
python -m maya.tests.live_response_language_scenarios pt-PT --medication
python -m maya.tests.live_response_language_runner
python -m maya.tests.live_response_language_review
```

The scenario harness loads actual session JSON through `load_session`, then calls
the real supervisor with real models and existing literature retrieval. It uses
synthetic public-SDK records, synthetic Phyxio history and isolated RAM memory.
Its reports retain ontology selection/assessment, observations, clarification,
rewrite/search context, retrieval, answer/reference metadata, cache decisions and
tool calls. `--label` retains separate rerun reports. Run large model suites
sequentially to avoid exhausting the deployment's shared rate limits.

The runner harness calls the actual main loop with both typed input and generated
audio fed through simulated microphone callbacks, real VAD, Azure Whisper, models,
retrieval and Azure TTS. Speaker playback is intercepted. These are cloud audio
checks, not physical microphone/listening validation. A subset of language codes
can be supplied as positional arguments. The language review is a separate test
model judgement of saved generated text, not native-speaker certification or a
clinical-grounding check.

Optional positional report paths limit language review to a particular run. Keep
older failed reports when repeating model-dependent checks. Medication checks use
Unicode word boundaries: finding `doxorubicin` inside an inflected or translated
word is not evidence that the original name was copied exactly.

All reports, audio and logs stay local in ignored output/private validation
directories. No synthetic facts are written into the real participant's memory
or FHIR records. Native-speaker review remains necessary, including Portuguese
regional wording, medication pronunciation and clinically appropriate phrasing.

## Changed files

| Files | Purpose |
| --- | --- |
| `maya/framework/conversation/response_languages.json` (new) | Single language-name/default/instruction configuration |
| `maya/framework/conversation/personalization.py` | Validate and resolve the preference; export onboarding options/schema |
| `maya/agents/config.jsonl` | Remove conflicting exercise user-language instructions |
| `maya/agents/health_rag/config.json` | Use configured language for clarification, answer and urgent wording |
| `maya/agents/health_rag/agent.py` | Place the shared wording instruction after retrieved evidence in the final prompt |
| `maya/agents/phyxio_exercise_agent/agent.py` | Align user-facing structured-output descriptions |
| `maya/framework/integrations/biomarkers/config.json` | Align biomarker readout language instruction |
| `maya/session.example.json` | Demonstrate the startup preference |
| `README.md`, `docs/RESPONSE_LANGUAGE.md` (new) | Onboarding, boundaries, validation and deployment notes |
| `maya/tests/test_personalization.py` | Configuration, precedence, schema, invalid codes, ASR/voice separation |
| `maya/tests/test_phyxio_offline_integration.py` | Exercise propagation and exact/semantic answer-cache isolation |
| `maya/tests/test_health_rag_clarification.py` | Wording gets the instruction; machine-only assessment does not |
| `maya/tests/test_health_rag_retrieval.py` | Machine-only query rewriting excludes wording instructions |
| `maya/tests/test_runner_voice.py` | Preference preservation after model-request failures |
| `maya/tests/live_health_config_scenarios.py` | Give the existing Spanish wording case an explicit Spanish preference |
| `maya/tests/live_response_language_scenarios.py` (new) | Real supervisor/model/retrieval scenarios with fixture tools and RAM memory |
| `maya/tests/live_response_language_runner.py` (new) | Actual runner with generated microphone audio, real cloud ASR/TTS and intercepted playback |
| `maya/tests/live_response_language_review.py` (new) | Separate model-assisted review of the saved output language |

There are 19 intended files. Existing `maya-demo-main/` and `pgvector/` untracked
trees are unrelated and untouched. Backups and private evidence are outside the
repository at `~/.local/share/maya-validation/response-language-20261021T165543Z`.
Private session, credentials, service settings, participant mappings and partner
ontology files are checked against their original hashes. No commit, push or
dependency upgrade is part of this work.

## Validation on 21 October 2026

The offline regression suite passed **302 tests** using the mirror's actual
Python 3.13.9 environment. This includes cancellation/retry, silence handling,
reference separation, processing/final-speech ordering and no startup speech,
as well as the new preference/schema/cache checks. SDK versions were retained.

The final sequential supervisor run passed **56/56 core scenarios**: eight per
language, covering direct response, health retrieval, clarification with a scoped
fixture lookup, resumed clarification with query rewriting and stored ontology
memory, exercise history, exercise feedback, biomarker readout and urgent guidance.
These checks used real models and the existing literature collection, with fixture
FHIR/Phyxio responses and isolated RAM memory. They are not seven live patient
record checks or a certification of clinical answer grounding.

The additional exact-medication-name check passed **1/7** (English). In all six
other languages the answer localised or inflected `doxorubicin` despite the shared
verbatim-copy instruction. `bisoprolol`, the supplied dose, readings and units were
retained in these cases, and the wording still called for a care-team review.
This is an unresolved requirement for exact source-name preservation, not a fully
passing localisation release. No output-replacement or automatic translation
service was added to conceal it. The first substring-based test was too permissive;
the final evidence was rechecked using Unicode word boundaries.

The extended Finnish run passed **12/12 additional edge scenarios**: mixed input,
omitted preference, unknown short follow-up, empty and unavailable exercise
history, language change with existing memory, other-person context isolation,
participant isolation, repeated-answer cache separation, ambiguous empty and
failed biomarker results, and absent retrieval evidence. Its full result was
20/21 including the eight core cases and the failed medication-name case. Earlier
Romanian/Italian no-evidence failures came from an ineffective test injection;
injecting the empty retriever when building the graph corrected that harness.

Earlier concurrent runs hit model rate limits. A French memory verification timed
out, and an earlier Finnish verification received malformed structured model
output. The final core runs stored the expected memory record, and the extended
run reused it after a language change, but those earlier failures remain evidence
of model-dependent verification instability. Memory verification, recovery and
clinical rules were not changed or claimed universally reliable.

The actual main loop's first complete cloud speech run passed the full checks in
five languages after the spoken turn's route was explicitly checked. Romanian's
spoken conversational turn worked; its next health
request received a provider `InternalServerError`. The runner recovered with the
configured English retry message. Finnish recognition changed the imperative
request into a statement; the supervisor routed it to general memory and returned
the fixed English acknowledgement. This was isolated RAM memory, not the real
participant's records. It is an ASR/intent failure, not evidence of a generated
Finnish answer. The test now checks that the spoken question reaches the direct
answer route, rather than accepting any nonempty response.

Generated microphone audio, real VAD, Azure
Whisper, real supervisor/retrieval and Azure TTS were used; speaker playback was
intercepted. Romanian and Finnish recognition contained minor wording/spacing
differences, so exact transcription is not claimed. Physical room acoustics and
native pronunciation require separate checks.

The participant-scoped live FHIR diagnostic returned an adapter `ValueError` and
no verified usable observations. It does not establish that the account has no
records. All interpretation cases therefore used fixtures. No FHIR writes were
performed.

The existing Spanish health regression passed **1/1**, and existing live-model
exercise regressions passed **13/13** with fixture sessions and simulated actions.
The exercise case with Spanish input and an omitted language preference answered
in English, as the new default contract requires.

| Response language | Direct | Health retrieval | Clarification → answer | Exercise history/feedback | Biomarker readout | Cloud runner | Exact medication name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| English | Pass | Pass | Pass | Pass | Pass | Pass | Pass |
| Romanian | Pass | Pass | Pass | Pass | Pass | Pass on rerun after server error | Fail |
| Italian | Pass | Pass | Pass | Pass | Pass | Pass | Fail |
| Spanish | Pass | Pass | Pass | Pass | Pass | Pass | Fail |
| Finnish | Pass | Pass | Pass | Pass | Pass | Mixed; see below | Fail |
| Portuguese (Portugal) | Pass* | Pass* | Pass* | Pass* | Pass* | Pass* | Fail |
| French | Pass | Pass | Pass | Pass | Pass | Pass | Fail |

The Romanian runner rerun passed **11/11 checks**. Finnish repeated the original
ASR/intent failure, then passed **11/11** with an additional, differently worded
question, `Mikä on Kuun etäisyys Maasta?`. That demonstrates a working Finnish
speech path but does not resolve the original utterance's repeated failure. Both
source utterances, transcripts and routes are retained. Speech preferences were
not changed to obtain this result.

*A separate model-assisted review inspected 100 generated outputs: 98 matched
its language/region check; two Portuguese passages were flagged for potentially
Brazilian phrasing. The instruction explicitly selects Portugal, and sampled
outputs also contain European forms such as `facto`, `cancro` and `registados`.
The regional flags are review judgements, not a native linguistic verdict.
Portugal-specific consistency remains unverified pending native-speaker review.
An initial reviewer falsely inferred Finnish influence from filenames of tests
that intentionally switched to Romanian; anonymising review IDs removed that
bias. Fixed memory acknowledgements are excluded from generated-language review,
while their incorrect invocation remains a failed runner-route check.

The temporary native runner connected to the same Phyxio account and its
read-only exercise-history check returned `ok` with real, non-synthetic records.
Physical cancellation, microphone, playback and reference handling remain
**pending**. The manual check was requested, but no recording or transcription
arrived during the temporary session. The temporary unit was stopped and the
original `maya-supervised.service` restored. Its saved session remains unchanged
and therefore uses the configured English default. No test runners remain active;
cloud TTS success alone is not recorded as a listening pass.

Reports are in ignored `output/response_language_*_final2.json`,
`output/response_language_fi_edges_sequential.json`, the
`output/response_language_runner*.json` files and
`output/response_language_review.json`. Earlier failures and pre-correction
reports are retained in the private validation directory.
