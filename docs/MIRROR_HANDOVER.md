# Mirror deployment: participant identity and live tools

This file is the handover for running and diagnosing MAYA without the development
conversation. Run commands from the repository root, using the same account and
Python environment as the deployed service. The supported runner is
`maya.framework.supervisor.run_with_vad`.

Package ownership and moved command/configuration paths are documented in
[the codebase structure guide](ARCHITECTURE.md).

## What is implemented and what still needs confirmation

| Component | Implemented | Live deployment requirement |
| --- | --- | --- |
| Participant identity | One validated startup `SessionConfig` supplies the supervisor, memory and tools. Each launch has a participant-scoped conversation thread. | UCLM supplies the authenticated participant ID and restarts MAYA when the login changes. |
| Exercise history | Real `getSessionList(page)` adapter; page 0; latest record and comparisons; both history tools validate the runtime participant ID. | UCLM connector and ICE installed, exactly one authenticated Phyxio browser connected. |
| Biomarkers | Read-only libfhir public-API adapter; validates units, dates, identity and provenance; shared by supervisor and HealthRAG. | Oscar's credential JSON plus a confirmed mapping from MAYA user ID to FHIR Patient ID. |

The supplied Phyxio API returns the browser's current account history without a
participant ID. It cannot independently verify that the browser login matches
MAYA's startup payload. A browser cookie or `/enduser/3/` URL is not a confirmed
FHIR Patient ID. Oscar confirms that the Patient ID equals the Phyxio login username.
On 21 October, the user confirmed Oscar's mapping for this mirror:
`participant-001` → `pedram-rgu`. The local MAYA identity is unchanged. The reviewed
[mirror profile](../deployment/mirror/README.md) preserves this mapping and the
non-secret session/provider settings. Do not apply it to another account.

Mac validation uses synthetic service responses. It proves routing, identity
guards and adapter behaviour, not a live Linux connection or real patient access.
The Mac's local `.env` selects dummy providers; it is not a mirror configuration.

Validation on 19 October 2026: 262 offline tests, 14 real-model exercise scenarios
and seven real-model FHIR/HealthRAG scenarios passed. The runner check covered
native connector callbacks, microphone failure/retry, supervisor routing, exercise
retrieval and the TTS call using simulated transport/audio. A credentialed request
to UCLM's real device-list operation succeeded; patient observations remain
untested until the confirmed Patient ID mapping is supplied. Ontology files were
unchanged. These are development results; repeat the live checks on the mirror.

## Before updating the mirror

1. Include the new source, examples, tests and this guide in the reviewed commit.
   `git commit -a` alone omits new untracked files. Inspect configuration contents:
   explicitly authorized identifiers and application settings may be committed,
   including participant mappings and session JSON. Never commit `.env`, actual
   credentials, private keys, recordings, memory databases or health-record logs.
   Keep those local files intact.
2. On the mirror, record the currently deployed commit with `git rev-parse HEAD`
   and retain the existing environment and service configuration for rollback.
3. Stop the current MAYA process/service. Use Ctrl+C for a terminal process;
   Ctrl+Z only suspends it and can leave the port and microphone occupied.
4. Pull the reviewed commit and run the checks below before enabling the service.

To find the actual service unit (do not assume its name):

```bash
systemctl list-units --type=service --all
systemctl cat <actual-MAYA-service-unit>
```

Check its `User`, `WorkingDirectory`, `EnvironmentFile` and `ExecStart`. It must run
the updated checkout/environment and the supported module. A service using a
different checkout will not see a pull into your personal directory.

## Python and partner dependencies

The repository lock currently requires Python 3.13. The older mirror log showed
Python 3.12; use the new environment explicitly and update the service's interpreter
path accordingly. Do not delete the working environment before validation.

```bash
uv sync --locked --inexact --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python /path/to/libfhir-26.10.16-py3-none-any.whl
uv pip check --python .venv/bin/python
```

UCLM's `libphyxio-connect` Debian package must already provide its wheel under
`/usr/share/libphyxio-connect/dist/`, Slice files under `/usr/share/slice/phyxio/`,
and the connector certificates. `requirements.txt` selects connector 26.5.29 and
ICE 3.7.11 or a later 3.7 patch. ICE 3.8 changes the API used by the supplied
connector ([ZeroC upgrade guide](https://docs.zeroc.com/ice/3.8/upgrade-guide)).
`--inexact` preserves these separately installed partner packages.

On Linux, ICE may need to compile from source; if that fails, keep the full build
error and use UCLM's supported build for this Python environment. Do not substitute
ICE 3.8 to bypass an installation failure. Linux build/transport verification is
still required even though the 3.7.11 initialization API was checked on the Mac.

## Configuration: one participant input, two provider selections

Keep active deployment files in a private directory readable by the service
account. Reviewed, non-secret profiles can be versioned separately, as in
[`deployment/mirror`](../deployment/mirror/README.md); credentials and runtime
participant data stay outside Git. An example private layout is:

```text
session.json          # supplied by UCLM at login
exercise.json         # copy of session_config.phyxio.example.json
biomarkers.json       # copy of config.fhir.example.json, completed for this deployment
client-rgu.json       # Oscar's decrypted credential file
```

Use directory permissions `700` and private file permissions `600`. Transfer
credentials securely; the mirror needs the decrypted configuration, not your GPG
private key or passphrase. Keep the existing model, speech and database credentials.

UCLM supplies `session.json`, for example (replace the placeholder):

```json
{
  "user_id": "UCLM-CONFIRMED-PARTICIPANT-ID",
  "preferences": {"personality_id": "supportive"}
}
```

The ID must be stable for that participant. It is not selected by the LLM. Optional
`tts_voice` and `asr_language` preferences are documented in the root README.
There is no separate exercise participant ID to maintain.

Select these files using absolute Linux paths in the service environment or `.env`:

```dotenv
MAYA_SESSION_CONFIG=/absolute/private/path/session.json
MAYA_EXERCISE_SESSION_CONFIG=/absolute/private/path/exercise.json
MAYA_BIOMARKER_CONFIG=/absolute/private/path/biomarkers.json
MAYA_ENABLE_PHYXIO_IO=1
```

- `exercise.json`: copy
  `maya/agents/phyxio_exercise_agent/session_config.phyxio.example.json`.
  Its provider must be `phyxio`, not `dummy` or `disabled`.
- `biomarkers.json`: copy `maya/framework/integrations/biomarkers/config.fhir.example.json`.
  Set `fhir.client_config_path` to the credential file, and set `patient_map` to
  `{"UCLM-CONFIRMED-PARTICIPANT-ID": "CONFIRMED-FHIR-PATIENT-ID"}`. The left side
  must exactly match `session.json`. The right side is the confirmed Phyxio login
  username (FHIR Patient ID), without the `Patient/` prefix. No default exists.
  Keep `fhir.observation_codes` aligned with the configured resource types, using
  the supplied SDK documentation. Credentials support inline `token` or `token_file`;
  relative token paths are anchored to the credential JSON directory.
- The real FHIR example exposes blood pressure, heart rate and body weight.
  Glucose is not supported by the supplied library's current catalogue; do not
  copy the dummy glucose configuration into the real provider.

An explicit `--session-config` overrides `MAYA_SESSION_CONFIG`. Process environment
variables override `.env`; a service's old environment may otherwise select dummy
providers even after `.env` is edited. Changing these files does not switch a
running participant session. Stop on logout; launch with the next login payload.

## Diagnose configuration before starting conversation

```bash
.venv/bin/python -m maya.cli.check_tools --session-config /absolute/private/path/session.json
```

This prints the Python interpreter, selected configuration paths, effective MAYA
user ID, selected providers and failures. It checks
the native imports, speech configuration and FHIR mapping without making model
calls or fetching records. It deliberately fails for dummy/disabled providers.
`configured` means settings passed, not that live data was retrieved.

FHIR can be checked independently while exercise setup is in progress:

```bash
.venv/bin/python -m maya.cli.check_tools --session-config /absolute/private/path/session.json --only biomarkers --fetch
```

For exercise access, stop MAYA first so it releases the connector port. Log in to
the matching participant on the mirror. Start this check, then reload that Phyxio
page during the connection wait if necessary. Do not start exercise/voice input
while running the diagnostic:

```bash
.venv/bin/python -m maya.cli.check_tools --session-config /absolute/private/path/session.json --only exercises --fetch --wait-seconds 60
```

The checker uses the real adapters and the same session identity. It prints record
counts and status, not health values, patient IDs, raw records or credentials. It
does not invoke conversational models or write conversation memory. The exercise
check starts/stops the existing connector; it does not start an exercise. Keep
the resulting report local because it contains the MAYA participant ID.

`ok` means records were read; `no_data` means the request completed but no eligible
records were found. For FHIR, an empty SDK response is ambiguous and fails the check;
`no_data` is possible only if returned records all have excluded statuses. Both exit 0; `configured` is only possible without `--fetch`.
A failed/unavailable check exits 1. No synthetic fallback is used. A successful
FHIR check still requires UCLM to confirm that the configured Patient belongs to
the participant, and that the proxy returns complete relevant history.

## Validate conversation on the mirror

Run regression tests first (FHIR tests require the partner wheel; check for skips):

```bash
LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false .venv/bin/python -m unittest discover -s maya/tests -p 'test_*.py' -q
.venv/bin/python -m maya.framework.supervisor.run_with_vad --session-config /absolute/private/path/session.json
```

Use typed input first, then repeat through the mirror microphone:

| Request/check | Expected result |
| --- | --- |
| “Review my latest recorded Phyxio exercise session and compare it with earlier sessions.” | Routes to exercise agent; latest native date and metrics; up to five comparable earlier sessions. Confirm against the browser's page-0 history. Partial/skipped routines must not be described as fully completed. |
| “List my recorded Phyxio exercise sessions.” | History listing through the same participant-checked connection. |
| “How did my friend perform?” | No substitution of your exercise records. |
| “Show my latest blood pressure measurement and its date, without interpreting it.” | Supervisor biomarker readout; values/date match the real FHIR source, or an explicit missing/unavailable response. |
| “What does my latest blood pressure mean considering my cancer treatment?” | HealthRAG can fetch the observation, ask for relevant missing treatment context, then retrieve evidence. It must not invent missing clinical information. |
| Log out, stop MAYA, log in as a second participant, relaunch with their payload. | New conversation thread; the new participant's memory, exercise login and confirmed FHIR mapping. No records substituted from the first participant. |
| Cancel voice capture; then try again. | Capture cleans up and a fresh request works. Final speech must not overlap a waiting acknowledgement or read reference entries aloud. |

Only enable the deployed service after these checks. Ensure its `ExecStart` uses
the exact tested interpreter, module and session payload. Until UCLM connects
login/logout to process lifecycle, keep testing supervised with one participant.

## Troubleshooting without this conversation

| Result | Check next |
| --- | --- |
| Provider is `dummy` or `disabled` | Correct the two provider environment paths; confirm the service has no stale overrides. |
| Invalid/missing session | Check the selected file and required non-empty `user_id`. No fallback identity is used. |
| Missing `Ice`, `phyxio` or `libfhir` | Install the supplied dependencies into the interpreter actually used by the service. An OS package alone may not be visible in a virtual environment. |
| Slice-file or certificate error | Verify UCLM's Debian package/system files, permissions and connector version. Do not enable the unused ICE compatibility shim. |
| No Phyxio client connected | Check browser login/page reload, certificate trust and connector logs. Only one client should connect. |
| Port 1560 busy / another runner active | Stop the old service/process, including suspended ones. Do not delete a lock file to bypass a running process. `ss -ltnp` identifies listeners on Linux. |
| FHIR `patient_map` missing | Obtain the confirmed MAYA-to-FHIR relationship from UCLM; credentials alone are insufficient. |
| FHIR unavailable / `ValueError` | The public SDK hides some HTTP/network errors and parsing failures. The adapter cannot reliably distinguish those from an empty search; ask UCLM to investigate privately. Do not paste tokens or health records into logs/chat. |
| FHIR `ValueError` after configuration passed | The proxy response or returned resource failed validation. Confirm resource types, patient reference, timestamps and response contract with UCLM. |
| `no_data` | No eligible returned records; check mapping/statuses. An empty FHIR SDK response instead becomes unavailable because it cannot establish absence of records. |
| Exercise `unavailable` | Check connection, pagination bounds, timeout and native response shape. `phyxio.max_pages` limits traversal; increase it only for confirmed longer histories. |
| Microphone “Device unavailable” | Check service-account audio access, other microphone users and `MAYA_MIC_DEVICE`. List devices with `.venv/bin/python -m sounddevice`. |
| HealthRAG fails after the tool works | Check model credentials and PostgreSQL/pgvector configuration. Tool access and retrieval are separate. Do not rebuild the literature index as a generic repair. |

Service logs: `journalctl -u <actual-MAYA-service-unit> -n 100 --no-pager`.
For a support handover, record the commit, Python version, provider selections,
diagnostic status and failing step. Review logs for participant data before
sharing; do not share `.env`, credential JSON, browser cookies or raw health records.

If the update fails, stop the new process and restore the recorded working
checkout/environment/service configuration. Preserve participant storage and
deployment files. Diagnose with the commands above before retrying.

## Public FHIR API correction on this mirror — 20 October 2026

The supplied `~/Downloads/libfhir-usage.pdf` and installed SDK 26.10.16 were read.
The adapter now uses only `libfhir.client.FHIRProxyClient`, scoped `get()` and
`record.dump()`. Partner library and ontology files remain unchanged. Resource
names and expected measurement codes live in configuration. All SDK-returned
fields are retained; numeric quantities without verifiable units are rejected.
The active private configuration was backed up outside Git before removing the
obsolete timeout setting and adding the documented coding pairs. At this stage
its patient mapping was empty and the existing local session identity unchanged.

The SDK drops scalar quantity units, so ordinary scalar weight/heart-rate records
cannot safely be read out with this version. It may omit records with timestamp
offsets/fractions or unsupported fields; even a successful result is only the latest
accepted record it returned, with ties preserved. Empty responses do not establish
that a patient has no records. No bounded network timeout is available through the
public API. These are partner-library limitations; MAYA does not bypass them with
private HTTP or monkey-patches. See the README for the complete configuration and
limitations. Ask UCLM for lossless quantities, visible error/parse outcomes and a
supported timeout option in a future SDK.

At this stage, live reads awaited the participant's confirmed Phyxio login
username. No participant-scoped live FHIR call was attempted on 20 October. Do not
enumerate patients or infer their identities from cookies, URLs or local usernames.
The confirmed mapping and later live results are recorded below.

Initial validation for this correction (20 October 2026, before the extended audio checks below):

- 285 offline tests passed with no skips; dependency check passed (108 packages).
  Installed-SDK tests reproduced scalar-unit loss, ambiguous empty/error outcomes,
  timestamp omissions and no transport timeout. These used synthetic HTTP fixtures.
- Real-model FHIR suite: **12/13**, not a complete pass. Readout, treatment
  clarification/resume, ontology-guided HealthRAG lookup, follow-up query rewriting,
  retrieval, empty evidence, missing units, failures and identity isolation passed.
  `missing_observation` failed in the full run and the focused rerun after a generic
  biomarker-context clarification instruction was added. The model treated a
  requested/unmapped lookup as resolving the measurement, returned no graph
  interrupt, and asked for the values in its final answer with misleading wording
  about records it could access. No other participant's records were queried.
  Fixing this reliably requires further HealthRAG clarification work; neither
  supervisor code nor partner clinical/ontology rules were changed in this task.
- Exercise regression: **14/14** using real models and simulated Phyxio responses.
- Memory conversation/recovery: **7/7** with real models and isolated RAM memory.
  The separate rewrite-fault scenario initially failed its timeout branch because
  a malformed model verification response left memory pending before retrieval;
  the empty-rewrite branch recovered and passed. This remains evidence of model
  variability. A focused rerun passed both timeout and empty-rewrite branches;
  it does not erase the initial verification failure.
- The actual `run_with_vad.main()` loop passed **9/9 checks**: typed readout,
  clarification/resume, retrieval, ambiguous-empty handling, scoped SDK calls,
  speech-output boundary and exit/lock cleanup. Service replies and audio were
  simulated; models/retrieval were real and memory was in RAM. Initial harness
  attempts exposed the pending-memory guard and a report-path bug caused by the
  runner changing directory; the test now waits for verification and uses an
  absolute report path. No production runner changes were needed.
- Answer grounding is not proven by references alone. In the initial rewrite-fault
  run, an answer inferred no specific bisoprolol contraindication from retrieved
  excerpts discussing other medications; that inference was not established by
  the displayed evidence. No clinical guidance or rules were altered to mask it.
- Actual private configuration loads under the deployed interpreter; the diagnostic
  stops at the missing confirmed mapping before any live patient query. No live
  FHIR observations, native audio or physical exercise were validated in this task.
  The earlier Phyxio calibration-loading blocker was not retested or repaired.

The detailed local report is `output/fhir_public_api_validation.json`; synthetic
transcripts and private backups are outside Git under
`~/.local/share/maya-validation/fhir-public-api-20261020-wl7gdogg/`.
The previous mirror source, partner SDK/ontology files, credentials, session and
exercise settings were preserved. No Git transfer or commit was performed.

## Extended supervisor and native audio checks — 20 October 2026

This later pass used the mirror's actual Python 3.13.9 environment and unmodified
libfhir 26.10.16. All ten installed SDK Python files match the supplied wheel.
The adapter uses the documented public client, scoped `get()` and `dump()`;
credentials, resource types, expected codes, participant mappings and accepted
statuses remain configuration. Fixed SDK-version, FHIR-field and HTTPS validation
constants are protocol constraints, not deployment identities or clinical rules.
Synthetic test fixtures contain fixed example identities and values intentionally.

New defects found and corrected:

- Malformed quantity values could bypass the unit guard. Numeric strings,
  booleans, missing values and invalid quantity shapes now fail validation.
- A readout selected one of two conflicting records with the same timestamp.
  The configurable readout instruction now requires both source/status-qualified
  results; the expanded scenario passed after this change.
- During the physical clarification test, Whisper misrecognized a benign reply
  and Azure rejected the resulting text. An unhandled model API error terminated
  the runner. The runner now gives the configurable `request_unavailable_text`,
  keeps the same participant and preferences, and accepts the next request in a
  fresh thread without automatically replaying the rejected message. Provider
  rejection, connection failure, timeout and rate-limit recovery are tested with
  injected failures. Immediate context in the failed thread is not resumed.
- A silent microphone attempt admitted a low-level noise burst; Whisper produced
  Japanese text and the agent answered in Japanese. Phyxio capture now combines
  WebRTC voice activity with the energy check and a configurable minimum detected
  speech duration. No language forcing, phrase blacklist or partner-library patch
  was introduced. The user repeated silence followed by a greeting and confirmed
  both behaved correctly after the fix.
  Inspection then found that reaching the normal recording deadline could be
  mislabelled as a device failure when the last callback did not align with it.
  This now ends normally when audio was received; an inactive stream or no audio
  delivery still produces a recoverable failure.
  The final physical rerun captured about 19.9 seconds of silence, detected zero
  speech, made no transcription request and produced no device error. The next
  spoken greeting and audible reply passed, confirmed by the user.

Latest automated results:

| Check | Result | Real and simulated boundaries |
| --- | --- | --- |
| Offline suite | 294 passed, no skips | Synthetic transports, model failures and audio callbacks; includes installed-SDK characterization |
| Expanded FHIR supervisor suite | 16/17 | Real models and retrieval; synthetic SDK replies and isolated RAM memory |
| Exercise regression | 14/14 | Real models; synthetic Phyxio responses and isolated RAM memory |
| Main runner harness | 9/9 checks | Actual main loop and real models/retrieval; synthetic SDK and audio boundaries |
| Speech/agent round trips | 6/6 | Real Azure TTS-generated input, Whisper, supervisor, retrieval and output synthesis; no physical playback; isolated validation database |
| Voice-activity gate | 5/5 | Real detector, synthetic silence/click and three Azure-generated utterances; simulated microphone callbacks |

The FHIR failure remains `missing_observation`: an unmapped interpretation request
sometimes produces a final answer asking for values instead of the resumable
clarification interrupt. Its wording also implies access to records that were not
queried. A preceding 13-case run passed, so this is inconsistent model behaviour,
not a complete pass. The initial four added edge cases passed 3/4 before the tied
readout instruction was corrected; their results are retained. The earlier memory
verification/rewrite and answer-grounding issues above remain unresolved by this
work; references alone do not establish that an answer is supported.

The physical runner used the real Phyxio connection, microphone, Whisper, models
and Azure speaker output. Conversation memory was directed to a dedicated
validation database before initialization, while literature retrieval retained its
existing index. No production function or SDK method was replaced in that run.
The user confirmed audible greeting, avatar cancel/retry, truthful unlinked-account
FHIR readout, unknown-test clarification followed by a fresh request, and the
post-fix silence/retry check. The user also confirmed the audible read-only exercise
history matched the logged-in account. A separate real `getRoutineList` read
confirmed that the spoken `(DEBUG)` labels are present in Phyxio's exercise names;
MAYA did not add them and no source names were changed. The greeting described reading the transcribed
message; input modality is not conveyed to the agent, so voice-specific wording
remains imperfect. Recognition errors and acoustic variability are still possible.

No live patient observations were fetched during this pass. Scoped local/browser inspection found
no saved Phyxio login username; numeric browser URLs and the local MAYA alias are
not identity evidence. The user confirmed the visible full name is a profile/display
name, not an explicitly labelled login username. The mapping was still unconfirmed. Scalar-unit loss, timestamp
omissions and unbounded SDK network requests remain partner-library limitations.
The earlier physical exercise calibration-loading problem was not repaired by
these read-only checks.

Source backups and private test artifacts for this pass are outside Git under
`~/.local/share/maya-validation/fhir-e2e-20261020-wh3aaoa_/`.
The consolidated local report is `output/fhir_e2e_validation.json`. Credentials,
session identity/preferences, exercise configuration, clinical rules and partner
ontology models were preserved. The only changes outside the FHIR adapter,
configuration, tests and documentation address the observed runner crash and
native speech false trigger. No Git transfer or commit was performed.
All test runners were stopped afterward. At the end of this pass,
`maya-supervised.service` was inactive with no main process and connector port
1560 had no listener. The service was restarted after the mapping update below.

## Confirmed mapping and live FHIR checks — 21 October 2026

Oscar confirmed the Phyxio login username/FHIR Patient ID `pedram-rgu`. The active
session continues to use `participant-001`; only that user's `patient_map` entry
was updated after a private backup. The provider remains `fhir`. The committed
[mirror profile](../deployment/mirror/README.md) includes the reviewed session,
mapping, resource/code configuration, exercise settings and runtime selections,
with a relative credential path instead of this machine's private absolute path.

The deployed Python 3.13.9 environment with libfhir 26.10.16 passed the mapping and
configuration checks. Participant-scoped public SDK reads for blood pressure,
heart rate and body weight all returned empty lists. MAYA reported unavailable;
there are no verified usable observations, and these ambiguous responses do not
prove that the account has no records. No patient enumeration was performed.

A later, explicitly authorized one-time write/read diagnostic used the documented
SDK API with a genuine user-supplied measurement. The proxy rejected the request
with `not allowed request`; creation was not confirmed. The following scoped read
was still empty. The request was not retried. UCLM needs to clarify the proxy's
authorization/response behaviour; this does not establish a particular token role.
The diagnostic payload and logs remain private, and the application adapter
continues to be read-only.

`maya-supervised.service` was restarted with the updated private mapping and
connected successfully. Its interpreter is
`~/.local/share/maya-validation/mirror-20261019T184537Z/venv/bin/python`;
the corresponding `runtime.env` selects the private session/provider files.
No code or SDK changes were needed for the mapping or one-time diagnostic.
The model/grounding inconsistencies, SDK limitations and exercise calibration
blocker documented above remain open.

The commit review on 21 October reran all **294 offline tests** successfully with
no skips and checked all **108 installed packages** for dependency compatibility.
Both the active configuration and a private copy of the portable profile passed
the configuration-only diagnostic. Session, exercise and runtime settings match;
the portable FHIR file differs only in its credential path. The reviewed staged
tree and existing unpushed commit were scanned for known local credentials and
token/private-key patterns, with no matches. Active private files remained
byte-for-byte unchanged and the service stayed running. Live models, audio and
patient reads were not rerun for this packaging review; their earlier outcomes
and unresolved issues above still apply.

To run the normal mirror session in a terminal (this uses normal participant
memory, not the isolated test database):

```bash
cd /home/rgu/Documents/MAYA
MAYA_CHECK_DIR="$(cat output/validation_location.txt)"
set -a
. "$MAYA_CHECK_DIR/runtime.env"
set +a
"$MAYA_CHECK_DIR/venv/bin/python" -m maya.framework.supervisor.run_with_vad --session-config "$MAYA_CHECK_DIR/session.json"
```

Use Ctrl+C to stop this foreground runner. The user service is currently active;
stop it with `systemctl --user stop maya-supervised.service` before starting a
foreground runner. Use `systemctl --user start maya-supervised.service` to return
to the service later. Do not run both simultaneously. The FHIR mapping is now
confirmed, but usable live observations remain unverified.
