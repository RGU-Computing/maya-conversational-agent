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
| Biomarkers | Read-only adapter using the documented libfhir `get()` and `dump()` methods; shared by supervisor and HealthRAG. | Oscar's credentials plus the participant's Phyxio username as FHIR Patient ID. See the SDK limitations below before deployment. |

Oscar has confirmed that a FHIR Patient ID is the participant's **Phyxio username**.
Automatic retrieval of that authenticated username from PhyxioConnect is awaiting
UCLM's library update. A local MAYA identity can differ from the username; its private
`patient_map` must explicitly associate them. Do not substitute an unverified candidate.

The supplied Phyxio API returns the browser's current account history without a
participant ID. It cannot independently verify that the browser login matches
MAYA's startup payload. A browser cookie or `/enduser/3/` URL is not a confirmed
FHIR Patient ID. Do not guess this mapping.

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

Validation follow-up on 20 October 2026: the public-SDK correction passes 277
offline tests and the 14 real-model exercise scenarios. The focused FHIR tests
confirm that quantities without units are rejected and that HealthRAG can ask for
the measurement, use the user's value and unit in retrieval, and answer without
repeating the fetch. Synthetic FHIR responses, RAM memory and simulated audio are
not live participant or mirror validation.
The final full FHIR/supervisor/runner run passed 9 of 10 scenarios: one assessment
returned the safe error response instead of clarifying. The focused scenario
subsequently passed. Preserve both outcomes; the full run was not a clean pass.

**Delivery blockers found during broader checks:** normal treatment-memory recall
and participant isolation passed, but the fault-injection recovery scenario did
not consistently pass. Additional model timeouts produced a pending response;
one run failed to retain an additional condition assertion. A traced run later
stored both facts, which does not explain or resolve the earlier omission.
One weight follow-up also described a value as healthy while acknowledging that
the retrieved literature supplied no relevant threshold. These memory and answer
grounding findings remain unresolved; do not describe the full system as validated
for delivery. Diagnostic artifacts are kept locally under
`output/fhir_public_api_validation/`, outside Git. The partner ontology was not changed.

## Before updating the mirror

1. Include the new source, examples, tests and this guide in the reviewed commit.
   `git commit -a` alone omits new untracked files. Never commit `.env`, credentials,
   participant mappings, local session JSON or generated output.
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

Keep deployment files outside Git, readable by the service account. An example
layout is a private `maya` directory belonging to that account:

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
  must exactly match `session.json`. The right side is the participant's actual
  Phyxio username, without the `Patient/` prefix; no default exists.
- The real FHIR example exposes blood pressure, heart rate and body weight.
  Glucose is not supported by the supplied library's current catalogue; do not
  copy the dummy glucose configuration into the real provider.

An explicit `--session-config` overrides `MAYA_SESSION_CONFIG`. Process environment
variables override `.env`; a service's old environment may otherwise select dummy
providers even after `.env` is edited. Changing these files does not switch a
running participant session. Stop on logout; launch with the next login payload.

## FHIR public API and current limits

The adapter follows the supplied **LibFHIR Usage Guide, 26.10.16**, pages 3 and 7:
initialize `FHIRProxyClient`, call `client.get(resource_type, subject=patient_id)`,
and read each `record.dump()`. It does not construct HTTP requests, access private
client attributes, inspect the SDK's internal catalogue or modify the partner library.
The previous private-transport workaround was removed on 20 October 2026.

The three enabled measurement types are a deployment choice, not the full library
catalogue. Additional documented Observation types can be enabled through `catalog`
and `fhir.resource_types`; Python contains no list of allowed measurement codes.
The tool intentionally reads latest measurements, as required by MAYA. Date-range
searches, record-ID searches and writes are not part of this conversational tool.
The SDK supports those operations separately; they are not required to use its read API.

Both documented authentication forms are supported in the private credential JSON:
`url` plus `token`, or `url`, `token: null` and `token_file`. A relative token-file
path is resolved against the credential file's directory. HTTPS and certificate
verification remain application requirements.

**Migration:** remove `fhir.timeout_seconds` from existing private biomarker
configuration files. The public SDK offers no request-timeout parameter. Keeping
that setting now produces a configuration error rather than silently ignoring it.
There is currently no application-enforced FHIR request deadline; a stalled request
can hold up its conversational turn. UCLM needs to provide a supported timeout
mechanism before bounded response time can be claimed.

Known behaviour of the supplied 26.10.16 SDK, reproduced locally:

- `record.dump()` drops scalar quantity units. Compound components preserve theirs.
  MAYA requires a supplied unit or unit code for numeric FHIR quantities, including
  components. It rejects an observation that fails this check and reports the lookup
  as unavailable; a reference range or a conventional unit cannot fill the gap.
  Scalar readings such as weight and heart rate therefore remain unusable through
  this SDK version until UCLM preserves their units. Blood-pressure components with
  units work. Prompt-only warnings did not reliably prevent an assumed unit in testing,
  so invalid quantities are withheld from both readout and HealthRAG.
- The documented whole-second UTC `Z` timestamps work. Other timestamp forms can
  fail parsing. The SDK can omit records that fail parsing, so this integration
  cannot independently prove that returned history is complete. “Latest” means
  latest eligible among the records the SDK returned.
- `get()` can return the same empty list for no records and some network, protocol
  or parsing errors. An entirely ambiguous empty lookup is reported as unavailable,
  never as proof that the participant has no measurements. If another requested
  measurement returned records, those can be shown and the unresolved measurements
  are listed as not returned. An explicit SDK failure (`None`) fails the lookup.
- Numeric HTTP status and raw FHIR fields discarded by the SDK are not available
  through the public result. Diagnostics must not claim those details were verified.

MAYA validates participant ownership and quantity units, applies configured accepted statuses,
and selects latest timestamps locally while preserving ties. For an interval it
uses the period end (or start if absent) for ordering. Its internal Observation
envelope contains all SDK-serialized fields: identity, status and reference ranges
are represented separately; remaining fields go in `result`. This is application
translation of `dump()`, not a replacement SDK parser.

These limitations must be reported to UCLM; using the public API alone does not
establish production readiness. Tests of synthetic HTTP responses do not establish
real participant access or mirror audio/ICE operation. Run
`python -m maya.tests.live_fhir_scenarios` for supervisor and main-runner scenarios
using real models/retrieval, the actual SDK parser, synthetic FHIR HTTP and RAM memory.
Actual microphone and speech playback are excluded from this Mac test.

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
records were found. Both exit 0; `configured` is only possible without `--fetch`.
A failed/unavailable check exits 1. No synthetic fallback is used. The public FHIR
SDK's ambiguous empty results fail the check; see the limitations above.
A successful FHIR check still requires a verified username binding and confirmation
that the returned records are sufficient for the intended use.

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
| FHIR `patient_map` missing | Map the startup MAYA ID to the participant's confirmed Phyxio username. Credentials alone do not identify the patient. |
| FHIR lookup unavailable | The public SDK does not reliably distinguish empty data, access failure and parsing failure. Inspect private SDK logs for UCLM support; do not share tokens or raw records. |
| FHIR `ValueError` after configuration passed | The SDK returned no verifiable observations, failed, or supplied a record that failed validation. Do not infer that the patient has no records. |
| Configuration error at `fhir.timeout_seconds` | Remove the obsolete setting. The public SDK has no supported request-timeout option; see the migration and delivery limitation above. |
| `no_data` | Confirm this account really has records and the correct patient mapping/statuses. Empty results are distinct from connection failure. |
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
