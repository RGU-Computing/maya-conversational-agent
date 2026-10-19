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
| Biomarkers | Real read-only libfhir adapter, preserving measurement units, dates and provenance; shared by supervisor and HealthRAG. | Oscar's credential JSON plus a confirmed mapping from MAYA user ID to FHIR Patient ID. |

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
  must exactly match `session.json`. The right side is the Patient ID without the
  `Patient/` prefix. UCLM must confirm the relationship; no default exists.
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
records were found. Both exit 0; `configured` is only possible without `--fetch`.
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
| FHIR `HTTPError` | The diagnostic includes `http_status`. For 401/403, check credentials/permissions with UCLM; do not paste the token into logs or chat. |
| FHIR `ValueError` after configuration passed | The proxy response or returned resource failed validation. Confirm resource types, patient reference, timestamps and response contract with UCLM. |
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
