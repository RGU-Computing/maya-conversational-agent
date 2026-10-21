# Tested mirror profile

These files preserve the mirror's reviewed, non-secret settings as of 21 October
2026. The participant explicitly authorized including the identifiers in Git.
They contain no credentials or observations. The active deployment remains in its
existing private directory; committing these copies does not reconfigure it.

- `session.json` preserves the local MAYA identity `participant-001`, supportive
  personality, Azure voice and automatic ASR language selection.
- `biomarkers.json` selects the read-only FHIR provider and maps that identity to
  `pedram-rgu`, the Phyxio login username/FHIR Patient ID confirmed by Oscar. Only
  the credential path differs from the active file: `client-rgu.json` is resolved
  beside this configuration, so the profile is portable. Resource types and
  expected codes match the active configuration.
- `exercise.json` preserves the Phyxio history provider, accepted statuses,
  pagination and metric settings.
- `runtime.env.example` preserves the selected runtime settings with configurable
  file paths. It contains shell expansions; do not pass it directly to systemd's
  `EnvironmentFile`. An installed unit needs explicit absolute paths.

This is a profile for one confirmed account, not a default for other users or a
designated clinical test account. Keep the same Phyxio browser account when using
it. On another participant's deployment, replace the session and mapping together
using their confirmed identity. Fictional scenarios belong in the isolated test
harnesses, never this participant's persistent records.

To reproduce this setup, copy the three JSON files to a private directory owned
by the service account (directory mode `700`, files `600`). Supply the authorized
`client-rgu.json` separately in that directory, keeping its token outside Git.
For a `token_file` credential, supply that file separately too; relative token
paths resolve against the credential JSON's directory. Preserve HTTPS and
certificate verification. Model, speech and database credentials remain in the
local `.env` or the service's private environment. The exact supplied libfhir
26.10.16 wheel and native Phyxio/ICE dependencies are also required; see
[the mirror handover](../../docs/MIRROR_HANDOVER.md).

From the repository root, with an absolute `MAYA_CONFIG_DIR` pointing to the
private copy and `MAYA_PYTHON` pointing to the tested environment's Python:

```bash
set -a
. deployment/mirror/runtime.env.example
set +a
"$MAYA_PYTHON" -m maya.cli.check_tools --session-config "$MAYA_SESSION_CONFIG"
"$MAYA_PYTHON" -m maya.framework.supervisor.run_with_vad --session-config "$MAYA_SESSION_CONFIG"
```

Only run one native runner at a time. These settings use normal participant
memory. For synthetic supervisor/main-runner tests, use the documented harnesses
that substitute service responses and isolate memory.

The confirmed mapping passes configuration checks, but live observation queries
returned ambiguous empty SDK results; usable observations are not yet verified.
A separately authorized write/read diagnostic was rejected by the proxy. The
MAYA adapter remains read-only. The handover records these limitations and the
earlier incomplete model-based test results.
