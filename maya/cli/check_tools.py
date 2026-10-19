"""Check mirror tool configuration and optionally read live records, without an LLM.

Run from the repository root: python -m maya.cli.check_tools --help
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from maya.framework.conversation.personalization import load_session


def check_biomarkers(session, fetch: bool) -> dict:
    from maya.framework.integrations.biomarkers import load_biomarker_service

    service = load_biomarker_service()
    result = {"provider": service.config.provider, "status": "failed"}
    if service.config.provider != "fhir":
        return {**result, "detail": "Select a real FHIR configuration with MAYA_BIOMARKER_CONFIG."}
    patient_id = service.config.patient_map.get(session.user_id)
    if not patient_id:
        return {**result, "detail": "Add this user_id and its confirmed FHIR Patient ID to patient_map."}
    if not service.config.catalog:
        return {**result, "detail": "The configured observation catalog is empty."}
    if not fetch:
        return {**result, "status": "configured", "detail": "Patient mapping exists; server access has not been tested."}

    # Use the same adapter as the tool. Keep transport errors visible here; never
    # print credentials, patient IDs, raw records or exception messages.
    observations = service.provider.fetch_latest(patient_id, list(service.config.catalog))
    return {**result, "status": "ok" if observations else "no_data",
            "observation_count": len(observations),
            "measurement_ids": sorted({record.measurement_id for record in observations})}


def check_exercises(session, fetch: bool, wait_seconds: int) -> dict:
    from maya.agents.phyxio_exercise_agent.sessions import load_session_service

    service = load_session_service()
    result = {"provider": service.config.provider, "status": "failed"}
    if service.config.provider != "phyxio":
        return {**result, "detail": "Select the real Phyxio configuration with MAYA_EXERCISE_SESSION_CONFIG."}
    if os.getenv("MAYA_ENABLE_PHYXIO_IO", "1").strip().lower() not in {"1", "true", "yes"}:
        return {**result, "detail": "MAYA_ENABLE_PHYXIO_IO disables the mirror connection."}
    ice = importlib.import_module("Ice")
    result["ice_version"] = ice.stringVersion()
    # Supplied connector 26.5.29 uses the Ice 3.7 API (including initialize(data=)).
    if not result["ice_version"].startswith("3.7."):
        return {**result, "detail": "The supplied Phyxio connector requires Ice 3.7; see requirements.txt."}
    importlib.import_module("phyxio.connect")
    speech = bool(os.getenv("AZURE_SPEECH_KEY") and os.getenv("AZURE_SPEECH_REGION"))
    asr = bool(os.getenv("OPENAI_WHISPER_API_KEY") or all(os.getenv(key) for key in (
        "AZURE_OPENAI_WHISPER_API_KEY", "AZURE_OPENAI_WHISPER_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_WHISPER")))
    if not speech or not asr:
        return {**result, "detail": "The mirror connection requires configured Whisper and Azure Speech credentials."}
    if not fetch:
        return {**result, "status": "configured", "detail": "Imports and settings checked; ICE connection has not been tested."}

    from maya.framework.integrations.phyxio.bridge import get_phyxio_service

    connection = get_phyxio_service(session)
    try:
        print(f"Waiting up to {wait_seconds}s for the participant's Phyxio page. Keep other MAYA processes stopped.", file=sys.stderr)
        if not connection._phyxio.ready.wait(wait_seconds):
            return {**result, "detail": "No Phyxio client connected. Check the browser, connector logs and port 1560."}
        connection.session_history_reader(session.user_id, service.config.phyxio.timeout_ms)
        # Use the same identity guard, selection and comparisons as the agent tool.
        outcome = service.latest(session.user_id)
        return {**result, "status": outcome["status"], "synthetic": outcome["synthetic"],
                "latest_record_found": outcome["latest"] is not None,
                "previous_session_count": len(outcome["previous"])}
    finally:
        connection.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-config", type=Path)
    parser.add_argument("--only", choices=("exercises", "biomarkers"))
    parser.add_argument("--fetch", action="store_true", help="Read live records; stop MAYA first for the exercise check.")
    parser.add_argument("--wait-seconds", type=int, default=30, help="Time allowed for the Phyxio browser to connect.")
    args = parser.parse_args(argv)
    if args.wait_seconds <= 0:
        parser.error("--wait-seconds must be positive")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    # Diagnostics are local, do not call models and must not export tool data.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    report = {"mode": "live_read" if args.fetch else "configuration_only", "checks": {},
              "python": sys.executable, "python_version": sys.version.split()[0],
              "configuration_files": {
                  "session": str(args.session_config or os.getenv("MAYA_SESSION_CONFIG") or "MAYA_RUN_USER_ID"),
                  "exercises": os.getenv("MAYA_EXERCISE_SESSION_CONFIG") or "repository default",
                  "biomarkers": os.getenv("MAYA_BIOMARKER_CONFIG") or "repository default"}}
    try:
        session = load_session(args.session_config)
        report["user_id"] = session.user_id
    except (OSError, ValueError, TypeError) as exc:
        report["checks"]["session"] = {"status": "failed", "error_type": type(exc).__name__,
                                        "detail": "Supply a valid session JSON with a non-empty user_id."}
    else:
        checks = {"exercises": lambda: check_exercises(session, args.fetch, args.wait_seconds),
                  "biomarkers": lambda: check_biomarkers(session, args.fetch)}
        for name, check in checks.items():
            if args.only and args.only != name:
                continue
            try:
                report["checks"][name] = check()
            except Exception as exc:
                failure = {"status": "failed", "error_type": type(exc).__name__}
                if (response := getattr(exc, "response", None)) is not None:
                    failure["http_status"] = response.status_code
                if isinstance(exc, ModuleNotFoundError):
                    failure["missing_module"] = exc.name
                elif isinstance(exc, ValidationError):
                    failure["invalid_fields"] = [list(error["loc"]) for error in exc.errors()]
                report["checks"][name] = failure
    report["passed"] = all(check["status"] in {"configured", "ok", "no_data"} for check in report["checks"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
