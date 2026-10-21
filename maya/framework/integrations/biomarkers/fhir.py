"""Read-only adapter for UCLM's documented libfhir 26.10.16 API."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from libfhir.client import FHIRProxyClient

from . import FHIRConfig, Observation


class FHIRObservationProvider:
    synthetic = False

    def __init__(self, config: FHIRConfig, config_dir: Path, accepted_statuses: list[str]):
        if version("libfhir") != "26.10.16":
            raise ValueError("This adapter requires UCLM libfhir 26.10.16.")
        self.config, self.accepted_statuses = config, accepted_statuses
        credential_path = config_dir / Path(config.client_config_path).expanduser()
        credentials = json.loads(credential_path.read_text())
        url = urlsplit(credentials["url"])
        if (url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment
                or credentials.get("is_secure", True) is not True):
            raise ValueError("FHIR credentials require HTTPS and certificate verification.")
        token_file = credentials.get("token_file")
        if token_file:
            token_path = credential_path.parent / Path(token_file).expanduser()
            # The SDK silently falls back to token if the file is missing. Fail closed.
            with token_path.open() as source:
                if not source.readline().strip():
                    raise ValueError("FHIR token file is empty.")
            credentials["token_file"] = str(token_path.resolve())
            credentials.setdefault("token", None)
        elif not isinstance(credentials.get("token"), str) or not credentials["token"].strip():
            raise ValueError("FHIR credentials require token or token_file.")
        credentials["is_secure"] = True
        self.client = FHIRProxyClient(**credentials)

    @staticmethod
    def _check_units(value, *, quantity=False):
        """Reject numeric quantities stripped of units; never repair SDK output."""
        if quantity and (not isinstance(value, dict)
                         or not isinstance(value.get("value"), (int, float))
                         or isinstance(value.get("value"), bool)):
            raise ValueError("FHIR quantity does not contain a numeric measurement.")
        if isinstance(value, dict):
            number = value.get("value")
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                text = lambda key: isinstance(value.get(key), str) and bool(value[key].strip())
                if not math.isfinite(number) or not (text("unit") or (text("system") and text("code"))):
                    raise ValueError("FHIR numeric quantity has no verifiable units.")
            for key, child in value.items():
                FHIRObservationProvider._check_units(
                    child, quantity=key in {"valueQuantity", "low", "high", "origin", "numerator", "denominator"})
        elif isinstance(value, list):
            for child in value:
                FHIRObservationProvider._check_units(child)

    def _validate_identity(self, item: dict, patient_id: str, measurement_id: str):
        expected = self.config.observation_codes[measurement_id]
        if (item["resourceType"] != "Observation" or item["subject"]["reference"] != f"Patient/{patient_id}"
                or not any(code.get("system") == known.system and code.get("code") == known.code
                           for code in item["code"]["coding"] for known in expected)):
            raise ValueError("FHIR observation does not match the requested patient or measurement.")

    def _normalize(self, item: dict, patient_id: str, measurement_id: str) -> Observation:
        period = item.get("effectivePeriod", {})
        measured_at = item.get("effectiveDateTime") or period.get("end") or period.get("start")
        if not any(key.startswith("value") or key in {"component", "dataAbsentReason"} for key in item):
            raise ValueError("FHIR observation has neither a result nor an absence reason.")
        self._check_units(item)
        return Observation(id=item["id"], patient_id=patient_id, measurement_id=measurement_id,
                           measured_at=measured_at, status=item["status"], source=f"FHIR Observation/{item['id']}",
                           result=deepcopy(item), reference_range={"ranges": deepcopy(item["referenceRange"])}
                           if "referenceRange" in item else None)

    def fetch_latest(self, patient_id: str, measurement_ids: list[str]) -> list[Observation]:
        latest = []
        try:
            if not patient_id.strip():
                raise ValueError("FHIR patient identity is required.")
            for measurement_id in measurement_ids:
                # No timeout is supported by the public API. Empty results are ambiguous:
                # the SDK also returns [] on network/proxy failures and parsing omissions.
                items = self.client.get(self.config.resource_types[measurement_id], subject=patient_id)
                if not isinstance(items, list) or not items:
                    raise ValueError("FHIR lookup outcome cannot be verified.")
                records = []
                for record in items:
                    item = record.dump()
                    self._validate_identity(item, patient_id, measurement_id)
                    if item["status"] in self.accepted_statuses:
                        records.append(self._normalize(item, patient_id, measurement_id))
                if records:
                    measured_at = max(record.measured_at for record in records)
                    latest.extend(record for record in records if record.measured_at == measured_at)
        except Exception:
            # SDK/network exceptions can include response content. Expose no raw details.
            raise ValueError("FHIR observations could not be accessed or validated.") from None
        return latest
