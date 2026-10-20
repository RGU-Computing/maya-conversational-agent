"""Read-only adapter using the documented libfhir client.get()/record.dump() API."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from libfhir.client import FHIRProxyClient

from . import FHIRConfig, Observation


class FHIRObservationProvider:
    synthetic = False

    def __init__(self, config: FHIRConfig, config_dir: Path, accepted_statuses: list[str]):
        self.config, self.accepted_statuses = config, accepted_statuses
        credential_path = config_dir / Path(config.client_config_path).expanduser()
        credentials = json.loads(credential_path.read_text())
        url = urlsplit(credentials["url"])
        if (url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment
                or credentials.get("is_secure", True) is not True):
            raise ValueError("FHIR credentials require an HTTPS URL and certificate verification.")
        if credentials.get("token_file"):
            token_path = credential_path.parent / Path(credentials["token_file"]).expanduser()
            if not token_path.is_file():
                raise ValueError("FHIR token_file must point to an existing private file.")
            credentials["token_file"] = str(token_path)
            credentials.setdefault("token", None)
        elif not credentials.get("token"):
            raise ValueError("FHIR credentials require token or token_file.")
        credentials["url"] = credentials["url"].rstrip("/")
        self.client = FHIRProxyClient(**credentials)

    def _normalize(self, item: dict, patient_id: str, measurement_id: str) -> Observation:
        if item["resourceType"] != "Observation" or item["subject"]["reference"] != f"Patient/{patient_id}":
            raise ValueError("FHIR observation does not match the session patient.")
        quantities = [item.get("valueQuantity", {}), *(
            component.get("valueQuantity", {}) for component in item.get("component", []))]
        if any("value" in quantity and not (quantity.get("unit") or quantity.get("code")) for quantity in quantities):
            raise ValueError("FHIR quantity has no verifiable unit.")
        # Keep the period as context; use its end (or start for an open period) for ordering.
        period = item.get("effectivePeriod", {})
        measured_at = item.get("effectiveDateTime") or period.get("end") or period.get("start")
        # Keep all SDK-serialized fields except identity/metadata represented separately.
        result = {key: value for key, value in item.items()
                  if key not in {"resourceType", "id", "subject", "status", "referenceRange"}}
        if not any(key.startswith("value") or key in {"component", "dataAbsentReason"} for key in result):
            raise ValueError("FHIR observation has neither a result nor an absence reason.")
        return Observation(id=item["id"], patient_id=patient_id, measurement_id=measurement_id,
                           measured_at=measured_at, status=item["status"], source=f"FHIR Observation/{item['id']}",
                           result=result, reference_range={"ranges": item["referenceRange"]}
                           if "referenceRange" in item else None)

    def fetch_latest(self, patient_id: str, measurement_ids: list[str]) -> list[Observation]:
        latest = []
        unresolved = False
        try:
            for measurement_id in measurement_ids:
                items = self.client.get(self.config.resource_types[measurement_id], subject=patient_id)
                # 26.10.16 returns [] for both empty searches and some transport/parser
                # failures. Do not claim a successful empty search from that result.
                if items is None:
                    raise ValueError("libfhir could not retrieve observations.")
                if not items:
                    unresolved = True
                    continue
                records = [self._normalize(item.dump(), patient_id, measurement_id) for item in items]
                records = [record for record in records if record.status in self.accepted_statuses]
                if records:
                    measured_at = max(record.measured_at for record in records)
                    latest.extend(record for record in records if record.measured_at == measured_at)
        except (KeyError, TypeError, AttributeError, AssertionError):
            raise ValueError("Invalid libfhir observation result.") from None
        if not latest and unresolved:
            raise ValueError("libfhir returned no verifiable observations; the lookup outcome is ambiguous.")
        return latest
