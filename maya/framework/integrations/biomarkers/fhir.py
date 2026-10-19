"""Read-only adapter for UCLM's libfhir 26.10.16 proxy contract."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from libfhir import FHIRProxyClient, ResourceCreator
from requests import HTTPError

from . import FHIRConfig, Observation


class FHIRObservationProvider:
    synthetic = False

    def __init__(self, config: FHIRConfig, config_dir: Path, accepted_statuses: list[str]):
        self.config, self.accepted_statuses = config, accepted_statuses
        unsupported = set(config.resource_types.values()) - ResourceCreator.supported_observations.keys()
        if unsupported:
            raise ValueError(f"Unsupported libfhir observation types: {sorted(unsupported)}")
        credentials = json.loads((config_dir / Path(config.client_config_path).expanduser()).read_text())
        url = urlsplit(credentials["url"])
        if (url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment
                or credentials.get("is_secure", True) is not True or not credentials.get("token")):
            raise ValueError("FHIR credentials require an HTTPS URL, a token and certificate verification.")
        self.client = FHIRProxyClient(**credentials)

    def _get(self, resource_type: str, **params) -> list[dict]:
        # libfhir.get() discards scalar units and cannot parse timezone offsets.
        # Use its authenticated transport, retaining the original FHIR resources.
        with self.client._session.get(
            self.client._base_url.rstrip("/") + "/record/get",
            json={"token": self.client._token, "resource_type": resource_type, "value": params},
            verify=True, timeout=self.config.timeout_seconds, allow_redirects=False,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                raise HTTPError("FHIR observation service is unavailable.", response=response)
            payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "ok" or not isinstance(payload.get("items"), list):
            raise ValueError("Invalid FHIR observation response.")
        return payload["items"]

    def _normalize(self, item: dict, patient_id: str, measurement_id: str) -> Observation:
        resource_type = self.config.resource_types[measurement_id]
        expected = ResourceCreator.supported_observations[resource_type]["code"]["coding"]
        if (item["resourceType"] != "Observation" or item["subject"]["reference"] != f"Patient/{patient_id}"
                or not any(code.get("system") == known["system"] and code.get("code") == known["code"]
                           for code in item["code"]["coding"] for known in expected)):
            raise ValueError("FHIR observation does not match the requested patient or measurement.")
        # Keep the period as context; use its end (or start for an open period) for ordering.
        period = item.get("effectivePeriod", {})
        measured_at = item.get("effectiveDateTime") or period.get("end") or period.get("start")
        context_fields = {"code", "component", "effectivePeriod", "dataAbsentReason", "interpretation",
                          "method", "bodySite", "specimen", "device", "note"}
        result = {key: value for key, value in item.items() if key.startswith("value") or key in context_fields}
        if not any(key.startswith("value") or key in {"component", "dataAbsentReason"} for key in result):
            raise ValueError("FHIR observation has neither a result nor an absence reason.")
        return Observation(id=item["id"], patient_id=patient_id, measurement_id=measurement_id,
                           measured_at=measured_at, status=item["status"], source=f"FHIR Observation/{item['id']}",
                           result=result, reference_range={"ranges": item["referenceRange"]}
                           if "referenceRange" in item else None)

    def fetch_latest(self, patient_id: str, measurement_ids: list[str]) -> list[Observation]:
        latest = []
        try:
            for measurement_id in measurement_ids:
                items = self._get(self.config.resource_types[measurement_id], subject=patient_id)
                records = [self._normalize(item, patient_id, measurement_id)
                           for item in items if item["status"] in self.accepted_statuses]
                if records:
                    measured_at = max(record.measured_at for record in records)
                    latest.extend(record for record in records if record.measured_at == measured_at)
        except (KeyError, TypeError, AttributeError):
            raise ValueError("Malformed FHIR observation response.") from None
        return latest
