"""Participant-scoped observations, independent of the upstream transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Protocol

from langchain_core.tools import tool
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from maya.framework.conversation.responses import localize_text


class BiomarkerRequest(BaseModel):
    subject: Literal["current_user", "other", "unknown"]
    reason: str = Field(description=(
        "Identify the measurement needed from the user's request and established context before matching the catalog. "
        "Explain the match or what remains unidentified; catalog availability is not evidence of what the user means."
    ))
    catalog_match: Literal["matched", "ambiguous", "unsupported"] = Field(description=(
        "Whether the measurement's identity matches the catalog, NOT whether its value is known. "
        "matched when the measurement is identified and has a catalog entry, even without a value or unit; "
        "ambiguous only when its identity is unknown; unsupported when identified but absent from the catalog. "
        "Missing method, value or unit does not change catalog membership."
    ))
    measurement_ids: list[str] = Field(description=(
        "Exact catalog IDs for matched measurements only; empty for ambiguous or unsupported requests. "
        "Never substitute a different measurement or select a merely related catalog item."
    ))


class Observation(BaseModel):
    """Provider output; retain source units and component grouping without conversion."""

    model_config = ConfigDict(extra="forbid")
    id: str
    patient_id: str
    measurement_id: str
    measured_at: AwareDatetime
    status: str
    source: str
    # Source values may be quantities, coded/text results, or grouped components.
    result: dict
    reference_range: dict | None = None


class FHIRCode(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    system: str = Field(min_length=1)
    code: str = Field(min_length=1)


class FHIRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_config_path: str
    resource_types: dict[str, str]
    observation_codes: dict[str, list[FHIRCode]]

    @model_validator(mode="after")
    def validate_measurements(self):
        if (set(self.observation_codes) != set(self.resource_types)
                or any(not value.strip() for value in self.resource_types.values())
                or any(not codes for codes in self.observation_codes.values())):
            raise ValueError("FHIR resource types require matching, nonempty observation codes.")
        return self


class BiomarkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["disabled", "dummy", "fhir"]
    catalog: dict[str, str]
    patient_map: dict[str, str]
    accepted_statuses: list[str]
    fixture_path: str | None = None
    fhir: FHIRConfig | None = None
    request_instruction: str
    context_instruction: str
    readout_instruction: str
    synthetic_notice: str


class ObservationProvider(Protocol):
    synthetic: bool

    def fetch_latest(self, patient_id: str, measurement_ids: list[str]) -> list[Observation]: ...


class DummyObservationProvider:
    synthetic = True

    def __init__(self, path: Path, accepted_statuses: list[str]):
        self.records = [Observation.model_validate(item) for item in json.loads(path.read_text())]
        self.accepted_statuses = accepted_statuses

    def fetch_latest(self, patient_id: str, measurement_ids: list[str]) -> list[Observation]:
        records = [r for r in self.records if r.patient_id == patient_id
                   and r.measurement_id in measurement_ids and r.status in self.accepted_statuses]
        latest = {key: max(r.measured_at for r in records if r.measurement_id == key)
                  for key in {r.measurement_id for r in records}}
        # Preserve ties rather than arbitrarily selecting a conflicting measurement.
        return [r for r in records if r.measured_at == latest[r.measurement_id]]


class BiomarkerService:
    def __init__(self, config: BiomarkerConfig, provider: ObservationProvider | None):
        self.config, self.provider = config, provider

    def catalog_context(self, previous: list[dict] = ()) -> str:
        attempted = {key for result in previous for key in result["requested"]}
        catalog = {key: label for key, label in self.config.catalog.items() if key not in attempted}
        if self.provider is None:
            catalog = {}
        return self.config.request_instruction + "\nAvailable observation catalog:\n" + json.dumps(catalog)

    def fetch(self, user_id: str, request: BiomarkerRequest | None, previous: list[dict]) -> list[dict]:
        if request is None or request.subject != "current_user":
            return previous
        if request.catalog_match != "matched":
            return [*previous, {"requested": [], "observations": [], "missing": [], "synthetic": False,
                               "status": f"{request.catalog_match}_measurement", "reason": request.reason}]
        attempted = {key for result in previous for key in result["requested"]}
        requested = list(dict.fromkeys(key for key in request.measurement_ids if key not in attempted))
        if not requested:
            return previous

        # Bind identity outside the model-visible tool arguments.
        @tool
        def fetch_latest_biomarkers(measurement_ids: list[str]) -> dict:
            """Fetch latest observations for the session participant using catalog IDs."""
            result = {"requested": measurement_ids, "observations": [], "missing": measurement_ids,
                      "synthetic": bool(self.provider and self.provider.synthetic)}
            if any(key not in self.config.catalog for key in measurement_ids):
                return {**result, "status": "unsupported_measurement"}
            if self.provider is None:
                return {**result, "status": "unavailable"}
            patient_id = self.config.patient_map.get(user_id)
            if not patient_id:
                return {**result, "status": "unmapped_patient"}
            try:
                records = self.provider.fetch_latest(patient_id, measurement_ids)
                if any(r.patient_id != patient_id or r.measurement_id not in measurement_ids for r in records):
                    return {**result, "status": "invalid_response"}
                records = [r for r in records if r.status in self.config.accepted_statuses]
            except (OSError, ValueError, TimeoutError):
                return {**result, "status": "unavailable"}
            found = {r.measurement_id for r in records}
            return {**result, "status": "ok" if records else "no_data",
                    "missing": [key for key in measurement_ids if key not in found],
                    "observations": [r.model_dump(mode="json", exclude={"patient_id"}) for r in records]}

        # Each measurement keeps its own outcome: a failed lookup must not discard
        # another measurement's verified observations. Validation remains unchanged.
        return [*previous, *(fetch_latest_biomarkers.invoke({"measurement_ids": [key]}) for key in requested)]

    @staticmethod
    def model_results(results: list[dict]) -> list[dict]:
        """Project verified observations for the model; retain raw records locally."""
        clinical_fields = {"resourceType", "status", "code", "category", "effectiveDateTime",
                           "effectivePeriod", "issued", "interpretation", "referenceRange",
                           "bodySite", "method", "dataAbsentReason", "component"}

        def clinical_result(value):
            return {key: ([clinical_result(item) for item in child] if key == "component" else child)
                    for key, child in value.items() if key in clinical_fields or key.startswith("value")}

        return [{**result, "observations": [
            {key: (clinical_result(value) if key == "result" and value.get("resourceType") == "Observation" else value)
             for key, value in observation.items()
             if key in {"measurement_id", "measured_at", "status", "result", "reference_range"}}
            for observation in result.get("observations", [])]
        } for result in results]

    def render_context(self, results: list[dict]) -> str:
        if not results:
            return ""
        return self.config.context_instruction + "\nObservation lookup results:\n" + json.dumps(self.model_results(results))

    def label_answer(self, text: str, results: list[dict], response_language: str | None = None) -> str:
        if self.config.synthetic_notice and any(result["synthetic"] and result["observations"] for result in results):
            return localize_text(self.config.synthetic_notice, response_language) + "\n\n" + text
        return text


def load_biomarker_service() -> BiomarkerService:
    default_path = Path(__file__).with_name("config.json")
    path = Path(os.environ.get("MAYA_BIOMARKER_CONFIG") or default_path)
    config = BiomarkerConfig.model_validate({**json.loads(default_path.read_text()), **json.loads(path.read_text())})
    provider = None
    if config.provider == "dummy":
        if not config.fixture_path:
            raise ValueError("The dummy observation provider requires fixture_path.")
        provider = DummyObservationProvider(path.parent / config.fixture_path, config.accepted_statuses)
    elif config.provider == "fhir":
        from .fhir import FHIRObservationProvider

        if config.fhir is None or set(config.fhir.resource_types) != set(config.catalog):
            raise ValueError("The FHIR provider requires settings and a resource type for every catalog ID.")
        provider = FHIRObservationProvider(config.fhir, path.parent, config.accepted_statuses)
    return BiomarkerService(config, provider)
