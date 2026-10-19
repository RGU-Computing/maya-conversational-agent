"""FHIR adapter contract tests. All patients, credentials and responses are synthetic."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from maya.framework.integrations.biomarkers import load_biomarker_service
from maya.tests.test_biomarkers import request


EXAMPLE = Path(__file__).resolve().parents[1] / "framework/integrations/biomarkers/config.fhir.example.json"


def observation(resource_type="blood-pressure", **updates):
    from libfhir import ResourceCreator

    record = {
        "resourceType": "Observation", "id": "measurement-1", "status": "final",
        "subject": {"reference": "Patient/synthetic-patient"},
        "effectiveDateTime": "2026-10-18T10:30:00Z",
        "code": copy.deepcopy(ResourceCreator.supported_observations[resource_type]["code"]),
        "component": [
            {"code": {"text": "Systolic"}, "valueQuantity": {"value": 134, "unit": "mmHg", "code": "mm[Hg]"}},
            {"code": {"text": "Diastolic"}, "valueQuantity": {"value": 82, "unit": "mmHg", "code": "mm[Hg]"}},
        ],
    }
    return {**record, **updates}


def response(items=(), *, status=200, body=None):
    from requests import Response

    result = Response()
    result.status_code = status
    result._content = json.dumps(body if body is not None else {"status": "ok", "items": list(items)}).encode()
    result._content_consumed = True  # Match requests' default non-streaming response.
    return result


def configured_service(folder):
    folder = Path(folder)
    credentials = folder / "client.json"
    credentials.write_text(json.dumps({"url": "https://fhir.invalid/proxy/", "token": "synthetic-secret", "is_secure": True}))
    settings = json.loads(EXAMPLE.read_text())
    settings["fhir"]["client_config_path"] = credentials.name
    settings["patient_map"] = {"participant-001": "synthetic-patient"}
    config = folder / "biomarkers.json"
    config.write_text(json.dumps(settings))
    with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(config)}):
        return load_biomarker_service()


@unittest.skipUnless(importlib.util.find_spec("libfhir"), "Requires the UCLM libfhir wheel.")
class FHIRBiomarkerTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.service = configured_service(self.folder.name)
        self.addCleanup(self.service.provider.client._session.close)

    def fetch(self, records=(), *, ids=("blood_pressure",), **kwargs):
        with patch.object(self.service.provider.client._session, "get", return_value=response(records, **kwargs)) as get:
            result = self.service.fetch("participant-001", request(*ids), [])[0]
        return result, get

    def test_loads_provider_and_uses_scoped_read_only_contract(self):
        result, get = self.fetch([observation()])
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["synthetic"])
        get.assert_called_once_with("https://fhir.invalid/proxy/record/get",
            json={"token": "synthetic-secret", "resource_type": "blood-pressure", "value": {"subject": "synthetic-patient"}},
            verify=True, timeout=10, allow_redirects=False, headers={"Content-Type": "application/json"})
        self.assertNotIn("synthetic-secret", json.dumps(result))
        self.assertNotIn("synthetic-patient", json.dumps(result))

    def test_scalar_units_and_all_reference_ranges_are_preserved_without_defaults(self):
        record = observation("body-weight", valueQuantity={"value": 150, "unit": "lb", "system": "unit-system", "code": "[lb_av]"},
            referenceRange=[{"low": {"value": 100, "unit": "lb"}, "type": {"text": "source-defined"}}, {"text": "Other range"}])
        del record["component"]
        result, _ = self.fetch([record], ids=("body_weight",))
        observed = result["observations"][0]
        self.assertEqual(observed["result"]["valueQuantity"], record["valueQuantity"])
        self.assertEqual(observed["reference_range"], {"ranges": record["referenceRange"]})
        self.assertEqual(observed["source"], "FHIR Observation/measurement-1")
        record["valueQuantity"].pop("unit")
        result, _ = self.fetch([record], ids=("body_weight",))
        self.assertNotIn("unit", result["observations"][0]["result"]["valueQuantity"])

    def test_components_and_measurement_context_are_preserved(self):
        record = observation(method={"text": "Home cuff"}, bodySite={"text": "Left arm"},
                             note=[{"text": "Seated after rest"}], interpretation=[{"text": "Source interpretation"}])
        result, _ = self.fetch([record])
        for field in ("component", "method", "bodySite", "note", "interpretation"):
            self.assertEqual(result["observations"][0]["result"][field], record[field])

    def test_latest_uses_absolute_time_preserves_ties_and_excludes_invalid_status(self):
        records = [observation(id="old", effectiveDateTime="2026-10-18T11:00:00+02:00"),
                   observation(id="latest", effectiveDateTime="2026-10-18T10:30:00.125Z"),
                   observation(id="tie", effectiveDateTime="2026-10-18T11:30:00.125+01:00"),
                   observation(id="excluded", status="entered-in-error", effectiveDateTime="2026-10-19T12:00:00Z")]
        result, _ = self.fetch(records)
        self.assertEqual({r["id"] for r in result["observations"]}, {"latest", "tie"})

    def test_period_and_absent_measurement_are_explicit(self):
        record = observation(effectivePeriod={"start": "2026-10-18T10:00:00Z", "end": "2026-10-18T11:00:00Z"})
        del record["effectiveDateTime"]
        result, _ = self.fetch([record])
        self.assertEqual(result["observations"][0]["measured_at"], "2026-10-18T11:00:00Z")
        self.assertEqual(result["observations"][0]["result"]["effectivePeriod"], record["effectivePeriod"])
        del record["component"]
        record["dataAbsentReason"] = {"text": "Not measured"}
        result, _ = self.fetch([record])
        self.assertEqual(result["observations"][0]["result"]["dataAbsentReason"], {"text": "Not measured"})

    def test_empty_is_not_a_transport_or_protocol_failure(self):
        self.assertEqual(self.fetch()[0]["status"], "no_data")
        for status in (301, 401, 403, 404, 500):
            with self.subTest(status=status):
                self.assertEqual(self.fetch(status=status)[0]["status"], "unavailable")
        for payload in ([], {}, {"status": "error", "message": "synthetic-secret"}, {"status": "ok", "items": None}):
            with self.subTest(payload=payload):
                result, _ = self.fetch(body=payload)
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("synthetic-secret", json.dumps(result))

    def test_connection_timeout_and_invalid_json_are_handled(self):
        from requests.exceptions import ConnectionError, SSLError, Timeout

        for error in (Timeout, ConnectionError, SSLError):
            with self.subTest(error=error.__name__), patch.object(self.service.provider.client._session, "get", side_effect=error("synthetic-secret")):
                result = self.service.fetch("participant-001", request("blood_pressure"), [])[0]
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("synthetic-secret", json.dumps(result))
        invalid = response()
        invalid._content = b"not json"
        with patch.object(self.service.provider.client._session, "get", return_value=invalid):
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure"), [])[0]["status"], "unavailable")

    def test_wrong_patient_wrong_measurement_and_malformed_records_are_rejected(self):
        bad_records = [observation(subject={"reference": "Patient/someone-else"}),
                       observation("heart-rate"), observation(effectiveDateTime="2026-10-18T10:00:00"),
                       observation(effectiveDateTime=None), observation(code={}),
                       observation(resourceType="Patient"), None, {}]
        for record in bad_records:
            with self.subTest(record=record):
                result, _ = self.fetch([observation(), record])
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["observations"], [])

    def test_identity_guards_do_not_call_the_server(self):
        with patch.object(self.service.provider.client._session, "get") as get:
            self.assertEqual(self.service.fetch("unmapped", request("blood_pressure"), [])[0]["status"], "unmapped_patient")
            self.assertEqual(self.service.fetch("participant-001", request("invented"), [])[0]["status"], "unsupported_measurement")
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure", subject="other"), []), [])
            get.assert_not_called()

    def test_each_participant_uses_only_their_patient_mapping(self):
        self.service.config.patient_map = {"participant-a": "patient-a", "participant-b": "patient-b"}
        for user_id, patient_id in self.service.config.patient_map.items():
            record = observation(subject={"reference": f"Patient/{patient_id}"})
            with patch.object(self.service.provider.client._session, "get", return_value=response([record])) as get:
                result = self.service.fetch(user_id, request("blood_pressure"), [])[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(get.call_args.kwargs["json"]["value"], {"subject": patient_id})

    def test_multiple_measurements_are_scoped_and_not_refetched_on_resume(self):
        with patch.object(self.service.provider.client._session, "get", side_effect=[response([observation()]), response()]) as get:
            results = self.service.fetch("participant-001", request("blood_pressure", "heart_rate"), [])
            self.assertEqual(results[0]["missing"], ["heart_rate"])
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure"), results), results)
            self.assertEqual(get.call_count, 2)

    def test_invalid_mapping_and_insecure_credentials_fail_at_configuration(self):
        config_path = Path(self.folder.name) / "biomarkers.json"
        original = json.loads(config_path.read_text())
        for value in ({}, {"blood_pressure": "not-a-libfhir-type", "heart_rate": "heart-rate", "body_weight": "body-weight"}):
            config = copy.deepcopy(original)
            config["fhir"]["resource_types"] = value
            config_path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(config_path)}), self.assertRaises(ValueError):
                load_biomarker_service()
        config_path.write_text(json.dumps(original))
        for update in ({"url": "http://fhir.invalid"}, {"is_secure": False}, {"token": ""}, {"url": "https://fhir.invalid/?token=secret"}):
            credentials = {"url": "https://fhir.invalid", "token": "synthetic-secret", **update}
            (Path(self.folder.name) / "client.json").write_text(json.dumps(credentials))
            with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(config_path)}), self.assertRaises(ValueError):
                load_biomarker_service()


if __name__ == "__main__":
    unittest.main()
