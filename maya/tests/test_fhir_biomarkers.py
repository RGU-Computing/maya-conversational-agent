"""Public libfhir API tests. All patients, credentials and HTTP responses are synthetic."""

import copy
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from maya.framework.integrations.biomarkers import load_biomarker_service
from maya.tests.test_biomarkers import request


EXAMPLE = Path(__file__).resolve().parents[1] / "framework/integrations/biomarkers/config.fhir.example.json"


def observation(resource_type="blood-pressure", **updates):
    # LOINC codes from the supplied Usage Guide; no SDK internal catalogue access.
    codes = {"blood-pressure": "85354-9", "body-weight": "29463-7", "heart-rate": "8867-4", "body-height": "8302-2"}

    record = {
        "resourceType": "Observation", "id": "measurement-1", "status": "final",
        "subject": {"reference": "Patient/synthetic-patient"},
        "effectiveDateTime": "2026-10-18T10:30:00Z",
        "code": {"coding": [{"system": "http://loinc.org", "code": codes[resource_type]}]},
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

    def fetch(self, records=(), *, ids=("blood_pressure",), **kwargs):
        with patch("requests.Session.get", return_value=response(records, **kwargs)) as get:
            result = self.service.fetch("participant-001", request(*ids), [])[0]
        return result, get

    def test_loads_provider_and_uses_scoped_read_only_contract(self):
        result, get = self.fetch([observation()])
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["synthetic"])
        get.assert_called_once_with("https://fhir.invalid/proxy/record/get",
            json={"token": "synthetic-secret", "resource_type": "blood-pressure", "value": {"subject": "synthetic-patient"}},
            verify=True, headers={"Content-Type": "application/json"})
        self.assertNotIn("synthetic-secret", json.dumps(result))
        self.assertNotIn("synthetic-patient", json.dumps(result))

    def test_scalar_unit_loss_is_rejected_without_inventing_units(self):
        record = observation("body-weight", valueQuantity={"value": 150, "unit": "lb", "system": "http://unitsofmeasure.org", "code": "[lb_av]"},
            referenceRange=[{"low": {"value": 100, "unit": "lb"}, "high": {"value": 200, "unit": "lb"}}])
        del record["component"]
        result, _ = self.fetch([record], ids=("body_weight",))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["observations"], [])
        # A conventional unit or a reference range must not supply an assumed unit.
        self.assertNotIn("150", json.dumps(result))

    def test_component_unit_loss_is_also_rejected(self):
        record = observation()
        quantity = record["component"][0]["valueQuantity"]
        del quantity["unit"], quantity["code"]
        result, _ = self.fetch([record])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["observations"], [])

    def test_sdk_dump_fields_and_component_units_are_retained(self):
        from libfhir.observation import Observation

        record = observation(device={"reference": "Device/synthetic-device"},
            basedOn=[{"reference": "CarePlan/synthetic-plan"}],
            derivedFrom=[{"reference": "Observation/synthetic-source"}],
            encounter={"reference": "Encounter/synthetic-encounter"})
        expected = Observation.load(copy.deepcopy(record)).dump()
        result, _ = self.fetch([record])
        for field in ("code", "component", "device", "basedOn", "derivedFrom", "encounter", "effectiveDateTime"):
            self.assertEqual(result["observations"][0]["result"][field], expected[field])

    def test_latest_uses_documented_utc_timestamps_preserves_ties_and_filters_status(self):
        records = [observation(id="old", effectiveDateTime="2026-10-17T10:30:00Z"),
                   observation(id="latest"), observation(id="tie"),
                   observation(id="excluded", status="entered-in-error", effectiveDateTime="2026-10-19T12:00:00Z")]
        result, _ = self.fetch(records)
        self.assertEqual({r["id"] for r in result["observations"]}, {"latest", "tie"})
        self.assertEqual(self.fetch([observation(status="entered-in-error")])[0]["status"], "no_data")

    def test_documented_period_is_retained(self):
        record = observation(effectivePeriod={"start": "2026-10-18T10:00:00Z", "end": "2026-10-18T11:00:00Z"})
        del record["effectiveDateTime"]
        result, _ = self.fetch([record])
        self.assertEqual(result["observations"][0]["measured_at"], "2026-10-18T11:00:00Z")
        self.assertEqual(result["observations"][0]["result"]["effectivePeriod"], record["effectivePeriod"])

    def test_ambiguous_empty_sdk_results_do_not_claim_absent_records(self):
        self.assertEqual(self.fetch()[0]["status"], "unavailable")
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
            with self.subTest(error=error.__name__), patch("requests.Session.get", side_effect=error("synthetic-secret")):
                result = self.service.fetch("participant-001", request("blood_pressure"), [])[0]
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("synthetic-secret", json.dumps(result))
        invalid = response()
        invalid._content = b"not json"
        with patch("requests.Session.get", return_value=invalid):
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure"), [])[0]["status"], "unavailable")

    def test_wrong_patient_and_unparseable_sdk_results_are_unavailable(self):
        bad_records = [observation(subject={"reference": "Patient/someone-else"}),
                      observation(effectiveDateTime="2026-10-18T10:30:00+00:00"),
                      observation(effectiveDateTime="2026-10-18T10:30:00"),
                      observation(effectiveDateTime=None), observation(code={}), None, {}]
        for record in bad_records:
            with self.subTest(record=record):
                result, _ = self.fetch([record])
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["observations"], [])

    def test_identity_guards_do_not_call_the_server(self):
        with patch("requests.Session.get") as get:
            self.assertEqual(self.service.fetch("unmapped", request("blood_pressure"), [])[0]["status"], "unmapped_patient")
            self.assertEqual(self.service.fetch("participant-001", request("invented"), [])[0]["status"], "unsupported_measurement")
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure", subject="other"), []), [])
            get.assert_not_called()

    def test_each_participant_uses_only_their_patient_mapping(self):
        self.service.config.patient_map = {"participant-a": "patient-a", "participant-b": "patient-b"}
        for user_id, patient_id in self.service.config.patient_map.items():
            record = observation(subject={"reference": f"Patient/{patient_id}"})
            with patch("requests.Session.get", return_value=response([record])) as get:
                result = self.service.fetch(user_id, request("blood_pressure"), [])[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(get.call_args.kwargs["json"]["value"], {"subject": patient_id})

    def test_multiple_measurements_are_scoped_and_not_refetched_on_resume(self):
        with patch("requests.Session.get", side_effect=[response([observation()]), response()]) as get:
            results = self.service.fetch("participant-001", request("blood_pressure", "heart_rate"), [])
            self.assertEqual(results[0]["missing"], ["heart_rate"])
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure"), results), results)
            self.assertEqual(get.call_count, 2)

    def test_invalid_mapping_and_insecure_credentials_fail_at_configuration(self):
        config_path = Path(self.folder.name) / "biomarkers.json"
        original = json.loads(config_path.read_text())
        config = copy.deepcopy(original)
        config["fhir"]["resource_types"] = {}
        config_path.write_text(json.dumps(config))
        with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(config_path)}), self.assertRaises(ValueError):
            load_biomarker_service()
        config_path.write_text(json.dumps(original))
        for update in ({"url": "http://fhir.invalid"}, {"is_secure": False}, {"token": ""}, {"url": "https://fhir.invalid/?token=secret"}):
            credentials = {"url": "https://fhir.invalid", "token": "synthetic-secret", **update}
            (Path(self.folder.name) / "client.json").write_text(json.dumps(credentials))
            with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(config_path)}), self.assertRaises(ValueError):
                load_biomarker_service()

    def reload(self):
        with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(Path(self.folder.name) / "biomarkers.json")}):
            self.service = load_biomarker_service()

    def test_only_public_get_and_dump_are_required(self):
        with patch.object(self.service.provider.client, "get", wraps=self.service.provider.client.get) as get:
            self.fetch([observation()])
        get.assert_called_once_with("blood-pressure", subject="synthetic-patient")
        record = Mock(spec=["dump"])
        record.dump.return_value = observation()
        client = Mock(spec=["get"])
        client.get.return_value = [record]
        with patch("maya.framework.integrations.biomarkers.fhir.FHIRProxyClient", return_value=client):
            self.reload()
        result = self.service.fetch("participant-001", request("blood_pressure"), [])[0]
        self.assertEqual(result["status"], "ok")
        record.dump.assert_called_once_with()

    def test_documented_token_file_authentication_and_precedence(self):
        folder = Path(self.folder.name)
        (folder / "token.txt").write_text("synthetic-file-token\n")
        for token in (None, "synthetic-inline-token"):
            with self.subTest(token=token):
                (folder / "client.json").write_text(json.dumps({
                    "url": "https://fhir.invalid/proxy", "token": token, "token_file": "token.txt"}))
                self.reload()
                result, http = self.fetch([observation()])
                self.assertEqual(result["status"], "ok")
                self.assertEqual(http.call_args.kwargs["json"]["token"], "synthetic-file-token")
                self.assertNotIn("synthetic-file-token", json.dumps(result))
        (folder / "token.txt").unlink()
        with self.assertRaises(ValueError):
            self.reload()

    def test_additional_documented_measurement_is_enabled_by_configuration(self):
        path = Path(self.folder.name) / "biomarkers.json"
        settings = json.loads(path.read_text())
        settings["catalog"]["body_height"] = "Recorded height."
        settings["fhir"]["resource_types"]["body_height"] = "body-height"
        path.write_text(json.dumps(settings))
        self.reload()
        record = observation("body-height", valueQuantity={"value": 1.8, "unit": "m"})
        del record["component"]
        result, http = self.fetch([record], ids=("body_height",))
        self.assertEqual(result["status"], "unavailable")  # SDK also drops scalar height units.
        self.assertEqual(http.call_args.kwargs["json"]["resource_type"], "body-height")

    def test_obsolete_timeout_configuration_is_not_silently_ignored(self):
        path = Path(self.folder.name) / "biomarkers.json"
        settings = json.loads(path.read_text())
        settings["fhir"]["timeout_seconds"] = 10
        path.write_text(json.dumps(settings))
        with self.assertRaises(ValueError):
            self.reload()


if __name__ == "__main__":
    unittest.main()
