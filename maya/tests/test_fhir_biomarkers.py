"""Synthetic public-API contract tests; no UCLM network or participant records."""

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
CLIENT = "maya.framework.integrations.biomarkers.fhir.FHIRProxyClient"


def observation(resource_type="blood-pressure", **updates):
    codes = {"blood-pressure": "85354-9", "heart-rate": "8867-4", "body-weight": "29463-7"}
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


class PublicRecord:
    def __init__(self, payload):
        self.payload = payload

    def dump(self):
        return copy.deepcopy(self.payload)


def records(*payloads):
    return [PublicRecord(payload) for payload in payloads]


def response(items=(), *, status=200, body=None):
    """Transport fixture for installed SDK characterization only."""
    from requests import Response

    result = Response()
    result.status_code = status
    result._content = json.dumps(body if body is not None else {"status": "ok", "items": list(items)}).encode()
    result._content_consumed = True
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
    # Substitute our dependency at the adapter boundary, never mutate UCLM classes.
    with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(config)}), patch(CLIENT) as client:
        client.return_value = Mock(spec=["get"])
        return load_biomarker_service()


@unittest.skipUnless(importlib.util.find_spec("libfhir"), "Requires the UCLM libfhir wheel.")
class FHIRBiomarkerTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.service = configured_service(self.folder.name)

    def fetch(self, payloads=(), *, ids=("blood_pressure",), error=None):
        get = self.service.provider.client.get
        get.reset_mock()
        get.side_effect = error
        get.return_value = records(*payloads)
        return self.service.fetch("participant-001", request(*ids), [])[0], get

    def test_public_scoped_read_only_contract(self):
        result, get = self.fetch([observation()])
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["synthetic"])
        get.assert_called_once_with("blood-pressure", subject="synthetic-patient")
        self.assertNotIn("synthetic-secret", json.dumps(result))
        self.assertEqual(result["observations"][0]["source"], "FHIR Observation/measurement-1")

    def test_all_dump_fields_and_units_are_preserved(self):
        record = observation(method={"text": "Home cuff"}, note=[{"text": "Seated"}],
            extension=[{"url": "https://example.invalid/context", "valueString": "source context"}],
            referenceRange=[{"low": {"value": 100, "unit": "mmHg"}}, {"text": "Other range"}])
        result, _ = self.fetch([record])
        observed = result["observations"][0]
        self.assertEqual(observed["result"], record)
        self.assertEqual(observed["reference_range"], {"ranges": record["referenceRange"]})
        observed["result"]["component"].clear()
        self.assertEqual(len(record["component"]), 2)

    def test_scalar_units_must_be_present_in_dump_not_invented_from_catalog(self):
        record = observation("body-weight", valueQuantity={"value": 150, "unit": "lb"})
        record.pop("component")
        self.assertEqual(self.fetch([record], ids=("body_weight",))[0]["status"], "ok")
        for quantity in ({"value": 150}, {"value": 150, "code": "kg"},
                         {"value": 150, "unit": " "}, {"value": float("nan"), "unit": "kg"}):
            record["valueQuantity"] = quantity
            self.assertEqual(self.fetch([record], ids=("body_weight",))[0]["status"], "unavailable")
        record["valueQuantity"] = {"value": 150, "system": "http://unitsofmeasure.org", "code": "[lb_av]"}
        self.assertEqual(self.fetch([record], ids=("body_weight",))[0]["status"], "ok")

    def test_component_and_range_quantities_require_units(self):
        for field in ("component", "referenceRange"):
            record = observation()
            if field == "component":
                record[field][0]["valueQuantity"] = {"value": 134}
            else:
                record[field] = [{"low": {"value": 100}}]
            self.assertEqual(self.fetch([record])[0]["status"], "unavailable")

    def test_malformed_quantity_values_cannot_bypass_unit_validation(self):
        for quantity in ({"value": "134"}, {"value": True}, {"value": None},
                         {"unit": "mmHg"}, "134", [134]):
            with self.subTest(quantity=quantity):
                record = observation()
                record["component"][0]["valueQuantity"] = quantity
                self.assertEqual(self.fetch([record])[0]["status"], "unavailable")

    def test_latest_absolute_time_ties_and_accepted_statuses(self):
        payloads = [observation(id="old", effectiveDateTime="2026-10-18T11:00:00+02:00"),
                    observation(id="latest", status="amended", effectiveDateTime="2026-10-18T10:30:00.125Z"),
                    observation(id="tie", status="corrected", effectiveDateTime="2026-10-18T11:30:00.125+01:00"),
                    observation(id="excluded", status="entered-in-error", effectiveDateTime="2026-10-19T12:00:00Z")]
        result, _ = self.fetch(payloads)
        self.assertEqual({r["id"] for r in result["observations"]}, {"latest", "tie"})
        self.assertEqual(self.fetch([payloads[-1]])[0]["status"], "no_data")

    def test_period_and_absence_fields_if_returned_by_sdk(self):
        record = observation(effectivePeriod={"start": "2026-10-18T10:00:00Z", "end": "2026-10-18T11:00:00Z"})
        del record["effectiveDateTime"]
        result, _ = self.fetch([record])
        self.assertEqual(result["observations"][0]["measured_at"], "2026-10-18T11:00:00Z")
        del record["component"]
        record["dataAbsentReason"] = {"text": "Not measured"}
        self.assertEqual(self.fetch([record])[0]["observations"][0]["result"], record)

    def test_empty_none_and_errors_are_unavailable_not_absent_records(self):
        self.assertEqual(self.fetch()[0]["status"], "unavailable")
        from requests.exceptions import ConnectionError, SSLError, Timeout
        for error in (Timeout, ConnectionError, SSLError, ValueError, RuntimeError):
            result, _ = self.fetch(error=error("synthetic-secret"))
            self.assertEqual(result["status"], "unavailable")
            self.assertNotIn("synthetic-secret", json.dumps(result))
        self.service.provider.client.get.side_effect = None
        self.service.provider.client.get.return_value = None
        self.assertEqual(self.service.fetch("participant-001", request("blood_pressure"), [])[0]["status"], "unavailable")

    def test_wrong_patient_measurement_and_malformed_records_rejected(self):
        bad = [observation(subject={"reference": "Patient/someone-else"}),
               observation(subject={"reference": "Patient/someone-else"}, status="entered-in-error"),
               observation("heart-rate"), observation(effectiveDateTime="2026-10-18T10:00:00"),
               observation(effectiveDateTime=None), observation(code={}), observation(resourceType="Patient"), None, {}]
        for item in bad:
            result, _ = self.fetch([observation(), item])
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["observations"], [])

    def test_identity_guards_do_not_call_server(self):
        get = self.service.provider.client.get
        self.assertEqual(self.service.fetch("unmapped", request("blood_pressure"), [])[0]["status"], "unmapped_patient")
        self.assertEqual(self.service.fetch("participant-001", request("invented"), [])[0]["status"], "unsupported_measurement")
        for subject in ("other", "unknown"):
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure", subject=subject), []), [])
        get.assert_not_called()
        with self.assertRaises(ValueError):
            self.service.provider.fetch_latest(" ", ["blood_pressure"])
        get.assert_not_called()

    def test_each_participant_uses_only_their_mapping(self):
        self.service.config.patient_map = {"participant-a": "patient-a", "participant-b": "patient-b"}
        for user, patient in self.service.config.patient_map.items():
            get = self.service.provider.client.get
            get.reset_mock()
            get.return_value = records(observation(subject={"reference": f"Patient/{patient}"}))
            self.assertEqual(self.service.fetch(user, request("blood_pressure"), [])[0]["status"], "ok")
            get.assert_called_once_with("blood-pressure", subject=patient)

    def test_multiple_measurements_no_refetch_and_no_partial_success_after_failure(self):
        get = self.service.provider.client.get
        get.side_effect = [records(observation()), records(observation("heart-rate"))]
        result = self.service.fetch("participant-001", request("blood_pressure", "heart_rate"), [])
        self.assertEqual(result[0]["status"], "ok")
        self.assertEqual(self.service.fetch("participant-001", request("blood_pressure"), result), result)
        self.assertEqual(get.call_count, 2)
        get.side_effect = [records(observation()), []]
        result = self.service.fetch("participant-001", request("blood_pressure", "heart_rate"), [])[0]
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["observations"], [])

    def load_config(self):
        with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(Path(self.folder.name) / "biomarkers.json")}):
            return load_biomarker_service()

    def test_resource_types_and_codes_are_configurable_without_sdk_catalogue(self):
        path = Path(self.folder.name) / "biomarkers.json"
        settings = json.loads(path.read_text())
        settings["fhir"]["resource_types"]["blood_pressure"] = "partner-configured-type"
        settings["fhir"]["observation_codes"]["blood_pressure"] = [{"system": "partner-system", "code": "partner-code"}]
        path.write_text(json.dumps(settings))
        with patch(CLIENT):
            service = self.load_config()
        service.provider.client.get.return_value = records(observation(code={"coding": [{"system": "partner-system", "code": "partner-code"}]}))
        self.assertEqual(service.fetch("participant-001", request("blood_pressure"), [])[0]["status"], "ok")
        service.provider.client.get.assert_called_once_with("partner-configured-type", subject="synthetic-patient")
        for update in ({"observation_codes": {}}, {"resource_types": {}}, {"timeout_seconds": 10}):
            path.write_text(json.dumps({**settings, "fhir": {**settings["fhir"], **update}}))
            with self.assertRaises(ValueError):
                self.load_config()

    def test_inline_token_and_relative_token_file_constructor(self):
        credential = Path(self.folder.name) / "client.json"
        with patch(CLIENT) as client:
            self.load_config()
            self.assertEqual(client.call_args.kwargs, {"url": "https://fhir.invalid/proxy/", "token": "synthetic-secret", "is_secure": True})
        token_path = credential.parent / "token.txt"
        token_path.write_text("synthetic-file-token\n")
        for inline in ({}, {"token": None}, {"token": "synthetic-fallback"}):
            credential.write_text(json.dumps({"url": "https://fhir.invalid", "token_file": "token.txt", **inline}))
            with patch(CLIENT) as client:
                self.load_config()
                self.assertEqual(client.call_args.kwargs["token_file"], str(token_path.resolve()))
                self.assertIs(client.call_args.kwargs["is_secure"], True)
                self.assertIn("token", client.call_args.kwargs)
        token_path.unlink()
        with self.assertRaises(OSError):
            self.load_config()
        token_path.write_text("\n")
        with self.assertRaises(ValueError):
            self.load_config()

    def test_insecure_credentials_and_older_sdk_fail_at_configuration(self):
        for update in ({"url": "http://fhir.invalid"}, {"is_secure": False}, {"token": ""},
                       {"url": "https://user:pass@fhir.invalid"}, {"url": "https://fhir.invalid/?token=secret"}):
            credentials = {"url": "https://fhir.invalid", "token": "synthetic-secret", **update}
            (Path(self.folder.name) / "client.json").write_text(json.dumps(credentials))
            with self.assertRaises(ValueError):
                self.load_config()
        with patch("maya.framework.integrations.biomarkers.fhir.version", return_value="25.5.14"), self.assertRaisesRegex(ValueError, "26.10.16"):
            self.load_config()


@unittest.skipUnless(importlib.util.find_spec("libfhir"), "Requires the UCLM libfhir wheel.")
class InstalledSDKLimitationsTests(unittest.TestCase):
    """Exercise the installed public client against an in-memory HTTP boundary."""

    def setUp(self):
        from importlib.metadata import version
        from libfhir.client import FHIRProxyClient
        self.assertEqual(version("libfhir"), "26.10.16")
        self.client = FHIRProxyClient(url="https://fhir.invalid", token="synthetic-secret", is_secure=True)

    def get(self, payloads):
        with patch("requests.Session.get", return_value=response(payloads)) as transport:
            result = self.client.get("blood-pressure", subject="synthetic-patient")
        self.assertIs(transport.call_args.kwargs["verify"], True)
        self.assertNotIn("timeout", transport.call_args.kwargs)
        return result

    def test_scalar_units_are_lost_but_components_survive(self):
        scalar = observation("body-weight", valueQuantity={"value": 70, "unit": "kg", "system": "http://unitsofmeasure.org", "code": "kg"})
        scalar.pop("component")
        dumped = self.get([scalar])[0].dump()
        self.assertEqual(dumped["valueQuantity"], {"value": 70.0})
        self.assertEqual(self.get([observation()])[0].dump()["component"], observation()["component"])

    def test_actual_sdk_serialization_reaches_adapter_unit_guard(self):
        scalar = observation("body-weight", valueQuantity={"value": 70, "unit": "kg"})
        scalar.pop("component")
        with TemporaryDirectory() as folder:
            service = configured_service(folder)
            service.provider.client = self.client
            for payload, measurement, expected in ((scalar, "body_weight", "unavailable"),
                                                     (observation(), "blood_pressure", "ok")):
                with patch("requests.Session.get", return_value=response([payload])):
                    result = service.fetch("participant-001", request(measurement), [])[0]
                self.assertEqual(result["status"], expected)

    def test_documented_token_file_is_used_by_real_client(self):
        from libfhir.client import FHIRProxyClient
        with TemporaryDirectory() as folder:
            token_path = Path(folder) / "token.txt"
            token_path.write_text("synthetic-file-token\n")
            client = FHIRProxyClient(url="https://fhir.invalid", token=None,
                                     token_file=str(token_path), is_secure=True)
            with patch("requests.Session.get", return_value=response()) as transport:
                client.get("blood-pressure", subject="synthetic-patient")
            self.assertEqual(transport.call_args.kwargs["json"]["token"], "synthetic-file-token")

    def test_failure_empty_and_rejected_records_have_ambiguous_outcomes(self):
        from requests.exceptions import Timeout
        self.assertEqual(self.get([]), [])
        with patch("requests.Session.get", side_effect=Timeout("synthetic timeout")):
            self.assertEqual(self.client.get("blood-pressure", subject="synthetic-patient"), [])
        with patch("requests.Session.get", return_value=response(body={"status": "error"})):
            self.assertEqual(self.client.get("blood-pressure", subject="synthetic-patient"), [])
        with patch("requests.Session.get", return_value=response(status=403)):
            self.assertIsNone(self.client.get("blood-pressure", subject="synthetic-patient"))

    def test_timestamp_offsets_fractions_and_absence_only_are_omitted(self):
        for timestamp in ("2026-10-18T10:30:00.125Z", "2026-10-18T10:30:00+00:00"):
            self.assertEqual(self.get([observation(effectiveDateTime=timestamp)]), [])
            self.assertEqual(len(self.get([observation(), observation(id="newer", effectiveDateTime=timestamp)])), 1)
        absent = observation(dataAbsentReason={"text": "Not measured"})
        absent.pop("component")
        self.assertEqual(self.get([absent]), [])


if __name__ == "__main__":
    unittest.main()
