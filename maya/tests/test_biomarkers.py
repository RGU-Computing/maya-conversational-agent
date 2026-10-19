"""Provider boundary and identity isolation; all records are synthetic."""

import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from maya.framework.integrations.biomarkers import BiomarkerRequest, load_biomarker_service


EXAMPLE = Path(__file__).resolve().parents[1] / "framework/integrations/biomarkers/config.example.json"


def demo_service():
    with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": str(EXAMPLE)}):
        return load_biomarker_service()


def request(*ids, subject="current_user"):
    return BiomarkerRequest(subject=subject, measurement_ids=list(ids))


class BiomarkerTests(unittest.TestCase):
    def setUp(self):
        self.service = demo_service()

    def fetch(self, *ids, user="participant-001", previous=()):
        return self.service.fetch(user, request(*ids), list(previous))

    def test_latest_per_measurement_preserves_components_units_dates_and_source(self):
        result = self.fetch("blood_pressure", "blood_glucose")[0]
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["synthetic"])
        self.assertEqual({r["id"] for r in result["observations"]}, {"bp-latest", "glucose-latest"})
        bp = next(r for r in result["observations"] if r["id"] == "bp-latest")
        self.assertEqual(bp["result"]["components"][0], {"label": "Systolic", "value": 134, "unit": "mmHg"})
        self.assertTrue(bp["measured_at"].startswith("2026-10-10"))
        self.assertTrue(bp["source"])
        self.assertNotIn("patient_id", bp)

    def test_no_fallback_patient_and_no_lookup_for_other_person(self):
        self.assertEqual(self.fetch("blood_pressure", user="someone-else")[0]["status"], "unmapped_patient")
        for subject in ("other", "unknown"):
            self.assertEqual(self.service.fetch("participant-001", request("blood_pressure", subject=subject), []), [])

    def test_attempted_ids_are_not_fetched_again_and_additional_ids_are_allowed(self):
        with patch.object(self.service.provider, "fetch_latest", wraps=self.service.provider.fetch_latest) as fetch:
            results = self.fetch("blood_pressure", "blood_pressure")
            results = self.fetch("blood_pressure", previous=results)
            self.assertEqual(fetch.call_count, 1)
            results = self.fetch("blood_pressure", "blood_glucose", previous=results)
            self.assertEqual(fetch.call_args.args[1], ["blood_glucose"])
            self.assertEqual(len(results), 2)

    def test_no_data_failure_and_unsupported_request_are_distinct(self):
        self.service.config.catalog["not_recorded"] = "Unrecorded demo measure"
        self.assertEqual(self.fetch("not_recorded")[0]["status"], "no_data")
        self.assertEqual(self.fetch("invented")[0]["status"], "unsupported_measurement")
        with patch.object(self.service.provider, "fetch_latest", side_effect=TimeoutError):
            self.assertEqual(self.fetch("blood_pressure")[0]["status"], "unavailable")

    def test_provider_replacement_cannot_return_another_patients_record(self):
        record = self.service.provider.records[0].model_copy(update={"patient_id": "other"})
        self.service.provider = Mock(synthetic=False)
        self.service.provider.fetch_latest.return_value = [record]
        result = self.fetch("blood_pressure")[0]
        self.assertEqual(result["status"], "invalid_response")
        self.assertEqual(result["observations"], [])

    def test_latest_ties_are_kept_and_invalid_status_is_excluded(self):
        provider = self.service.provider
        latest = next(r for r in provider.records if r.id == "bp-latest")
        provider.records.extend([latest.model_copy(update={"id": "tied"}),
                                 latest.model_copy(update={"id": "error", "status": "entered-in-error"})])
        ids = {r["id"] for r in self.fetch("blood_pressure")[0]["observations"]}
        self.assertEqual(ids, {"bp-latest", "tied"})

    def test_default_provider_is_disabled_and_labels_only_actual_synthetic_results(self):
        with patch.dict(os.environ, {"MAYA_BIOMARKER_CONFIG": ""}):
            service = load_biomarker_service()
        self.assertIsNone(service.provider)
        self.assertEqual(self.service.label_answer("Answer", []), "Answer")
        results = self.fetch("blood_pressure")
        self.assertEqual(self.service.label_answer("Answer", results), "Answer")
        self.assertTrue(results[0]["synthetic"])
        self.service.config.synthetic_notice = service.config.synthetic_notice
        self.assertIn(self.service.config.synthetic_notice,
                      self.service.label_answer("Answer", results))


if __name__ == "__main__":
    unittest.main()
