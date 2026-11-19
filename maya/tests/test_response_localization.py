"""Localization boundaries, cache isolation and outage behavior without network calls."""

import json
from unittest import TestCase
from unittest.mock import Mock, patch

from maya.tests.model_stubs import model_call_stub

from pydantic import ValidationError

from maya.framework.conversation import responses
from maya.framework.supervisor import run_with_vad as runner


class TranslationModel:
    """A translation boundary stub; it does not stand in for a language-quality test."""

    azure_endpoint = "https://localization.invalid"
    deployment_name = "test-model"
    openai_api_version = "test-version"

    def __init__(self):
        self.calls = []
        self.error = None

    def with_structured_output(self, schema, *, strict):
        assert strict is True

        def translate(messages):
            self.calls.append(messages)
            if self.error:
                raise self.error
            return schema.model_validate({"texts": ["Texte traduit : " + value
                                                   for value in json.loads(messages[-1].content)]})
        return Mock(invoke=translate)


class ResponseLocalizationTests(TestCase):
    def setUp(self):
        self.enterContext(patch("maya.framework.integrations.models.model_call", side_effect=model_call_stub))
        self.enterContext(patch("maya.framework.conversation.responses.model_call", side_effect=model_call_stub))
        responses._translated_messages.cache_clear()
        self.addCleanup(responses._translated_messages.cache_clear)
        self.model = TranslationModel()
        self.enterContext(patch.object(responses, "get_chat_model", return_value=self.model))
        self.source = responses.load_response_policy()

    def test_source_language_preserves_exact_configuration_without_model(self):
        self.assertEqual(responses.response_policy(self.source.source_language), self.source)
        self.assertEqual(self.model.calls, [])

    def test_language_review_checks_translation_fidelity_only_with_a_source(self):
        from system_test import language_review
        samples = [
            {"id": "speech", "language_samples": ["Answer plus the configured disclaimer."]},
            {"id": "translated_notice", "language_samples": ["Changed meaning."], "translation_sources": ["Original meaning."]},
            {"id": "wrong_language", "language_samples": ["Bonjour."]},
        ]
        judge = Mock()
        judge.with_structured_output.return_value.invoke.return_value = Mock(findings=[
            Mock(sample_id=sample["id"] + "/0", matches_language=sample["id"] != "wrong_language",
                 preserves_source_meaning=False, explanation="Controlled reviewer finding.")
            for sample in samples
        ])
        with patch.dict("os.environ", {}), patch("maya.framework.integrations.models.get_chat_model", return_value=judge):
            results = language_review(samples, "en", judge_model="test-model")
        self.assertEqual([row["status"] for row in results], ["PASS", "FAIL", "FAIL"])
        self.assertNotIn("model_assisted_translation_fidelity", results[0]["checks"])
        self.assertFalse(results[1]["checks"]["model_assisted_translation_fidelity"])
        self.assertFalse(results[2]["checks"]["model_assisted_language_match"])
        self.assertFalse(results[0]["translation_review_applicable"])
        self.assertFalse(results[0]["judge_translation_fidelity"])  # Raw judgement remains visible.

    def test_language_review_keeps_default_and_session_targets_separate(self):
        from system_test import language_review
        rows = [{"id": "session", "language_samples": ["Bonjour."] * 12},
                {"id": "default", "language_samples": ["Hello."], "expected_language": "en"}]
        batches = []

        def review(messages):
            batch = json.loads(messages[-1].content)
            batches.append(batch)
            return Mock(findings=[Mock(sample_id=sample["sample_id"], matches_language=True,
                                       preserves_source_meaning=True, explanation="Controlled finding.")
                                  for sample in batch])

        judge = Mock()
        judge.with_structured_output.return_value.invoke.side_effect = review
        with patch.dict("os.environ", {}), patch("maya.framework.integrations.models.get_chat_model", return_value=judge):
            results = language_review(rows, "fr", judge_model="test-model")
        self.assertEqual(len(results), 13)
        self.assertEqual(len({row["id"] for row in results}), 13)
        self.assertTrue(all(len({sample["language"] for sample in batch}) == 1 for batch in batches))
        targets = {sample["sample_id"]: sample["language"] for batch in batches for sample in batch}
        self.assertEqual(targets["default/0"], "English")
        self.assertEqual(targets["session/0"], "French")

    def test_only_user_facing_policy_fields_are_translated(self):
        result = responses.response_policy("fr")
        fields = json.loads(self.model.calls[0][-1].content)
        self.assertEqual(fields, [getattr(self.source, key) for key in responses.ResponseMessages.model_fields])
        self.assertEqual(result.privacy_notice_url, self.source.privacy_notice_url)
        self.assertEqual(result.health_output_instruction, self.source.health_output_instruction)
        self.assertEqual(result.health_processing_delay_seconds, self.source.health_processing_delay_seconds)
        self.assertIn("French", self.model.calls[0][0].content)
        self.assertNotEqual(result.health_disclaimer, self.source.health_disclaimer)

    def test_cached_notices_remain_available_during_model_outage(self):
        translated = responses.response_policy("fr")
        self.model.error = TimeoutError("Model unavailable")
        self.assertEqual(responses.response_policy("fr"), translated)
        self.assertEqual(len(self.model.calls), 1)

    def test_cold_translation_failure_is_not_cached_or_replaced_with_english(self):
        self.model.error = TimeoutError("Model unavailable")
        with self.assertRaises(TimeoutError):
            responses.response_policy("fr")
        self.model.error = None
        self.assertNotEqual(responses.response_policy("fr").health_disclaimer, self.source.health_disclaimer)
        self.assertEqual(len(self.model.calls), 2)

    def test_language_and_model_switches_do_not_reuse_wrong_translation(self):
        responses.response_policy("fr")
        responses.response_policy("fi")
        responses.response_policy("fr")
        self.assertEqual(len(self.model.calls), 2)
        self.model.deployment_name = "another-model"
        responses.response_policy("fr")
        self.assertEqual(len(self.model.calls), 3)

    def test_changed_partner_wording_invalidates_translation(self):
        responses.response_policy("fr")
        revised = self.source.model_copy(update={"health_disclaimer": "Different partner wording."})
        with patch.object(responses, "load_response_policy", return_value=revised):
            self.assertIn("Different partner wording.", responses.response_policy("fr").health_disclaimer)
        self.assertEqual(len(self.model.calls), 2)

    def test_source_language_is_configured_not_an_english_branch(self):
        revised = self.source.model_copy(update={"source_language": "fr"})
        with patch.object(responses, "load_response_policy", return_value=revised):
            self.assertEqual(responses.response_policy("fr"), revised)
            self.assertEqual(self.model.calls, [])
            responses.response_policy("en")
            self.assertEqual(len(self.model.calls), 1)

    def test_unknown_language_is_rejected_without_translation(self):
        for language in ("unknown", "", " "):
            with self.subTest(language=language), self.assertRaises(ValidationError):
                responses.response_policy(language)
        self.assertEqual(self.model.calls, [])

    def test_new_language_uses_configuration_without_a_code_branch(self):
        from maya.framework.conversation import personalization
        configured = responses.RESPONSE_LANGUAGES.model_copy(update={
            "languages": {**responses.RESPONSE_LANGUAGES.languages, "nl": "Dutch"}})
        with patch.object(responses, "RESPONSE_LANGUAGES", configured), \
                patch.object(personalization, "RESPONSE_LANGUAGES", configured):
            responses.response_policy("nl")
        self.assertIn("Target response language: Dutch", self.model.calls[0][0].content)

    def test_incomplete_or_blank_translation_cannot_remove_a_notice(self):
        for invalid in ({}, {"texts": [" " for _ in responses.ResponseMessages.model_fields]}, {"texts": ["Incomplete"]}):
            with self.subTest(invalid=bool(invalid)), patch.object(
                    self.model, "with_structured_output", return_value=Mock(invoke=Mock(return_value=invalid))):
                with self.assertRaises(ValueError):
                    responses.response_policy("fr")

    def test_display_speech_and_references_share_locale_without_translating_evidence(self):
        answer = "Doxorubicin — 116/67 mmHg; 49 bpm."
        citation = {"title": "Original source title", "source_path": "/private/paper.pdf", "page_number": 4}
        result = responses.health_presentation(answer, [citation, {}], "fr")
        policy = responses.response_policy("fr")
        self.assertEqual(result["answer_text"], answer)
        self.assertEqual(result["references"][0]["title"], citation["title"])
        self.assertIn(policy.unknown_source_label, result["display_text"])
        self.assertIn(policy.page_label + " 5", result["display_text"])
        self.assertEqual(result["speech_text"], answer + "\n\n" + policy.health_disclaimer)
        self.assertNotIn(citation["title"], result["speech_text"])
        source = self.model.calls[0][-1].content
        self.assertNotIn(answer, source)
        self.assertNotIn(citation["title"], source)
        self.assertNotIn("/private/", result["display_text"])

    def test_opening_keeps_privacy_url_outside_translation_and_speech(self):
        policy = self.source.model_copy(update={"privacy_notice_url": "https://example.org/privacy"})
        with patch.object(responses, "load_response_policy", return_value=policy):
            result = responses.opening_response("fr")
        self.assertIn(str(policy.privacy_notice_url), result.display_text)
        self.assertNotIn(str(policy.privacy_notice_url), result.speech_text)
        self.assertNotIn(str(policy.privacy_notice_url), self.model.calls[0][-1].content)

    def test_no_evidence_middleware_uses_explicit_language(self):
        from maya.agents.health_rag.agent import RetrieveLocalLiterature
        from maya.tests.test_health_rag_retrieval import StubRetriever
        middleware = RetrieveLocalLiterature(StubRetriever([]), None)
        result = middleware.before_model({"standalone_question": "Unmatched topic", "response_language": "fr"}, None)
        self.assertEqual(result["jump_to"], "end")
        self.assertNotEqual(result["messages"][0].content, middleware.settings.messages.no_evidence)
        self.assertIn("French", self.model.calls[0][0].content)

    def test_synthetic_notice_only_labels_actual_synthetic_observations(self):
        from maya.framework.integrations.biomarkers import BiomarkerConfig, BiomarkerService
        service = BiomarkerService(BiomarkerConfig(
            provider="dummy", catalog={}, patient_map={}, accepted_statuses=[],
            request_instruction="Request", context_instruction="Context", readout_instruction="Readout",
            synthetic_notice="Demonstration data only."), None)
        for results in ([], [{"synthetic": False, "observations": [{}]}],
                        [{"synthetic": True, "observations": []}]):
            self.assertEqual(service.label_answer("116/67 mmHg", results, "fr"), "116/67 mmHg")
        self.assertEqual(self.model.calls, [])
        result = service.label_answer("116/67 mmHg", [{"synthetic": True, "observations": [{}]}], "fr")
        self.assertTrue(result.startswith("Texte traduit"))
        self.assertTrue(result.endswith("116/67 mmHg"))

    def test_runner_does_not_accept_input_if_initial_localization_fails(self):
        self.model.error = TimeoutError("Model unavailable")
        with patch.object(runner.os, "chdir"), patch.object(runner.atexit, "register"), \
                patch.object(runner, "acquire_single_instance_lock", return_value=Mock()), \
                patch.object(runner, "read_text") as read, \
                patch.object(runner, "build_supervisor") as build:
            with self.assertRaises(TimeoutError):
                runner.main(runner.SessionConfig(user_id="localization-test", preferences={"response_language": "fr"}))
        read.assert_not_called()
        build.assert_not_called()

    def test_runner_progress_and_failure_use_prepared_locale_during_outage(self):
        import httpx
        from threading import Event
        from openai import APIConnectionError
        announced = Event()
        spoken = []
        policy = self.source.model_copy(update={"health_processing_delay_seconds": 0})

        def speak(tts, mirror, text, **kwargs):
            spoken.append(text)
            announced.set()

        def stream(*args, **kwargs):
            self.model.error = AssertionError("No translation call allowed during outage")
            kwargs["on_health_processing"]()
            self.assertTrue(announced.wait(5))
            raise APIConnectionError(request=httpx.Request("POST", "https://model.invalid"))

        with patch.object(responses, "load_response_policy", return_value=policy), \
                patch.object(runner.os, "chdir"), patch.object(runner.atexit, "register"), \
                patch.object(runner, "acquire_single_instance_lock", return_value=Mock()), \
                patch.object(runner, "build_supervisor"), \
                patch.object(runner, "init_phyxio_io", return_value=None), \
                patch.object(runner, "init_asr", return_value=(None, None)), \
                patch.object(runner, "init_tts"), \
                patch.object(runner, "read_text", side_effect=["Question", None]), \
                patch.object(runner, "speak_output", side_effect=speak), \
                patch.object(runner, "stream_updates", side_effect=stream), patch("builtins.print"):
            runner.main(runner.SessionConfig(user_id="localization-test", preferences={"response_language": "fr"}))
        self.assertEqual(spoken, ["Texte traduit : " + policy.health_processing_status,
                                  "Texte traduit : " + policy.request_unavailable_text])
        self.assertEqual(len(self.model.calls), 1)
