from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch
from concurrent.futures import Executor, Future
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from maya.framework.memory.ontology import (
    OntologyMemoryCheck,
    OntologyMemoryDecision,
    OntologyMemoryExtraction,
    OntologyMemoryRecord,
    OntologyMemoryService,
    OntologyMemoryNotReady,
    OntologyMemoryVerification,
    OntologyPropertyCheck,
)
from maya.framework.ontology.registry import OntologyRegistry


class ImmediateExecutor(Executor):
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class StubExtractor:
    def __init__(self, responses: list[Any] | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []
        self.error: Exception | None = None

    def invoke(self, value: list[Any], config: dict[str, Any]) -> OntologyMemoryExtraction:
        self.calls.append((value, config))
        if self.error is not None:
            raise self.error
        records = self.responses.pop(0) if self.responses else []
        return OntologyMemoryExtraction(should_write=bool(records), records=records)


class StubVerifier:
    def __init__(self, check_ids: tuple[str, ...]):
        self.check_ids = check_ids
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []
        self.failed_checks: set[str] = set()
        self.unsupported_properties: set[str] = set()

    def invoke(self, value: list[Any], config: dict[str, Any]) -> OntologyMemoryVerification:
        self.calls.append((value, config))
        proposed = json.loads(value[1].content)["proposed_records"]
        return OntologyMemoryVerification(
            decisions=[
                OntologyMemoryDecision(
                    record_index=item["record_index"],
                    checks=[
                        OntologyMemoryCheck(
                            check_id=check_id,
                            passed=check_id not in self.failed_checks,
                            reason="supported by test fixture",
                        )
                        for check_id in self.check_ids
                    ],
                    property_checks=[
                        OntologyPropertyCheck(
                            property_name=property_name,
                            explicitly_supported=(
                                property_name not in self.unsupported_properties
                            ),
                            reason="supported by test fixture",
                        )
                        for property_name in item["record"]["properties"]
                    ],
                )
                for item in proposed
            ]
        )


def medication_record(
    evidence: str,
    *,
    dose: str | None = "2.5 mg",
    medication_status: str | None = None,
) -> OntologyMemoryRecord:
    properties = {"medication_name": "Bisoprolol"}
    if dose:
        properties["dose"] = dose
    if medication_status:
        properties["status"] = medication_status
    return OntologyMemoryRecord(
        model_id="clinical_profile",
        binding_path="clinical_profile.medication_records",
        node_id="medication_record",
        properties=properties,
        evidence_text=evidence,
    )


def treatment_record(evidence: str) -> OntologyMemoryRecord:
    return OntologyMemoryRecord(
        model_id="cancer_history",
        binding_path="cancer_history.treatment_records",
        node_id="cancer_treatment_record",
        properties={
            "treatment_name": "Doxorubicin",
            "treatment_type": "chemotherapy",
        },
        evidence_text=evidence,
    )


class OntologyMemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = OntologyRegistry()
        self.store = InMemoryStore()
        self.extractor = StubExtractor()
        self.verifier = StubVerifier(self.registry.memory_verification_check_ids)
        self.service = OntologyMemoryService(
            model=object(),
            store=self.store,
            registry=self.registry,
            extractor_factory=lambda model: self.extractor,
            verifier_factory=lambda model: self.verifier,
            executor=ImmediateExecutor(),
            register_shutdown=False,
        )

    def schedule(self, source_text: str, model_ids: list[str]) -> bool:
        return self.service.schedule(
            user_id="user-1",
            thread_id="thread-1",
            source_message_id="message-1",
            source_text=source_text,
            model_ids=model_ids,
        )

    def records(self, user_id: str = "user-1") -> list[Any]:
        return self.store.search(self.service.memory_namespace(user_id), limit=100)

    def pending(self) -> list[Any]:
        return self.store.search(self.service.pending_namespace("user-1"), limit=100)

    def test_background_extraction_and_verification_keep_participant_privacy_context(self):
        evidence = "I take Bisoprolol 2.5 mg."
        identity = {"name": "Alice Example", "patient_id": "patient-1"}
        self.extractor.responses.append([medication_record(evidence)])
        self.service.schedule(user_id="user-1", thread_id="thread-1", source_message_id="message-1",
                              source_text=evidence, model_ids=["clinical_profile"], privacy_context=identity)
        for calls in (self.extractor.calls, self.verifier.calls):
            self.assertEqual(calls[0][1]["configurable"]["privacy_context"], identity)
            self.assertEqual(calls[0][1]["configurable"]["user_id"], "user-1")
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records("user-2"), [])
        self.assertEqual(identity, {"name": "Alice Example", "patient_id": "patient-1"})

    def test_recovered_legacy_job_loads_only_its_participants_profile(self):
        evidence = "I take Bisoprolol 2.5 mg."
        self.store.put(("profile", "user-1"), "profile", {"name": "Alice Example", "location": "York"})
        self.store.put(("profile", "user-2"), "profile", {"name": "Other Participant", "location": "Bath"})
        self.store.put(self.service.pending_namespace("user-1"), "legacy-job", {
            "type": "ontology_memory_job", "user_id": "user-1", "thread_id": "thread-1",
            "source_message_id": "message-1", "source_text": evidence,
            "selected_model_ids": ["clinical_profile"],
        })
        self.extractor.responses.append([medication_record(evidence)])
        self.service.render_context(user_id="user-1", agent_id="health_rag", model_ids=["clinical_profile"])
        for calls in (self.extractor.calls, self.verifier.calls):
            self.assertEqual(calls[0][1]["configurable"]["privacy_context"],
                             {"name": "Alice Example", "location": "York"})
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.pending(), [])

    def test_loaded_context_uses_one_record_snapshot_and_preserves_user_isolation(self):
        evidence = 'I received doxorubicin chemotherapy.'
        self.extractor.responses.append([treatment_record(evidence)])
        self.assertTrue(self.schedule(evidence, ['cancer_history']))
        with patch.object(self.service, 'load_records', wraps=self.service.load_records) as load:
            full, memory = self.service.load_context(
                user_id='user-1', agent_id='health_rag', model_ids=['cancer_history'])
        load.assert_called_once()
        self.assertIn('Doxorubicin', memory)
        self.assertIn('user_reported', memory)
        self.assertIn(memory, full)
        self.assertNotIn('source_message_id', memory)
        for user, models in [('user-2', ['cancer_history']), ('user-1', ['clinical_profile'])]:
            with self.subTest(user=user, models=models):
                full, memory = self.service.load_context(user_id=user, agent_id='health_rag', model_ids=models)
                self.assertEqual(memory, '')
                self.assertNotIn('Doxorubicin', full)

    def test_explicit_medication_assertion_is_persisted_with_provenance(self) -> None:
        evidence = "I started Bisoprolol 2.5 mg."
        self.extractor.responses.append([medication_record(evidence)])

        self.assertTrue(self.schedule(evidence, ["clinical_profile"]))

        items = self.records()
        self.assertEqual(len(items), 1)
        record = items[0].value["record"]
        self.assertEqual(record["properties"]["medication_name"], "Bisoprolol")
        self.assertEqual(record["source_message_id"], "message-1")
        self.assertEqual(record["source_type"], "user_reported")
        self.assertEqual(record["model_version"], "0.1.0-draft")
        self.assertEqual(self.pending(), [])

        messages, config = self.extractor.calls[0]
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(json.loads(messages[1].content)["source_message"]["text"], evidence)
        self.assertEqual(config["configurable"]["user_id"], "user-1")

    def test_verification_retries_only_incomplete_records(self):
        records = [medication_record("I take Bisoprolol.", dose=None),
                   treatment_record("I had Doxorubicin chemotherapy.")]
        invoke = self.verifier.invoke
        for omission in ("record", "rule", "property", "duplicate"):
            with self.subTest(omission=omission):
                self.verifier.calls.clear()

                def incomplete_once(messages, config):
                    result = invoke(messages, config)
                    if len(self.verifier.calls) == 1:
                        if omission == "record":
                            result.decisions.pop()
                        elif omission == "rule":
                            result.decisions[-1].checks.pop()
                        elif omission == "property":
                            result.decisions[-1].property_checks.pop()
                        else:
                            result.decisions.append(result.decisions[-1])
                    return result

                with patch.object(self.verifier, "invoke", side_effect=incomplete_once):
                    verified = self.service._verified_records(
                        records=records, source_text=" ".join(r.evidence_text for r in records),
                        source_message_id="message-1", model_ids=["clinical_profile", "cancer_history"], config={})
                self.assertEqual(verified, records)
                self.assertEqual(len(self.verifier.calls), 2)
                retried = json.loads(self.verifier.calls[-1][0][-1].content)["proposed_records"]
                self.assertEqual([item["record_index"] for item in retried], [1])

    def test_persistent_verification_omission_keeps_job_pending(self):
        records = [medication_record("I take Bisoprolol.", dose=None),
                   treatment_record("I had Doxorubicin chemotherapy.")]
        self.extractor.responses.append(records)
        invoke = self.verifier.invoke

        def omit_treatment(messages, config):
            result = invoke(messages, config)
            result.decisions = [d for d in result.decisions if d.record_index == 0]
            return result

        with patch.object(self.verifier, "invoke", side_effect=omit_treatment), \
                self.assertLogs("maya.framework.memory.ontology", level="WARNING"):
            self.schedule(" ".join(r.evidence_text for r in records), ["clinical_profile", "cancer_history"])
        self.assertEqual(self.records(), [])
        self.assertEqual(len(self.pending()), 1)
        self.assertEqual(len(self.verifier.calls), 2)

    def test_repeated_assertion_is_idempotent_and_correction_updates_same_record(self) -> None:
        started = "I started Bisoprolol 2.5 mg."
        stopped = "I stopped taking Bisoprolol."
        self.extractor.responses.extend(
            [
                [medication_record(started)],
                [medication_record(started)],
                [medication_record(stopped, dose=None, medication_status="stopped")],
            ]
        )

        self.schedule(started, ["clinical_profile"])
        self.schedule(started, ["clinical_profile"])
        self.service.schedule(
            user_id="user-1",
            thread_id="thread-2",
            source_message_id="message-2",
            source_text=stopped,
            model_ids=["clinical_profile"],
        )

        items = self.records()
        self.assertEqual(len(items), 1)
        record = items[0].value["record"]
        self.assertEqual(record["properties"]["status"], "stopped")
        self.assertEqual(record["source_message_id"], "message-2")
        self.assertEqual(record["evidence_text"], stopped)

    def test_cancer_treatment_uses_its_own_binding(self) -> None:
        evidence = "My chemotherapy included Doxorubicin."
        self.extractor.responses.append([treatment_record(evidence)])

        self.schedule(evidence, ["cancer_history"])

        items = self.records()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].value["model_id"], "cancer_history")
        self.assertEqual(
            items[0].value["record"]["binding_path"],
            "cancer_history.treatment_records",
        )

    def test_schema_and_evidence_violations_are_not_written(self) -> None:
        unsupported = "I started Bisoprolol."
        unsupported_record = medication_record(unsupported, dose=None)
        unsupported_record.properties["prescriber"] = "Dr Smith"
        unsupported_model = medication_record(unsupported, dose=None)
        unsupported_model.model_id = "cancer_history"
        wrong_evidence = medication_record("I started metformin.", dose=None)
        self.extractor.responses.extend(
            [
                [unsupported_record],
                [unsupported_model],
                [wrong_evidence],
            ]
        )

        self.schedule(unsupported, ["clinical_profile"])
        self.schedule(unsupported, ["clinical_profile"])
        self.schedule(unsupported, ["clinical_profile"])

        self.assertEqual(self.records(), [])
        self.assertEqual(self.pending(), [])

    def test_verifier_prunes_unsupported_optional_properties(self) -> None:
        evidence = "I started Bisoprolol 2.5 mg."
        record = medication_record(evidence, medication_status="current")
        self.extractor.responses.append([record])
        self.verifier.unsupported_properties.add("status")

        self.schedule(evidence, ["clinical_profile"])

        stored = self.records()[0].value["record"]
        self.assertEqual(
            stored["properties"],
            {"medication_name": "Bisoprolol", "dose": "2.5 mg"},
        )

    def test_failed_semantic_verification_prevents_write(self) -> None:
        evidence = "My father takes Bisoprolol."
        self.extractor.responses.append([medication_record(evidence, dose=None)])
        self.verifier.failed_checks.add("current_user_subject")

        self.schedule(evidence, ["clinical_profile"])

        self.assertEqual(self.records(), [])
        self.assertEqual(self.pending(), [])
        self.assertEqual(len(self.verifier.calls), 1)

    def test_model_without_active_binding_skips_llm_work(self) -> None:
        self.assertFalse(self.schedule("I read a clinical guideline.", ["knowledge_resource"]))
        self.assertEqual(self.extractor.calls, [])
        self.assertEqual(self.pending(), [])

    def test_loads_same_user_memory_across_threads_and_filters_models(self) -> None:
        medication = "I started Bisoprolol 2.5 mg."
        treatment = "My chemotherapy included Doxorubicin."
        self.extractor.responses.extend(
            [
                [medication_record(medication)],
                [treatment_record(treatment)],
            ]
        )
        self.schedule(medication, ["clinical_profile"])
        self.service.schedule(
            user_id="user-1",
            thread_id="another-thread",
            source_message_id="message-2",
            source_text=treatment,
            model_ids=["cancer_history"],
        )

        clinical = self.service.load_records("user-1", ["clinical_profile"])
        cancer = self.service.load_records("user-1", ["cancer_history"])
        other_user = self.service.load_records("user-2", ["clinical_profile", "cancer_history"])
        self.assertEqual([item["model_id"] for item in clinical], ["clinical_profile"])
        self.assertEqual([item["model_id"] for item in cancer], ["cancer_history"])
        self.assertEqual(other_user, [])

    def test_failed_job_remains_pending_and_is_recovered(self) -> None:
        evidence = "I started Bisoprolol 2.5 mg."
        self.extractor.error = RuntimeError("temporary extraction failure")

        with self.assertLogs("maya.framework.memory.ontology", level="WARNING"):
            self.assertTrue(self.schedule(evidence, ["clinical_profile"]))
        self.assertEqual(len(self.pending()), 1)
        self.assertEqual(self.records(), [])

        self.extractor.error = None
        self.extractor.responses.append([medication_record(evidence)])
        self.assertTrue(self.schedule(evidence, ["clinical_profile"]))

        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.pending(), [])

    def test_context_read_recovers_failed_write_without_another_user_message(self):
        evidence = "My chemotherapy included Doxorubicin."
        self.extractor.error = TimeoutError("request timed out")
        with self.assertLogs("maya.framework.memory.ontology", level="WARNING"):
            self.schedule(evidence, ["cancer_history"])
        with self.assertLogs("maya.framework.memory.ontology", level="WARNING"):
            with self.assertRaises(OntologyMemoryNotReady):
                self.service.render_context(user_id="user-1", agent_id="health_rag", model_ids=["cancer_history"])
        self.assertEqual(self.records(), [])
        self.extractor.error = None
        self.extractor.responses.append([treatment_record(evidence)])

        context = self.service.render_context(
            user_id="user-1", agent_id="health_rag", model_ids=["cancer_history"])

        self.assertIn("Doxorubicin", context)
        self.assertEqual(len(self.extractor.calls), 3)
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.pending(), [])

    def test_restart_holds_only_relevant_pending_memory_until_verified(self):
        evidence = "My chemotherapy included Doxorubicin."
        self.extractor.error = TimeoutError("request timed out")
        with self.assertLogs("maya.framework.memory.ontology", level="WARNING"):
            self.schedule(evidence, ["cancer_history"])

        class DeferredExecutor(Executor):
            def __init__(self):
                self.calls = []

            def submit(self, fn, /, *args, **kwargs):
                future = Future()
                self.calls.append((fn, args, kwargs, future))
                return future

        executor = DeferredExecutor()
        self.registry.runtime["memory_policy"]["recovery_wait_seconds"] = 0
        restarted = OntologyMemoryService(
            model=object(), store=self.store, registry=self.registry,
            extractor_factory=lambda model: self.extractor,
            verifier_factory=lambda model: self.verifier,
            executor=executor, register_shutdown=False)
        with self.assertRaises(OntologyMemoryNotReady):
            restarted.render_context(user_id="user-1", agent_id="health_rag", model_ids=["cancer_history"])
        self.assertEqual(self.records(), [])
        self.assertNotIn(evidence, restarted.render_context(
            user_id="user-1", agent_id="health_rag", model_ids=["clinical_profile"]))
        self.assertNotIn(evidence, restarted.render_context(
            user_id="user-2", agent_id="health_rag", model_ids=["cancer_history"]))
        with self.assertRaises(OntologyMemoryNotReady):
            restarted.render_context(user_id="user-1", agent_id="health_rag", model_ids=["cancer_history"])
        self.assertEqual(len(executor.calls), 1)

        # Complete the original validation pipeline; pending context then disappears.
        self.extractor.error = None
        self.extractor.responses.append([treatment_record(evidence)])
        fn, args, kwargs, future = executor.calls[0]
        future.set_result(fn(*args, **kwargs))
        context = restarted.render_context(user_id="user-1", agent_id="health_rag", model_ids=["cancer_history"])
        self.assertEqual(len(self.records()), 1)
        self.assertIn("Doxorubicin", context)

    def test_recovery_keeps_rejected_statement_out_of_context(self):
        evidence = "My father takes Bisoprolol."
        self.extractor.error = TimeoutError("request timed out")
        with self.assertLogs("maya.framework.memory.ontology", level="WARNING"):
            self.schedule(evidence, ["clinical_profile"])
        self.extractor.error = None
        self.extractor.responses.append([medication_record(evidence, dose=None)])
        self.verifier.failed_checks.add("current_user_subject")
        context = self.service.render_context(user_id="user-1", agent_id="health_rag", model_ids=["clinical_profile"])
        self.assertNotIn(evidence, context)
        self.assertEqual(self.records(), [])
        self.assertEqual(self.pending(), [])

    def test_schedule_returns_before_slow_extraction_finishes(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class SlowExtractor:
            def invoke(self, value: list[Any], config: dict[str, Any]) -> OntologyMemoryExtraction:
                entered.set()
                release.wait(timeout=2)
                return OntologyMemoryExtraction()

        service = OntologyMemoryService(
            model=object(),
            store=InMemoryStore(),
            registry=self.registry,
            extractor_factory=lambda model: SlowExtractor(),
            verifier_factory=lambda model: self.verifier,
            register_shutdown=False,
        )
        started = time.monotonic()
        scheduled = service.schedule(
            user_id="user-1",
            thread_id="thread-1",
            source_message_id="message-1",
            source_text="I started Bisoprolol.",
            model_ids=["clinical_profile"],
        )
        elapsed = time.monotonic() - started

        try:
            self.assertTrue(scheduled)
            self.assertLess(elapsed, 0.5)
            self.assertTrue(entered.wait(timeout=1))
            self.registry.runtime["memory_policy"]["recovery_wait_seconds"] = 0.01
            started = time.monotonic()
            with self.assertRaises(OntologyMemoryNotReady):
                service.render_context(user_id="user-1", agent_id="health_rag", model_ids=["clinical_profile"])
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            release.set()
            service.close()

    def test_context_wait_wakes_when_verification_finishes(self):
        release = threading.Event()
        evidence = "My chemotherapy included Doxorubicin."

        class SlowExtractor:
            def invoke(self, value, config):
                release.wait(timeout=2)
                return OntologyMemoryExtraction(should_write=True, records=[treatment_record(evidence)])

        service = OntologyMemoryService(
            model=object(), store=self.store, registry=self.registry,
            extractor_factory=lambda model: SlowExtractor(),
            verifier_factory=lambda model: self.verifier, register_shutdown=False)
        service.schedule(user_id="user-1", thread_id="thread", source_message_id="message",
                         source_text=evidence, model_ids=["cancer_history"])
        timer = threading.Timer(0.02, release.set)
        timer.start()
        try:
            context = service.render_context(user_id="user-1", agent_id="health_rag", model_ids=["cancer_history"])
            self.assertIn("Doxorubicin", context)
            self.assertEqual(len(self.records()), 1)
            self.assertEqual(self.pending(), [])
        finally:
            release.set()
            timer.join()
            service.close()


if __name__ == "__main__":
    unittest.main()
