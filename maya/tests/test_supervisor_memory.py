"""Exercise real graph/store behavior with controlled routing and extraction decisions."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from maya.tests.model_stubs import model_call_stub

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from maya.framework.conversation.personalization import SessionConfig


class SupervisorMemoryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("maya.framework.integrations.models.model_call", side_effect=model_call_stub))
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}))
        name = "maya.framework.supervisor._memory_test_factory"
        path = Path(__file__).resolve().parents[1] / "framework/supervisor/factory.py"
        spec = importlib.util.spec_from_file_location(name, path)
        self.factory = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.factory
        self.addCleanup(sys.modules.pop, name, None)
        self.model = Mock()
        with patch("maya.framework.integrations.models.get_chat_model", return_value=self.model):
            spec.loader.exec_module(self.factory)
        self.router = Mock()
        self.guard = Mock()
        self.guard.invoke.return_value = self.factory.HealthRouteGuardDecision(route_type="direct_answer")
        self.verifier = Mock()
        self.verifier.invoke.return_value = self.factory.MemoryVerification(supported=True)
        self.model.with_structured_output.side_effect = lambda schema, **kwargs: {
            self.factory.SupervisorRouteDecision: self.router,
            self.factory.HealthRouteGuardDecision: self.guard,
            self.factory.MemoryVerification: self.verifier,
        }[schema]
        self.model.invoke.return_value = AIMessage(content="Response from the normal model.")
        self.memory_extractor = Mock()
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate()
        self.profile_extractor = Mock()
        self.profile_extractor.invoke.return_value = self.factory.ProfileUpdate()
        self.enterContext(patch.object(self.factory, "memory_extractor", self.memory_extractor))
        self.enterContext(patch.object(self.factory, "profile_extractor", self.profile_extractor))
        self.store = InMemoryStore()
        ontology = Mock()
        ontology.load_context.return_value = ("", "")
        ontology.schedule.return_value = False
        self.enterContext(patch.object(self.factory, "OntologyMemoryService", return_value=ontology))
        self.enterContext(patch.object(self.factory, "get_postgres_memory", return_value=(self.store, InMemorySaver())))
        exercise = RunnableLambda(Mock(side_effect=AssertionError("Unexpected exercise dispatch")), name="phyxio_exercise_agent")
        self.enterContext(patch.object(self.factory, "_load_agents", return_value=([exercise], {})))
        self.graph = self.factory.build_supervisor()
        self.route("direct_answer")

    def route(self, name):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
            route_type=name, selected_ontology_models=[])

    def turn(self, text, *, user="user-1", language="en", history=()):
        config = SessionConfig(user_id=user, preferences={"response_language": language}).graph_config()
        return self.graph.invoke({"messages": [*history, HumanMessage(content=text, id="current-message")]}, config)

    def records(self, kind="memory", user="user-1"):
        return {item.key: item.value for item in self.store.search((kind, user))}

    def test_recall_loads_only_current_user_and_never_writes(self):
        self.store.put(("memory", "user-1"), "original", {"content": "User felt nervous."})
        before = self.records()
        self.turn("How did I feel?")
        self.assertIn("User felt nervous.", self.model.invoke.call_args.args[0][0].content)
        self.assertEqual(self.records(), before)
        self.turn("How did I feel?", user="user-2")
        self.assertNotIn("User felt nervous.", self.model.invoke.call_args.args[0][0].content)
        self.assertEqual(self.records(user="user-2"), {})
        self.memory_extractor.invoke.assert_not_called()
        self.profile_extractor.invoke.assert_not_called()

    def test_route_guard_receives_the_writable_ontology_catalog(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
            route_type="direct_answer",
            selected_ontology_models=["clinical_profile"])
        self.turn("I take my prescribed medication.")
        catalog = self.factory.get_ontology_registry().catalog_for_agent("supervisor")
        prompt = self.guard.invoke.call_args.args[0][0].content
        self.assertIn(catalog, prompt)
        self.assertIn("writable binding", prompt)

    def test_saved_identity_is_forwarded_to_router_guard_and_reply_per_participant(self):
        profile = self.factory.Profile(name="Alice Example", location="York").model_dump()
        self.store.put(("profile", "user-1"), "profile", profile)
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
            route_type="direct_answer", selected_ontology_models=["clinical_profile"])
        result = self.turn("Hello")
        self.assertEqual(result["privacy_context"]["name"], "Alice Example")
        for target in (self.router, self.guard, self.model):
            config = target.invoke.call_args.kwargs["config"]["configurable"]
            self.assertEqual(config["user_id"], "user-1")
            self.assertEqual(config["privacy_context"]["name"], "Alice Example")
        self.turn("Hello", user="user-2")
        self.assertIsNone(self.router.invoke.call_args.kwargs["config"]["configurable"]["privacy_context"]["name"])
        self.assertEqual(self.records("profile"), {"profile": profile})

    def test_profile_recall_uses_profile_even_without_ontology_records(self):
        profile = self.factory.Profile(name="Sam", location="York").model_dump()
        self.store.put(("profile", "user-1"), "profile", profile)
        self.turn("¿Cómo me llamo y dónde vivo?", language="es")
        prompt = self.model.invoke.call_args.args[0][0].content
        self.assertIn('"name": "Sam"', prompt)
        self.assertIn('"location": "York"', prompt)
        self.assertIn("<user_profile>", prompt)
        self.assertEqual(self.records("profile"), {"profile": profile})
        self.turn("¿Cómo me llamo y dónde vivo?", user="user-2", language="es")
        self.assertNotIn('"name": "Sam"', self.model.invoke.call_args.args[0][0].content)
        self.profile_extractor.invoke.assert_not_called()

    def test_write_route_with_empty_user_message_does_not_extract_or_save(self):
        for route in ("general_memory", "user_profile"):
            with self.subTest(route=route):
                self.route(route)
                self.turn("")
        self.memory_extractor.invoke.assert_not_called()
        self.profile_extractor.invoke.assert_not_called()
        self.assertEqual(self.records(), {})
        self.assertEqual(self.records("profile"), {})

    def test_ontology_selection_does_not_override_explicit_memory_write(self):
        source = "I visited Bath yesterday."
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[self.factory.Memory(content=source)])
        for selected in ([], ["clinical_profile"]):
            with self.subTest(selected=selected):
                self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
                    route_type="general_memory", selected_ontology_models=selected)
                result = self.turn(source)
                self.assertEqual(result["route_type"], "general_memory")
                self.assertEqual(next(iter(self.records().values()))["content"], source)

    def test_empty_extraction_cannot_create_memory_despite_wrong_route(self):
        question = "How did I say I felt before yesterday's appointment?"
        for route, kind in (("general_memory", "memory"), ("user_profile", "profile")):
            with self.subTest(route=route):
                self.route(route)  # Deliberately faulty write route for a recall question.
                self.turn(question)
                self.assertEqual(self.records(kind), {})
                response_context = json.loads(self.model.invoke.call_args.args[0][-1].content)
                self.assertFalse(response_context["saved_this_turn"])

    def test_extraction_sees_latest_user_text_not_assistant_or_old_history(self):
        source = "I went to the park."
        self.route("general_memory")
        self.turn(source, history=[HumanMessage(content="Old information."), AIMessage(content="You live in Paris.")])
        messages = self.memory_extractor.invoke.call_args.args[0]
        payload = json.loads(messages[-1].content)
        self.assertEqual(payload["latest_user_message"], source)
        self.assertNotIn("Paris", str(messages))
        self.assertNotIn("Old information", str(messages))

    def test_supported_memory_has_source_and_localized_reply_instructions(self):
        source = "Olin hermostunut ennen eilistä vastaanottoa."
        self.route("general_memory")
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[
            self.factory.Memory(content=source)])
        self.turn(source, language="fi")
        record = next(iter(self.records().values()))
        self.assertEqual(record["source_message_id"], "current-message")
        self.assertEqual(record["content"], source)
        self.assertEqual(record["source_text"], source)
        messages = self.model.invoke.call_args.args[0]
        self.assertIn("Finnish", messages[0].content)
        self.assertEqual(json.loads(messages[-1].content)["saved_this_turn"], [record])

    def test_unsupported_fact_and_foreign_record_id_are_rejected(self):
        source = "I visited Bath."
        self.route("general_memory")
        self.store.put(("memory", "user-2"), "foreign", {"content": "Other participant."})
        for evidence, key in (("I visited Paris.", None), ("", None), (source, "foreign")):
            with self.subTest(evidence=evidence, key=key):
                self.verifier.invoke.return_value = self.factory.MemoryVerification(supported=evidence == source)
                self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[
                    self.factory.Memory(content=evidence, memory_id=key)])
                self.turn(source)
                self.assertEqual(self.records(), {})
        self.assertEqual(self.records(user="user-2"), {"foreign": {"content": "Other participant."}})

    def test_correction_updates_existing_memory_and_duplicate_is_noop(self):
        self.store.put(("memory", "user-1"), "trip", {"content": "User visited Bristol."})
        source = "Correction: I visited Bath."
        self.route("general_memory")
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[
            self.factory.Memory(content=source, memory_id="trip")])
        self.turn(source)
        self.assertEqual(list(self.records()), ["trip"])
        self.assertEqual(self.records()["trip"]["content"], source)
        before = self.records()
        self.turn(source)
        self.assertEqual(self.records(), before)
        self.assertEqual(json.loads(self.model.invoke.call_args.args[0][-1].content)["saved_this_turn"], [])

    def test_profile_noop_and_invalid_evidence_preserve_existing_profile(self):
        original = self.factory.Profile(name="Sam", location="York").model_dump()
        self.store.put(("profile", "user-1"), "existing-profile", original)
        source = "My name is Sam."
        self.route("user_profile")
        self.verifier.invoke.return_value = self.factory.MemoryVerification(supported=False)
        for update in (self.factory.ProfileUpdate(),
                       self.factory.ProfileUpdate(profile=self.factory.Profile(name="Wrong")),
                       self.factory.ProfileUpdate(profile=self.factory.Profile(**original))):
            with self.subTest(update=update):
                self.profile_extractor.invoke.return_value = update
                self.turn(source)
                self.assertEqual(self.records("profile"), {"existing-profile": original})
                self.assertFalse(json.loads(self.model.invoke.call_args.args[0][-1].content)["saved_this_turn"])

    def test_profile_correction_uses_existing_key_and_current_language(self):
        self.store.put(("profile", "user-1"), "legacy-profile-id", self.factory.Profile(name="Sam", location="York").model_dump())
        source = "I now live in Bath."
        self.route("user_profile")
        self.profile_extractor.invoke.return_value = self.factory.ProfileUpdate(
            profile=self.factory.Profile(name="Sam", location="Bath"))
        self.turn(source, language="fr")
        records = self.records("profile")
        self.assertEqual(list(records), ["legacy-profile-id"])
        self.assertEqual(records["legacy-profile-id"]["name"], "Sam")
        self.assertEqual(records["legacy-profile-id"]["location"], "Bath")
        self.assertIn("French", self.model.invoke.call_args.args[0][0].content)
        proposal = json.loads(self.verifier.invoke.call_args.args[0][-1].content)["proposed_update"]
        self.assertEqual(proposal, {"location": "Bath"})

    def test_new_profile_verifies_asserted_fields_without_unknown_schema_defaults(self):
        self.route("user_profile")
        self.profile_extractor.invoke.return_value = self.factory.ProfileUpdate(
            profile=self.factory.Profile(name="Sam", location="York"))
        self.turn("Mă numesc Sam și locuiesc în York.", language="ro")
        proposal = json.loads(self.verifier.invoke.call_args.args[0][-1].content)["proposed_update"]
        self.assertEqual(proposal, {"name": "Sam", "location": "York"})
        self.assertEqual(self.records("profile")["profile"]["location"], "York")

    def test_empty_new_profile_does_not_verify_or_write(self):
        self.route("user_profile")
        self.profile_extractor.invoke.return_value = self.factory.ProfileUpdate(profile=self.factory.Profile())
        self.turn("Do you know my job?")
        self.verifier.invoke.assert_not_called()
        self.assertEqual(self.records("profile"), {})

    def test_profile_field_removal_requires_verification_and_keeps_existing_key(self):
        self.route("user_profile")
        original = self.factory.Profile(job="Engineer").model_dump()
        self.store.put(("profile", "user-1"), "existing", original)
        self.profile_extractor.invoke.return_value = self.factory.ProfileUpdate(profile=self.factory.Profile())
        self.verifier.invoke.return_value = self.factory.MemoryVerification(supported=False)
        self.turn("Please clear my saved job.")
        self.assertEqual(self.records("profile"), {"existing": original})
        proposal = json.loads(self.verifier.invoke.call_args.args[0][-1].content)["proposed_update"]
        self.assertEqual(proposal, {"job": None})
        self.verifier.invoke.return_value = self.factory.MemoryVerification(supported=True)
        self.turn("Please clear my saved job.")
        self.assertEqual(self.records("profile"), {"existing": self.factory.Profile().model_dump()})

    def test_store_failure_does_not_produce_success_acknowledgement(self):
        source = "I visited Bath."
        self.route("general_memory")
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[
            self.factory.Memory(content=source)])
        with patch.object(InMemoryStore, "put", side_effect=OSError("Test storage failure")):
            with self.assertRaises(OSError):
                self.turn(source)
        self.model.invoke.assert_not_called()
        self.assertEqual(self.records(), {})

    def test_verification_rejects_question_even_when_router_and_extractor_propose_write(self):
        question = "How did I say I felt before yesterday's appointment?"
        self.route("general_memory")
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[
            self.factory.Memory(content="User felt nervous before an appointment.")])
        self.verifier.invoke.return_value = self.factory.MemoryVerification(supported=False)
        self.turn(question)
        self.assertEqual(self.records(), {})
        self.assertFalse(json.loads(self.model.invoke.call_args.args[0][-1].content)["saved_this_turn"])

    def test_verification_error_cannot_save_a_partial_batch(self):
        self.route("general_memory")
        self.memory_extractor.invoke.return_value = self.factory.MemoryUpdate(memories=[
            self.factory.Memory(content="User visited Bath."), self.factory.Memory(content="User enjoyed painting.")])
        self.verifier.invoke.side_effect = [self.factory.MemoryVerification(supported=True), ValueError("Invalid structured verification")]
        with self.assertRaises(ValueError):
            self.turn("I visited Bath and enjoyed painting.")
        self.assertEqual(self.records(), {})
        self.model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
