"""Exercise dispatch and supervisor wiring without Azure, PostgreSQL, audio or ICE.

The real exercise node, tool wrappers and supervisor graph run against service and
model doubles. These tests validate execution contracts, not LLM understanding or
the Linux mirror's transport/protocol compatibility.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from maya.agents.health_rag.agent import ClarificationAssessmentError, DisambiguationDecision, HealthRagHandle
from maya.agents.phyxio_exercise_agent import agent as exercise
from maya.agents.phyxio_exercise_agent import phyxio_sessions
from maya.framework.conversation.personalization import Preferences, SessionConfig


def decision(action="show_exercises", **kwargs):
    return exercise.ExerciseActionDecision(
        action=action,
        session_subject=kwargs.get("session_subject", "current_user"),
        safety_hold=kwargs.get("safety_hold", False),
        needs_clarification=kwargs.get("needs_clarification", False),
        reason="Offline fixture decision.",
        user_response=kwargs.get("user_response", ""),
    )


def make_node(selected):
    decision_model = Mock()
    decision_model.invoke.return_value = selected
    response_model = Mock()
    # Echo the actual service result: no invented model success in these tests.
    response_model.invoke.side_effect = lambda messages, **kwargs: exercise.ExerciseAgentResponse(
        response=exercise._to_text(json.loads(messages[-1].content)["tool_result"])
    )
    node = exercise._PhyxioExerciseNode(
        decision_model=decision_model,
        response_model=response_model,
        decision_prompt="Select one exercise operation.",
        response_prompt="Use the actual result.",
        tools=list(exercise._TOOL_REGISTRY.values()),
    )
    return node, decision_model, response_model


def make_service():
    service = Mock(spec=["get_routine_list", "session_history_reader", "show_routine", "calibrate_routine", "run_routine"])
    service.get_routine_list.return_value = {
        "routines": {"42": {"name": "Assigned programme", "exercises": [{"id": "7", "iterations": 8}]}},
        "exercises": {"7": {"name": "Seated march", "description": "Lift each knee in turn."}},
    }
    reader = Mock()
    reader.getRoutineList.side_effect = lambda: service.get_routine_list()
    reader.getSessionList.return_value = {
        "page": 0,
        "sessions": [{"date": "2026-10-01", "routine_id": "42", "results": [
            {"exercise_id": "7", "status": "completed", "goal_iter": 8, "done_iter": 8}
        ]}],
    }
    service.session_history_reader.return_value = reader
    service.show_routine.return_value = "Showing routine"
    service.calibrate_routine.return_value = "Calibration requested"
    service.run_routine.return_value = "Started routine"
    return service


class ExerciseServiceContractTests(unittest.TestCase):
    def setUp(self):
        self.service = make_service()
        self.enterContext(patch.object(exercise, "get_phyxio_service", return_value=self.service))
        self.enterContext(patch.object(phyxio_sessions, "get_phyxio_service", return_value=self.service))
        from maya.tests.test_phyxio_sessions import configured_service
        folder = self.enterContext(TemporaryDirectory())
        configured_service(folder)
        self.enterContext(patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(Path(folder) / "exercise.json")}))
        self.enterContext(patch.object(exercise, "ROUTINE_ID", ""))
        self.enterContext(patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}))

    def run_node(self, selected):
        node, _, _ = make_node(selected)
        return node({"messages": [HumanMessage(content="Exercise request.")]},
                    config={"configurable": {"user_id": "offline-user"}})["messages"][0]

    def test_all_five_actions_reach_the_real_tool_wrappers(self):
        cases = [
            ("show_exercises", [call.get_routine_list(), call.show_routine("42")], "Showing routine"),
            ("calibrate_exercises", [call.get_routine_list(), call.calibrate_routine("42")], "Calibration requested"),
            ("start_exercises", [call.get_routine_list(), call.run_routine("42")], "Started routine"),
            ("list_exercises", [call.get_routine_list()], "Seated march"),
            ("get_exercise_history", [call.session_history_reader("offline-user", 5000), call.get_routine_list()], "8/8 reps"),
        ]
        for action, expected_calls, expected_text in cases:
            with self.subTest(action=action):
                self.service.reset_mock()
                message = self.run_node(decision(action))
                self.assertIsInstance(message, AIMessage)
                self.assertIn(expected_text, message.content)
                self.assertEqual(self.service.method_calls, expected_calls)

    def test_safety_and_clarification_holds_never_contact_service(self):
        for hold in ("safety_hold", "needs_clarification"):
            with self.subTest(hold=hold):
                message = self.run_node(decision("start_exercises", **{hold: True}, user_response="Please wait."))
                self.assertEqual(message.content, "Please wait.")
                self.assertEqual(self.service.mock_calls, [])

    def test_missing_ice_is_reported_for_every_tool(self):
        with patch.object(exercise, "get_phyxio_service", side_effect=ModuleNotFoundError("No module named 'Ice'")):
            for action in exercise.DEFAULT_TOOLS:
                if action in {"get_latest_exercise_session", "get_exercise_history"}:
                    continue  # Native history failures are covered by provider/connection tests.
                with self.subTest(action=action):
                    self.assertIn("Ice", self.run_node(decision(action)).content)
        self.assertEqual(self.service.mock_calls, [])

    def test_empty_assignment_prevents_mutating_actions(self):
        self.service.get_routine_list.return_value = {"routines": {}}
        for action in ("show_exercises", "calibrate_exercises", "start_exercises"):
            with self.subTest(action=action):
                self.assertIn("No assigned Phyxio routines", self.run_node(decision(action)).content)
        self.service.show_routine.assert_not_called()
        self.service.calibrate_routine.assert_not_called()
        self.service.run_routine.assert_not_called()

    def test_transport_failure_reaches_response_stage_without_retrying_action(self):
        self.service.run_routine.side_effect = TimeoutError("mirror timeout")
        node, _, response = make_node(decision("start_exercises"))
        result = node({"messages": [HumanMessage(content="Start my routine.")]})
        self.service.run_routine.assert_called_once_with("42")
        self.assertIn("mirror timeout", result["messages"][0].content)
        self.assertIn("failed", json.loads(response.invoke.call_args.args[0][-1].content)["tool_result"])

    def test_history_survives_unavailable_routine_names(self):
        self.service.get_routine_list.side_effect = RuntimeError("catalog unavailable")
        message = self.run_node(decision("get_exercise_history"))
        self.assertIn("Exercise 7", message.content)
        self.assertIn("8/8 reps", message.content)

    def test_configured_routine_is_selected_from_multiple_assignments(self):
        data = self.service.get_routine_list.return_value
        data["routines"]["99"] = {"name": "Second programme", "exercises": []}
        with patch.object(exercise, "ROUTINE_ID", "99"):
            self.run_node(decision("start_exercises"))
        self.service.run_routine.assert_called_once_with("99")

    def test_completed_legacy_tool_call_cannot_capture_a_new_reply(self):
        node, _, _ = make_node(decision())
        messages = [
            HumanMessage(content="Previous request."),
            AIMessage(content="", tool_calls=[{"name": "RouteState", "args": {}, "id": "old-call"}]),
            ToolMessage(content="Previous result.", tool_call_id="old-call"),
            AIMessage(content="Previous reply."),
            HumanMessage(content="Show my exercises again."),
        ]
        result = node({"messages": messages})
        self.assertIsInstance(result["messages"][0], AIMessage)

    def test_current_pending_tool_call_still_receives_its_result(self):
        node, _, _ = make_node(decision())
        result = node({"messages": [
            HumanMessage(content="Show my exercises."),
            AIMessage(content="", tool_calls=[{"name": "RouteState", "args": {}, "id": "current-call"}]),
        ]})
        self.assertIsInstance(result["messages"][0], ToolMessage)
        self.assertEqual(result["messages"][0].tool_call_id, "current-call")

    def test_answered_call_in_current_turn_is_not_reused(self):
        messages = [
            HumanMessage(content="Show my exercises."),
            AIMessage(content="", tool_calls=[{"name": "RouteState", "args": {}, "id": "done"}]),
            ToolMessage(content="Done.", tool_call_id="done"),
        ]
        self.assertIsNone(exercise._PhyxioExerciseNode._extract_tool_call_id(messages))

    def test_dictionary_history_respects_the_new_user_turn(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "old", "name": "RouteState"}]},
            {"role": "user", "content": "Show my exercises."},
        ]
        self.assertIsNone(exercise._PhyxioExerciseNode._extract_tool_call_id(messages))

    def test_personality_reaches_response_and_hold_wording_without_changing_tools(self):
        preferences = Preferences(personality_id="supportive")
        config = SessionConfig(user_id="offline-user", preferences=preferences).graph_config()
        node, classifier, responder = make_node(decision())
        node({"messages": [HumanMessage(content="Show my exercises.")]}, config)
        for model in (classifier, responder):
            self.assertIn(preferences.response_style(), model.invoke.call_args.args[0][0].content)
            self.assertEqual(model.invoke.call_args.kwargs["config"]["configurable"]["preferences"], preferences.model_dump())
        self.service.show_routine.assert_called_once_with("42")
        self.service.reset_mock()
        classifier.invoke.return_value = decision("start_exercises", safety_hold=True, user_response="Please wait.")
        result = node({"messages": [HumanMessage(content="Start my routine.")]}, config)
        self.assertEqual(result["messages"][0].content, "Please wait.")
        self.assertEqual(self.service.mock_calls, [])


class SupervisorExerciseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}))
        # Load an isolated copy: factory constructs model/extractor globals at import.
        # Patching those constructors keeps this suite credential-free and leaves the
        # normal application's module cache untouched.
        name = "maya.framework.supervisor._offline_factory"
        path = Path(__file__).resolve().parents[1] / "framework" / "supervisor" / "factory.py"
        spec = importlib.util.spec_from_file_location(name, path)
        self.factory = importlib.util.module_from_spec(spec)
        self.enterContext(patch.dict(sys.modules, {name: self.factory}))
        with patch("maya.framework.integrations.models.get_chat_model", return_value=Mock()), patch("trustcall.create_extractor"):
            spec.loader.exec_module(self.factory)

        self.service = make_service()
        self.enterContext(patch.object(exercise, "get_phyxio_service", return_value=self.service))
        self.enterContext(patch.object(exercise, "ROUTINE_ID", ""))
        self.node, self.decision_model, _ = make_node(decision())
        self.router = Mock()
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="phyxio_exercise_agent", selected_ontology_models=["exercise"]
        )
        self.factory.model.with_structured_output.return_value = self.router
        memory = Mock()
        memory.load_context.return_value = ("Recorded clinical context.", "")
        memory.schedule.return_value = False
        self.memory = memory
        self.enterContext(patch.object(self.factory, "OntologyMemoryService", return_value=memory))
        self.store = InMemoryStore(index={"embed": lambda texts: [[1.0, 0.0] for _ in texts], "dims": 2})
        self.health = SimpleNamespace(
            name="health_rag", agent=Mock(),
            assess_clarification=Mock(return_value=DisambiguationDecision(needs_clarification=False)),
        )
        handle = HealthRagHandle(name="health_rag", agent=None, ontology_block="Offline ontology")
        self.health.config = handle.config
        self.health.cache_scope = lambda style, context, memory_context="": HealthRagHandle(
            name="health_rag", agent=None, ontology_block=handle.ontology_block,
            config=self.health.config).cache_scope(style, context, memory_context)
        self.health.agent.invoke.return_value = {
            "messages": [AIMessage(content="Evidence-based fixture answer.")],
            "citations": [
                {"title": f"Offline source {i}", "source_path": f"/private/corpus/source-{i}.pdf", "page_number": i}
                for i in range(3)
            ],
        }
        self.enterContext(patch.object(self.factory, "get_postgres_memory", return_value=(self.store, InMemorySaver())))
        self.enterContext(patch.object(self.factory, "_load_agents", return_value=(
            [self.node, self.health], {"phyxio_exercise_agent": "Exercise control"}
        )))
        self.graph = self.factory.build_supervisor()
        self.config = {"configurable": {"thread_id": "offline-thread", "user_id": "offline-user"}}

    def run_turn(self, text="Show my exercises."):
        return self.graph.invoke({"messages": [HumanMessage(content=text)]}, self.config)

    def test_session_feedback_uses_history_and_identity_through_supervisor(self):
        from maya.tests.test_exercise_sessions import sample_service

        selected = decision("get_latest_exercise_session").model_copy(update={"session_subject": "current_user"})
        sessions = sample_service()
        # Preserve a single participant throughout the actual exercise interaction.
        self.config = SessionConfig(user_id="participant-001").graph_config()
        self.decision_model.invoke.return_value = decision("start_exercises")
        self.run_turn("Start my routine.")
        self.decision_model.invoke.return_value = selected
        from maya.agents.phyxio_exercise_agent import sessions as session_module
        with patch.object(session_module, "load_session_service", return_value=sessions), \
                patch.object(sessions.provider, "fetch_sessions", wraps=sessions.provider.fetch_sessions) as fetch:
            result = self.run_turn("I finished. How did I perform?")
        fetch.assert_called_once_with("participant-001")
        extra = result["messages"][-1].additional_kwargs
        self.assertEqual(extra["executed_tool"], "get_latest_exercise_session")
        self.assertEqual(len(extra["exercise_session"]["previous"]), 5)
        for messages in (self.router.invoke.call_args.args[0], self.decision_model.invoke.call_args.args[0]):
            self.assertTrue(any(m.content == "Start my routine." for m in messages))
            self.assertEqual(messages[-1].content, "I finished. How did I perform?")
        self.service.run_routine.assert_called_once()

    def test_session_subject_guard_prevents_cross_person_lookup(self):
        self.decision_model.invoke.return_value = decision("get_latest_exercise_session").model_copy(
            update={"session_subject": "other"})
        with patch.object(exercise, "_invoke_registered_tool") as fetch:
            self.run_turn("How did my friend perform?")
        fetch.assert_not_called()

    def test_session_lookup_for_different_participant_has_no_fallback_records(self):
        from maya.tests.test_exercise_sessions import sample_service
        from maya.agents.phyxio_exercise_agent import sessions as session_module

        self.config = SessionConfig(user_id="participant-without-records").graph_config()
        self.decision_model.invoke.return_value = decision("get_latest_exercise_session").model_copy(
            update={"session_subject": "current_user"})
        with patch.object(session_module, "load_session_service", return_value=sample_service()):
            result = self.run_turn("Review my latest exercise session.")
        data = result["messages"][-1].additional_kwargs["exercise_session"]
        self.assertEqual(data["status"], "no_data")
        self.assertIsNone(data["latest"])

    def test_session_response_failure_does_not_speak_raw_records(self):
        from maya.tests.test_exercise_sessions import sample_service
        from maya.agents.phyxio_exercise_agent import sessions as session_module

        self.config = SessionConfig(user_id="participant-001").graph_config()
        self.decision_model.invoke.return_value = decision("get_latest_exercise_session").model_copy(
            update={"session_subject": "current_user"})
        service = sample_service()
        self.node._response_model.invoke.side_effect = TimeoutError
        with patch.object(session_module, "load_session_service", return_value=service):
            result = self.run_turn("Review my latest exercise session.")
        self.assertEqual(result["messages"][-1].content, service.config.fallback_text)
        self.assertEqual(result["messages"][-1].additional_kwargs["exercise_session"]["status"], "ok")

    def test_guard_and_direct_answer_receive_dialogue_without_replaying_actions(self):
        self.run_turn("Show my routine.")
        self.router.invoke.side_effect = [
            self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="direct_answer", selected_ontology_models=[]),
            self.factory.HealthRouteGuardDecision(route_type="direct_answer"),
        ]
        self.factory.model.invoke.return_value = AIMessage(content="Which activity do you mean?")
        self.run_turn("I meant my presentation. How was it?")
        for messages in (self.router.invoke.call_args.args[0], self.factory.model.invoke.call_args.args[0]):
            self.assertTrue(any(m.content == "Show my routine." for m in messages))
        self.service.show_routine.assert_called_once()

    def test_supervisor_routes_to_real_exercise_node_and_finalizes_memory(self):
        result = self.run_turn()
        self.service.show_routine.assert_called_once_with("42")
        self.assertEqual(result["route_type"], "phyxio_exercise_agent")
        self.assertEqual(result["messages"][-1].content, "Showing routine")
        self.assertIn("clinical_profile", result["selected_ontology_models"])
        self.memory.schedule.assert_called_once()
        self.assertEqual(self.memory.schedule.call_args.kwargs["user_id"], "offline-user")
        self.assertIn("Recorded clinical context.", self.decision_model.invoke.call_args.args[0][0].content)

    def test_safety_hold_returns_through_graph_without_mirror_action(self):
        self.decision_model.invoke.return_value = decision("start_exercises", safety_hold=True, user_response="Please do not start.")
        result = self.run_turn("Start my workout; I feel unwell.")
        self.assertEqual(result["messages"][-1].content, "Please do not start.")
        self.assertEqual(self.service.mock_calls, [])

    def test_partner_ontology_reaches_exercise_decision_through_supervisor(self):
        registry = self.factory.get_ontology_registry()
        self.memory.load_context.side_effect = lambda **kwargs: (registry.render_context(
            kwargs["agent_id"], kwargs["model_ids"]), "")
        result = self.run_turn()
        context = registry.render_context("phyxio_exercise_agent", result["selected_ontology_models"])
        self.assertIn("[exercise]", context)
        self.assertIn("[clinical_profile]", context)
        self.assertIn("[cancer_history]", context)
        self.assertIn(context, self.decision_model.invoke.call_args.args[0][0].content)

    def test_partner_ontology_reaches_health_assessment_and_answer_through_supervisor(self):
        registry = self.factory.get_ontology_registry()
        self.memory.load_context.side_effect = lambda **kwargs: (registry.render_context(
            kwargs["agent_id"], kwargs["model_ids"]), "")
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
            question_subject="current_user", route_type="health_rag", selected_ontology_models=["cancer_history"])
        result = self.run_turn("What should I know about my cancer treatment?")
        context = registry.render_context("health_rag", result["selected_ontology_models"])
        self.assertIn("[knowledge_resource]", context)
        self.assertIn("[cancer_history]", context)
        self.assertEqual(context, self.health.assess_clarification.call_args.kwargs["ontology_context"])
        self.assertEqual(context, self.health.agent.invoke.call_args.args[0]["ontology_context"])

    def test_repeated_turns_execute_once_each_and_keep_conversation(self):
        self.run_turn()
        result = self.run_turn("Show my exercises again.")
        self.assertEqual(self.service.show_routine.call_count, 2)
        self.assertEqual(len([m for m in result["messages"] if isinstance(m, HumanMessage)]), 2)
        self.assertEqual(len([m for m in result["messages"] if isinstance(m, AIMessage)]), 2)

    def test_guard_can_recover_an_exercise_request_from_direct_answer(self):
        self.router.invoke.side_effect = [
            self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="direct_answer", selected_ontology_models=[]),
            self.factory.HealthRouteGuardDecision(route_type="phyxio_exercise_agent", exercise_reference="Show my exercises."),
        ]
        result = self.run_turn()
        self.assertEqual(result["route_type"], "phyxio_exercise_agent")
        self.service.show_routine.assert_called_once_with("42")

    def test_guard_cannot_invent_an_exercise_reference(self):
        for reference in ("", "Start my workout"):
            with self.subTest(reference=reference):
                self.config = SessionConfig(user_id="offline-user").graph_config()
                self.router.invoke.side_effect = [
                    self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="direct_answer", selected_ontology_models=[]),
                    self.factory.HealthRouteGuardDecision(route_type="phyxio_exercise_agent", exercise_reference=reference),
                ]
                self.factory.model.invoke.return_value = AIMessage(content="Which activity do you mean?")
                result = self.run_turn("How did I perform?")
                self.assertEqual(result["route_type"], "direct_answer")
        self.assertEqual(self.service.mock_calls, [])

    def test_direct_response_receives_style_without_adding_it_to_routing_prompts(self):
        preferences = Preferences(personality_id="concise")
        self.config = SessionConfig(user_id="offline-user", preferences=preferences).graph_config()
        self.router.invoke.side_effect = [
            self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="direct_answer", selected_ontology_models=[]),
            self.factory.HealthRouteGuardDecision(route_type="direct_answer"),
        ]
        self.factory.model.invoke.return_value = AIMessage(content="A short answer.")
        result = self.run_turn("Tell me about the moon.")
        self.assertEqual(result["messages"][-1].content, "A short answer.")
        self.assertIn(preferences.response_style(), self.factory.model.invoke.call_args.args[0][0].content)
        for invocation in self.router.invoke.call_args_list:
            self.assertNotIn("RESPONSE STYLE", invocation.args[0][0].content)

    def test_health_cache_and_wording_are_scoped_to_personality_and_user_not_voice(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        for personality, voice, user_id, expected_calls in [
            ("supportive", "en-GB-RyanNeural", "a", 1),
            ("supportive", "en-GB-LibbyNeural", "a", 1),
            ("concise", "en-GB-RyanNeural", "a", 2),
            ("supportive", "en-GB-RyanNeural", "a", 2),
            ("supportive", "en-GB-RyanNeural", "b", 3),
        ]:
            with self.subTest(personality=personality, voice=voice, user_id=user_id):
                preferences = Preferences(personality_id=personality, tts_voice=voice)
                self.config = SessionConfig(user_id=user_id, preferences=preferences).graph_config()
                self.run_turn("What is cardiovascular health?")
                self.assertEqual(self.health.agent.invoke.call_count, expected_calls)
                self.assertEqual(self.health.assess_clarification.call_args.kwargs["response_style"], preferences.response_style())
        self.assertEqual(self.health.agent.invoke.call_args.args[0]["response_style"], Preferences(personality_id="supportive").response_style())

    def test_semantic_cache_excludes_other_personalities_and_legacy_answers(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        self.config = SessionConfig(user_id="offline-user", preferences=Preferences(personality_id="concise")).graph_config()
        namespace = ("health_rag", "offline-user", "qa_history")
        value = {
            "type": "health_rag_summary", "user_question": "What is cardiovascular health?",
            "summary": "Other style answer", "citations": [],
        }
        self.store.put(namespace, "legacy", value)
        self.store.put(namespace, "other-style", {**value, "response_style": "v1:supportive"})
        self.store.put(namespace, "older-style-version", {**value, "response_style": "v3:concise"})
        self.run_turn("What is cardiovascular health?")
        self.health.agent.invoke.assert_called_once()
        # Remove the new exact entry; a matching style may still use semantic reuse.
        for item in self.store.search(namespace):
            if item.key.startswith("summary:"):
                self.store.delete(namespace, item.key)
        self.store.put(namespace, "matching-style", {**value, "response_style": self.health.cache_scope(Preferences(personality_id="concise").response_style(), *self.memory.load_context.return_value), "summary": "Matching answer"})
        self.config = SessionConfig(user_id="offline-user", preferences=Preferences(personality_id="concise")).graph_config()
        result = self.run_turn("What is cardiovascular health?")
        self.health.agent.invoke.assert_called_once()
        self.assertIn("Matching answer", result["messages"][-1].content)

    def test_supervisor_cache_excludes_answers_from_changed_prompt_or_profile(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        def ask():
            self.config = SessionConfig(user_id="config-cache-check").graph_config()
            return self.run_turn("What is cardiovascular health?")
        ask()
        ask()
        self.assertEqual(self.health.agent.invoke.call_count, 1)
        self.health.config = self.health.config.model_copy(update={"prompts":
            self.health.config.prompts.model_copy(update={"answer": "Updated partner wording."})})
        changed = ask()
        self.assertEqual(self.health.agent.invoke.call_count, 2)
        self.assertTrue(changed["messages"][-1].additional_kwargs["workflow_trace"][0]["retrieval_used"])
        self.memory.load_context.return_value = ("Updated clinical record.", "")
        ask()
        self.assertEqual(self.health.agent.invoke.call_count, 3)

    def test_saved_memory_reaches_retrieval_without_model_source_labels_and_skips_old_cache(self):
        from maya.agents.health_rag.agent import RetrieveLocalLiterature
        from maya.tests.test_health_rag_retrieval import StubRetriever, StubRewriteModel

        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
            question_subject='current_user', route_type='health_rag', selected_ontology_models=['cancer_history'])
        memory = 'User-reported treatment: doxorubicin.'
        context = 'Applicable ontology context:\n' + memory
        self.memory.load_context.return_value = (context, memory)
        # The assessor deliberately supplies no ontology_profile slot labels.
        self.assertTrue(all(slot.source == 'none' for _, slot in self.health.assess_clarification.return_value.picot))
        self.store.put(('health_rag', 'offline-user', 'qa_history'), 'old-summary', {
            'type': 'health_rag_summary', 'summary': 'Old answer that missed treatment in retrieval',
            'user_question': 'Is my medication okay with my treatments?', 'citations': [],
            'response_style': self.health.cache_scope(Preferences().response_style(), context),
        })
        retriever, rewrite = StubRetriever(), StubRewriteModel('Bisoprolol suitability after doxorubicin?')
        middleware = RetrieveLocalLiterature(retriever, rewrite)
        def answer(state, **kwargs):
            return {**middleware.before_model(state, None), 'messages': [AIMessage(content='Fixture answer.')]}
        self.health.agent.invoke.side_effect = answer
        result = self.run_turn('Is my medication okay with my treatments?')
        self.assertEqual(rewrite.calls, 1)
        self.assertEqual(retriever.queries, ['Bisoprolol suitability after doxorubicin?'])
        self.assertIn(memory, rewrite.prompts[0])
        trace = result['messages'][-1].additional_kwargs['workflow_trace'][0]
        self.assertEqual(trace['memory_lookup'], 'miss')
        self.assertEqual(trace['retrieval_query'], retriever.queries[0])
        self.config = SessionConfig(user_id='offline-user').graph_config()
        cached = self.run_turn('Is my medication okay with my treatments?')
        self.assertEqual(cached['messages'][-1].additional_kwargs['workflow_trace'][0]['memory_lookup'], 'exact')
        self.assertEqual(rewrite.calls, 1)

    def test_supervisor_passes_history_separately_from_retrieval_question(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        self.run_turn("What is cardiovascular health?")
        self.run_turn("How is that monitored?")
        payload = self.health.agent.invoke.call_args.args[0]
        self.assertEqual(payload["standalone_question"], "How is that monitored?")
        self.assertEqual(payload["conversation_context"].count("What is cardiovascular health?"), 1)

    def test_nonself_health_requests_never_receive_user_records_or_observation_access(self):
        from maya.framework.integrations.biomarkers import BiomarkerRequest
        for subject in ("other", "general", "unknown"):
            with self.subTest(subject=subject):
                self.config = SessionConfig(user_id="subject-check").graph_config()
                self.router.invoke.return_value = self.factory.SupervisorRouteDecision(
                    question_subject=subject, route_type="health_rag", selected_ontology_models=["cancer_history"],
                    biomarker_request=BiomarkerRequest(subject="current_user", measurement_ids=["blood_pressure"]))
                self.memory.load_context.reset_mock()
                self.run_turn("A health question about someone else.")
                self.memory.load_context.assert_not_called()
                kwargs = self.health.assess_clarification.call_args.kwargs
                self.assertFalse(kwargs["allow_biomarker_lookup"])
                self.assertEqual(kwargs["biomarker_catalog"], "")
                self.assertEqual(kwargs["biomarker_context"], "")
                self.assertNotIn("Recorded clinical context", kwargs["ontology_context"])
                self.assertIn("Relevant user ontology records: none found", kwargs["ontology_context"])
                self.assertEqual(self.health.agent.invoke.call_args.args[0]['ontology_memory_context'], '')

    def test_health_references_survive_fresh_exact_and_semantic_cache_delivery(self):
        from maya.framework.conversation.responses import load_response_policy

        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        policy = load_response_policy()
        for lookup, question in [
            ("miss", "What is cardiovascular health?"),
            ("exact", "What is cardiovascular health?"),
            ("semantic", "Explain what is cardiovascular health?"),
        ]:
            with self.subTest(lookup=lookup):
                self.config = SessionConfig(user_id="offline-references").graph_config()
                result = self.run_turn(question)
                message = result["messages"][-1]
                presentation = message.additional_kwargs["presentation"]
                trace = message.additional_kwargs["workflow_trace"][0]
                self.assertEqual(trace["memory_lookup"], lookup)
                self.assertEqual(len(trace["citations"]), 3)
                self.assertEqual(len(presentation["references"]), 3)
                self.assertIn("Offline source 2", presentation["display_text"])
                self.assertNotIn("Offline source", presentation["speech_text"])
                self.assertTrue(presentation["display_text"].endswith(policy.health_disclaimer))
                self.assertNotIn(policy.health_disclaimer, message.content)
        self.health.agent.invoke.assert_called_once()

    def test_cached_health_reply_uses_current_disclaimer(self):
        from maya.framework.conversation.responses import load_response_policy

        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        self.config = SessionConfig(user_id="offline-policy").graph_config()
        self.run_turn("What is cardiovascular health?")
        policy = load_response_policy().model_copy(update={"health_disclaimer": "Updated partner notice."})
        self.config = SessionConfig(user_id="offline-policy").graph_config()
        with patch("maya.framework.conversation.responses.load_response_policy", return_value=policy):
            result = self.run_turn("What is cardiovascular health?")
        self.health.agent.invoke.assert_called_once()
        presentation = result["messages"][-1].additional_kwargs["presentation"]
        self.assertTrue(presentation["display_text"].endswith(policy.health_disclaimer))
        self.assertTrue(presentation["speech_text"].endswith(policy.health_disclaimer))

    def test_health_progress_for_fresh_and_cached_answers_never_enters_memory(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=[])
        for lookup, question in [("miss", "What is cardiovascular health?"),
                                 ("exact", "What is cardiovascular health?"),
                                 ("semantic", "Explain what is cardiovascular health?")]:
            with self.subTest(lookup=lookup):
                self.config = SessionConfig(user_id="progress-cache").graph_config()
                chunks = list(self.graph.stream(
                    {"messages": [HumanMessage(content=question)]}, self.config,
                    stream_mode=["updates", "custom"], subgraphs=True))
                events = [data for _, mode, data in chunks if mode == "custom"]
                self.assertEqual(events, [{"event": "health_processing_started"}])
                state = self.graph.get_state(self.config).values
                self.assertEqual(state["messages"][-1].additional_kwargs["workflow_trace"][0]["memory_lookup"], lookup)
                self.assertNotIn("health_processing_started", str(state))
                self.assertNotIn("health_processing_started", str(self.store.search(("health_rag", "progress-cache", "qa_history"))))

    def test_health_progress_precedes_clarification_and_reappears_on_node_resume(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=[])
        self.health.assess_clarification.return_value = DisambiguationDecision(
            needs_clarification=True, clarification_question="Which treatment?")
        initial = list(self.graph.stream({"messages": [HumanMessage(content="Is this suitable?")]},
                                        self.config, stream_mode=["updates", "custom"], subgraphs=True))
        self.assertEqual([data for _, mode, data in initial if mode == "custom"],
                         [{"event": "health_processing_started"}])
        self.health.agent.invoke.assert_not_called()
        resumed = list(self.graph.stream(Command(resume="Doxorubicin"), self.config,
                                        stream_mode=["updates", "custom"], subgraphs=True))
        self.assertEqual([data for _, mode, data in resumed if mode == "custom"],
                         [{"event": "health_processing_started"}])

    def test_health_progress_for_failed_assessment_but_not_direct_answer(self):
        for route in ("health_rag", "direct_answer"):
            with self.subTest(route=route):
                self.config = SessionConfig(user_id="progress-errors").graph_config()
                self.router.invoke.side_effect = [
                    self.factory.SupervisorRouteDecision(question_subject="current_user", route_type=route, selected_ontology_models=[]),
                    self.factory.HealthRouteGuardDecision(route_type="direct_answer"),
                ]
                self.health.assess_clarification.side_effect = ClarificationAssessmentError("offline")
                self.factory.model.invoke.return_value = AIMessage(content="Hello.")
                chunks = list(self.graph.stream({"messages": [HumanMessage(content="Hello")]}, self.config,
                                               stream_mode=["updates", "custom"], subgraphs=True))
                self.assertEqual([data for _, mode, data in chunks if mode == "custom"],
                                 [{"event": "health_processing_started"}] if route == "health_rag" else [])
                self.health.agent.invoke.assert_not_called()

    def test_health_progress_delivered_while_clarification_assessment_is_still_running(self):
        from threading import Event
        from maya.framework.supervisor.run_with_vad import stream_updates

        progress_seen = Event()
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=[])

        def assess(*args, **kwargs):
            self.assertTrue(progress_seen.wait(5), "Progress must not wait for assessment")
            return DisambiguationDecision(needs_clarification=False)

        self.health.assess_clarification.side_effect = assess
        reply, interrupts = stream_updates(self.graph, {"messages": [HumanMessage(content="A health question")]},
                                           self.config, on_health_processing=progress_seen.set)
        self.assertFalse(interrupts)
        self.assertIsNotNone(reply)

    def test_required_treatment_pauses_before_retrieval_and_reaches_answer_on_resume(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        self.health.assess_clarification.return_value = DisambiguationDecision(
            question_type="personalized_guidance", needs_clarification=True,
            missing_ontology_fields=["treatment"], material_missing_fields=["treatment"],
            clarification_field="treatment", clarification_question="Which treatment do you mean?",
        )
        state = self.run_turn("Is my medicine okay with my treatments?")
        self.assertEqual(state["__interrupt__"][0].value, "Which treatment do you mean?")
        self.health.agent.invoke.assert_not_called()
        state = self.graph.invoke(Command(resume="I mean my previous doxorubicin treatment."), self.config)
        self.health.agent.invoke.assert_called_once()
        self.assertIn("previous doxorubicin", self.health.agent.invoke.call_args.args[0]["messages"][0].content)
        audit = state["messages"][-1].additional_kwargs["clarification"]
        self.assertEqual(audit["status"], "clarified")
        self.assertEqual(audit["material_missing_fields"], ["treatment"])
        self.health.assess_clarification.assert_called_once()

    def test_saved_assessment_survives_graph_rebuild_and_keeps_unknown_reply(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=[])
        self.health.assess_clarification.return_value = DisambiguationDecision(
            needs_clarification=True, clarification_question="Which test and value?")
        self.run_turn("My cardiac blood test was high. What does that mean?")
        self.health.assess_clarification.side_effect = AssertionError("Assessment must not rerun on resume")
        self.graph = self.factory.build_supervisor()
        result = self.graph.invoke(Command(resume="I don't remember, I don't know."), self.config)
        self.health.assess_clarification.assert_called_once()
        supplied_question = self.health.agent.invoke.call_args.args[0]["messages"][0].content
        self.assertIn("Which test and value?", supplied_question)
        self.assertIn("I don't remember, I don't know.", supplied_question)
        self.assertTrue(result["messages"][-1].content)

        self.health.assess_clarification.side_effect = None
        self.health.assess_clarification.return_value = DisambiguationDecision(needs_clarification=False)
        self.run_turn("What is cancer-related fatigue?")
        self.assertEqual(self.health.assess_clarification.call_count, 2)

    def test_failed_clarification_check_stops_before_retrieval_or_cache_write(self):
        from maya.framework.conversation.responses import load_response_policy

        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user", route_type="health_rag", selected_ontology_models=[])
        self.health.assess_clarification.side_effect = ClarificationAssessmentError("fixture failure")
        state = self.run_turn("Is my medicine okay with my treatments?")
        self.health.agent.invoke.assert_not_called()
        self.assertEqual(self.store.search(("health_rag", "offline-user", "qa_history")), [])
        message = state["messages"][-1]
        self.assertEqual(message.content, load_response_policy().clarification_unavailable_text)
        self.assertEqual(message.additional_kwargs["clarification"]["status"], "error")
        self.assertNotIn("presentation", message.additional_kwargs)

    def test_pending_memory_stops_before_clarification_or_retrieval(self):
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=["cancer_history"])
        for failure_in_health in (False, True):
            with self.subTest(failure_in_health=failure_in_health):
                self.config["configurable"]["thread_id"] = f"pending-{failure_in_health}"
                failure = self.factory.OntologyMemoryNotReady("pending")
                self.memory.load_context.side_effect = [("Recorded context", ""), failure] if failure_in_health else failure
                result = self.run_turn("Is my measurement safe given my treatment?")
                message = result["messages"][-1]
                self.assertEqual(message.additional_kwargs["source"], "ontology_memory")
                self.assertEqual(message.additional_kwargs["memory_status"], "pending")
                self.assertFalse(self.graph.get_state(self.config).interrupts)
                self.health.assess_clarification.assert_not_called()
                self.health.agent.invoke.assert_not_called()
                self.assertIn("cancer_history", self.memory.schedule.call_args.kwargs["model_ids"])
                self.assertEqual(self.memory.schedule.call_args.kwargs["source_text"],
                                 "Is my measurement safe given my treatment?")

    def setup_biomarkers(self, route="health_rag", supervisor_fetch=False):
        from maya.tests.test_biomarkers import demo_service, request
        self.biomarkers = demo_service()
        self.lookup = request("blood_pressure")
        self.config["configurable"]["user_id"] = "participant-001"
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type=route, selected_ontology_models=["measurement_biomarker"],
            biomarker_request=self.lookup if supervisor_fetch else None)
        self.graph = self.factory.build_supervisor(biomarker_service=self.biomarkers)

    def test_supervisor_lookup_reaches_health_assessment_and_retrieval_without_refetch(self):
        self.setup_biomarkers(supervisor_fetch=True)
        with patch.object(self.biomarkers.provider, "fetch_latest", wraps=self.biomarkers.provider.fetch_latest) as fetch:
            result = self.run_turn("What does my latest blood pressure mean?")
            fetch.assert_called_once()
        assessment_context = self.health.assess_clarification.call_args.kwargs["biomarker_context"]
        retrieval_context = self.health.agent.invoke.call_args.args[0]["biomarker_context"]
        self.assertIn("134", assessment_context)
        self.assertEqual(assessment_context, retrieval_context)
        final = result["messages"][-1]
        self.assertTrue(final.additional_kwargs["biomarker_results"])
        self.assertTrue(final.additional_kwargs["presentation"]["speech_text"].startswith("Evidence-based fixture answer."))
        self.assertEqual(self.store.search(("health_rag", "participant-001", "qa_history")), [])
        # Same question on a new turn must fetch and retrieve again, never reuse a summary.
        self.run_turn("What does my latest blood pressure mean?")
        self.assertEqual(self.health.agent.invoke.call_count, 2)

    def test_health_initiated_lookup_reassesses_then_survives_clarification_resume(self):
        self.setup_biomarkers()
        self.health.assess_clarification.side_effect = [
            DisambiguationDecision(needs_clarification=True, biomarker_request=self.lookup),
            DisambiguationDecision(needs_clarification=True, clarification_question="When was your treatment?"),
        ]
        with patch.object(self.biomarkers.provider, "fetch_latest", wraps=self.biomarkers.provider.fetch_latest) as fetch:
            self.run_turn("How should I understand my measurements after treatment?")
            self.assertTrue(self.graph.get_state(self.config).interrupts)
            self.health.agent.invoke.assert_not_called()
            result = self.graph.invoke(Command(resume="Last year."), self.config)
            fetch.assert_called_once()
        self.assertEqual(self.health.assess_clarification.call_count, 2)
        self.assertFalse(self.health.assess_clarification.call_args.kwargs["allow_biomarker_lookup"])
        self.assertIn("134", self.health.agent.invoke.call_args.args[0]["biomarker_context"])
        self.assertTrue(result["messages"][-1].content)

    def test_observation_lookup_bypasses_preexisting_cached_answer(self):
        self.setup_biomarkers()
        question = "Explain my blood pressure."
        self.run_turn(question)
        self.assertTrue(self.store.search(("health_rag", "participant-001", "qa_history")))
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=["measurement_biomarker"],
            biomarker_request=self.lookup)
        self.config["configurable"]["thread_id"] = "new-measurement-thread"
        result = self.run_turn(question)
        self.assertEqual(self.health.agent.invoke.call_count, 2)
        trace = result["messages"][-1].additional_kwargs["workflow_trace"][0]
        self.assertFalse(trace["memory_hit"])
        self.assertTrue(trace["retrieval_used"])

    def test_readout_returns_tool_data_without_health_retrieval(self):
        self.setup_biomarkers(route="biomarkers", supervisor_fetch=True)
        self.factory.model.invoke.return_value = AIMessage(content="Recorded blood pressure: 134/86 mmHg.")
        result = self.run_turn("Show my latest blood pressure.")
        self.health.assess_clarification.assert_not_called()
        self.health.agent.invoke.assert_not_called()
        self.assertIn("bp-latest", self.factory.model.invoke.call_args.args[0][-1].content)
        self.assertEqual(result["messages"][-1].additional_kwargs["source"], "biomarkers")

    def test_new_turn_clears_observations_and_urgent_assessment_does_not_fetch(self):
        self.setup_biomarkers(supervisor_fetch=True)
        self.run_turn("Explain my measurement.")
        self.router.invoke.return_value = self.factory.SupervisorRouteDecision(question_subject="current_user",
            route_type="health_rag", selected_ontology_models=[])
        self.health.assess_clarification.return_value = DisambiguationDecision(
            needs_clarification=False, urgent_safety_bypass=True, biomarker_request=self.lookup)
        with patch.object(self.biomarkers.provider, "fetch_latest") as fetch:
            result = self.run_turn("Urgent symptom fixture.")
            fetch.assert_not_called()
        self.assertEqual(result["biomarker_results"], [])
        self.assertEqual(self.health.agent.invoke.call_args.args[0]["biomarker_context"], "")


if __name__ == "__main__":
    unittest.main()
