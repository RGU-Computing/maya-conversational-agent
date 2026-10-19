"""Provider, comparison and dialogue boundaries; no model/network calls."""

import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from maya.agents.phyxio_exercise_agent.sessions import load_session_service, get_latest_exercise_session
from maya.framework.conversation.history import ConversationConfig, recent_dialogue


EXAMPLE = Path(__file__).resolve().parents[1] / "agents/phyxio_exercise_agent/session_config.example.json"


def sample_service():
    with patch.dict(os.environ, {"MAYA_EXERCISE_SESSION_CONFIG": str(EXAMPLE)}):
        return load_session_service()


class ExerciseSessionTests(unittest.TestCase):
    def setUp(self):
        self.service = sample_service()

    def latest(self):
        return self.service.latest("participant-001")

    def test_latest_and_five_comparable_sessions_with_exact_statistics(self):
        result = self.latest()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["latest"]["session_id"], "session-12")
        self.assertEqual(len(result["previous"]), 5)
        metric = result["comparison"]["seated_march_repetitions"]
        self.assertEqual(metric["previous_mean"], 15)
        self.assertEqual(metric["difference"], 5)
        self.assertNotIn("user_id", result["latest"])

    def test_limit_and_sort_are_deterministic(self):
        self.service.provider.sessions.reverse()
        self.service.config.comparison_limit = 2
        result = self.latest()
        self.assertEqual([s["session_id"] for s in result["previous"]], ["session-09", "session-07"])
        self.assertEqual(result["comparison"]["seated_march_repetitions"]["previous_mean"], 17)
        self.service.config.comparison_limit = 0
        self.assertEqual(self.latest()["comparison"], {})

    def test_missing_history_or_metrics_never_fabricates_a_baseline(self):
        self.service.provider.sessions = [self.service.provider.sessions[-1]]
        self.assertEqual(self.latest()["comparison"], {})
        self.assertEqual(self.latest()["previous"], [])
        self.service.provider.sessions[0].metrics = {}
        self.assertEqual(self.latest()["latest"]["metrics"], {})

    def test_other_routine_prescription_target_and_units_are_not_compared(self):
        records = self.service.provider.sessions
        records[0].routine_id = "other"
        records[1].comparison_key = "changed-prescription"
        records[2].metrics["seated_march_repetitions"].unit = "other-unit"
        records[3].metrics["seated_march_repetitions"].target = 30
        result = self.latest()
        metric = result["comparison"]["seated_march_repetitions"]
        self.assertEqual(metric["previous_count"], 1)
        self.assertEqual(metric["previous_mean"], 18)

    def test_interrupted_lower_and_zero_values_remain_honest(self):
        latest = self.service.provider.sessions[-1]
        latest.status = "interrupted"
        latest.metrics["seated_march_repetitions"].value = 0
        result = self.latest()
        self.assertEqual(result["latest"]["status"], "interrupted")
        self.assertEqual(result["comparison"]["seated_march_repetitions"]["difference"], -15)

    def test_in_progress_and_future_records_are_not_latest_performed(self):
        records = self.service.provider.sessions
        records[-1].status = "in_progress"
        records[-2].recorded_at = datetime.now(timezone.utc) + timedelta(days=1)
        self.assertEqual(self.latest()["latest"]["session_id"], "session-07")

    def test_empty_disabled_timeout_and_wrong_user_return_no_records(self):
        self.assertEqual(self.service.latest("unknown")["status"], "no_data")
        self.assertEqual(self.service.latest("")["status"], "unavailable")
        self.service.provider = Mock(synthetic=False)
        self.service.provider.fetch_sessions.side_effect = TimeoutError
        self.assertEqual(self.latest()["status"], "unavailable")
        self.service.provider = None
        self.assertEqual(self.latest()["status"], "unavailable")

    def test_provider_cannot_return_another_participants_sessions(self):
        records = self.service.provider.sessions
        records[0].user_id = "someone-else"
        self.service.provider = Mock(synthetic=False)
        self.service.provider.fetch_sessions.return_value = records
        result = self.latest()
        self.assertEqual(result["status"], "invalid_response")
        self.assertIsNone(result["latest"])

    def test_duplicate_ids_and_tied_latest_are_not_arbitrarily_selected(self):
        records = self.service.provider.sessions
        records[-2].session_id = records[-1].session_id
        self.assertEqual(self.latest()["status"], "invalid_response")
        records[-2].session_id = "distinct"
        records[-2].recorded_at = records[-1].recorded_at
        self.assertEqual(self.latest()["status"], "ambiguous_latest")

    def test_identity_is_not_in_model_visible_tool_arguments(self):
        self.assertEqual(get_latest_exercise_session.args, {})

    def test_dialogue_is_bounded_and_does_not_replay_tool_calls(self):
        messages = [HumanMessage(content="Old"), AIMessage(content="Old answer"),
                    HumanMessage(content="Start my routine"),
                    AIMessage(content="Started", tool_calls=[{"name": "start_exercises", "args": {}, "id": "x"}]),
                    ToolMessage(content="private tool data", tool_call_id="x"),
                    HumanMessage(content="How did I perform?")]
        with patch("maya.framework.conversation.history.conversation_config", return_value=ConversationConfig(message_limit=3)):
            dialogue = recent_dialogue(messages)
        self.assertEqual(len(dialogue), 3)
        self.assertEqual(dialogue[0].content, "Start my routine")
        self.assertEqual(dialogue[1].tool_calls, [])
        self.assertEqual(dialogue[-1].content, "How did I perform?")
