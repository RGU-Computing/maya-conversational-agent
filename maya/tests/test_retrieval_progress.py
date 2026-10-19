"""Progress delivery and speech ordering without network or audio hardware."""

import importlib.util
import queue
import sys
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END, MessagesState

from maya.framework.conversation.responses import load_response_policy
from maya.framework.supervisor import run_with_vad as runner


PROGRESS = ((), "custom", {"event": "health_processing_started"})
ANSWER = ((), "updates", {"health_rag": {"messages": [AIMessage(content="Final answer.")]}})


class RetrievalProgressTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_response_policy().model_copy(update={"health_processing_delay_seconds": 0.01})
        self.status = self.policy.health_processing_status
        self.enterContext(patch.object(runner, "load_response_policy", return_value=self.policy))
        self.acknowledged = Event()
        self.app = Mock()
        self.mirror = None
        self.tts = Mock()
        self.enterContext(patch.object(runner.os, "chdir"))
        self.enterContext(patch.object(runner.atexit, "register"))
        self.enterContext(patch.object(runner, "acquire_single_instance_lock", return_value=Mock()))
        self.enterContext(patch.object(runner, "build_supervisor", side_effect=lambda: self.app))
        self.enterContext(patch.object(runner, "init_phyxio_io", side_effect=lambda _: self.mirror))
        self.enterContext(patch.object(runner, "init_asr", return_value=(None, None)))
        self.enterContext(patch.object(runner, "init_tts", return_value=self.tts))
        self.read = self.enterContext(patch.object(runner, "read_text", side_effect=["Question", None]))
        self.printed = self.enterContext(patch("builtins.print"))
        self.printed.side_effect = lambda *args, **kwargs: (
            self.acknowledged.set() if args and str(args[0]).startswith("MAYA: ") else None)

    def run_main(self):
        runner.main(runner.SessionConfig(user_id="offline-progress"))

    def slow_stream(self, *chunks):
        self.acknowledged.clear()
        for chunk in chunks:
            yield chunk
            if chunk == PROGRESS:
                self.assertTrue(self.acknowledged.wait(5), "Slow work should trigger acknowledgement")

    def test_real_stream_runs_retrieval_during_speech_and_waits_before_final(self):
        started, retrieved = Event(), Event()
        order = []

        def retrieve(state):
            get_stream_writer()({"event": "health_processing_started"})
            self.assertTrue(started.wait(5), "Progress must arrive before retrieval completes")
            order.append("retrieved")
            retrieved.set()
            return {"messages": [AIMessage(content="Final answer.")]}

        self.app = (StateGraph(MessagesState).add_node("retrieve", retrieve)
                    .add_edge(START, "retrieve").add_edge("retrieve", END).compile())

        def speak(tts, text):
            if text == self.status:
                order.append("status started")
                started.set()
                self.assertTrue(retrieved.wait(5), "Speech must not block retrieval")
                order.append("status finished")
            else:
                order.append(text)

        with patch.object(runner, "speak", side_effect=speak):
            self.run_main()
        self.assertEqual(order, ["status started", "retrieved", "status finished", "Final answer."])

    def test_fast_result_duplicate_events_and_new_turn(self):
        self.app.stream.side_effect = [[PROGRESS, PROGRESS, ANSWER], [PROGRESS, ANSWER]]
        self.read.side_effect = ["Question one", "Question two", None]
        with patch.object(runner, "speak") as speak:
            self.run_main()
        self.assertEqual([c.args[1] for c in speak.call_args_list],
                         ["Final answer.", "Final answer."])

    def test_retry_does_not_repeat_acknowledgement(self):
        def failed_stream(*args, **kwargs):
            yield from self.slow_stream(PROGRESS)
            raise RuntimeError("recoverable history error")

        self.app.stream.side_effect = [failed_stream(), iter([PROGRESS, ANSWER])]
        with patch.object(runner, "is_tool_call_history_error", return_value=True), \
                patch.object(runner, "speak") as speak:
            self.run_main()
        self.assertEqual([c.args[1] for c in speak.call_args_list], [self.status, "Final answer."])

    def test_progress_finishes_before_clarification_and_does_not_repeat_on_resume(self):
        interrupt = ((), "updates", {"__interrupt__": [SimpleNamespace(value="Which treatment?")]})
        self.app.stream.side_effect = [self.slow_stream(PROGRESS, interrupt), [PROGRESS, ANSWER]]
        inputs = iter(["Question", None])

        def read(prompt, *args, **kwargs):
            if prompt:
                runner.speak(self.tts, prompt)
                return "Doxorubicin"
            return next(inputs)

        self.read.side_effect = read
        with patch.object(runner, "speak") as speak:
            self.run_main()
        self.assertEqual(self.read.call_args_list[1].args[0], "Which treatment?")
        self.assertEqual([c.args[1] for c in speak.call_args_list], [self.status, "Which treatment?", "Final answer."])
        self.assertEqual(self.app.stream.call_args_list[1].args[0].resume, "Doxorubicin")

    def test_unrelated_events_and_no_retrieval_do_not_announce(self):
        self.app.stream.return_value = [
            ((), "custom", "Other component status"),
            ((), "custom", {"event": "other_event"}), ANSWER,
        ]
        with patch.object(runner, "speak") as speak:
            self.run_main()
        speak.assert_called_once_with(self.tts, "Final answer.")

    def test_progress_tts_failure_does_not_stop_final_delivery(self):
        self.app.stream.return_value = self.slow_stream(PROGRESS, ANSWER)
        self.tts.speak_text_async.side_effect = [RuntimeError("TTS offline"), Mock(get=Mock(return_value=SimpleNamespace(audio_data=None)))]
        self.run_main()  # Real speak() handles the synthesis failure.
        self.assertEqual([c.args[0] for c in self.tts.speak_text_async.call_args_list],
                         [self.status, "Final answer."])

    def test_missing_tts_still_displays_status_and_answer(self):
        self.tts = None
        self.app.stream.return_value = self.slow_stream(PROGRESS, ANSWER)
        self.run_main()
        output = "\n".join(str(c.args) for c in self.printed.call_args_list)
        self.assertIn(self.status, output)
        self.assertIn("Final answer.", output)

    def test_graph_failure_joins_progress_and_does_not_speak_an_answer(self):
        finished = Event()

        def fail(*args, **kwargs):
            yield from self.slow_stream(PROGRESS)
            raise RuntimeError("retrieval failed")

        self.app.stream.side_effect = fail
        with patch.object(runner, "speak", side_effect=lambda *_: finished.set()) as speak:
            with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
                self.run_main()
        self.assertTrue(finished.is_set())
        speak.assert_called_once_with(self.tts, self.status)

    def test_mirror_uses_existing_queue_and_ends_only_after_final_submission(self):
        self.mirror = Mock()
        self.app.stream.return_value = self.slow_stream(PROGRESS, PROGRESS, ANSWER)
        with patch.object(runner, "speak") as local_speech:
            self.run_main()
        local_speech.assert_not_called()
        self.assertEqual([c.args[0] for c in self.mirror.speak_text.call_args_list],
                         [self.status, "Final answer."])
        self.mirror.end_interaction.assert_called_once()
        self.assertEqual(self.mirror.mock_calls[-1][0], "end_interaction")

    def test_partner_policy_controls_status_text(self):
        policy = self.policy.model_copy(update={"health_processing_status": "Un momento, por favor."})
        self.app.stream.return_value = self.slow_stream(PROGRESS, ANSWER)
        with patch.object(runner, "load_response_policy", return_value=policy), patch.object(runner, "speak") as speak:
            self.run_main()
        self.assertEqual(speak.call_args_list[0].args[1], policy.health_processing_status)

    def test_fast_clarification_cancels_timer_but_slow_resume_can_announce(self):
        interrupt = ((), "updates", {"__interrupt__": [SimpleNamespace(value="Which treatment?")]})
        self.app.stream.side_effect = [[PROGRESS, interrupt], self.slow_stream(PROGRESS, ANSWER)]
        inputs = iter(["Question", None])

        def read(prompt, *args, **kwargs):
            if prompt:
                self.assertFalse(self.acknowledged.is_set())
                runner.speak(self.tts, prompt)
                return "Doxorubicin"
            return next(inputs)

        self.read.side_effect = read
        with patch.object(runner, "speak") as speak:
            self.run_main()
        self.assertEqual([c.args[1] for c in speak.call_args_list],
                         ["Which treatment?", self.status, "Final answer."])

    def test_fast_mirror_response_never_queues_acknowledgement(self):
        self.mirror = Mock()
        self.app.stream.return_value = [PROGRESS, ANSWER]
        self.run_main()
        self.mirror.speak_text.assert_called_once_with("Final answer.")

    def test_graph_failure_cancels_pending_acknowledgement_without_waiting_for_delay(self):
        from time import monotonic

        def fail(*args, **kwargs):
            yield PROGRESS
            raise RuntimeError("retrieval failed")

        self.app.stream.side_effect = fail
        policy = self.policy.model_copy(update={"health_processing_delay_seconds": 30})
        with patch.object(runner, "load_response_policy", return_value=policy), patch.object(runner, "speak") as speak:
            started = monotonic()
            with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
                self.run_main()
            self.assertLess(monotonic() - started, 2)
        speak.assert_not_called()

    def test_delay_configuration_rejects_negative_or_nonfinite_values(self):
        from pydantic import ValidationError
        from maya.framework.conversation.responses import ResponsePolicy

        for value in (-1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ResponsePolicy.model_validate({**self.policy.model_dump(), "health_processing_delay_seconds": value})


class MirrorSpeechStateTests(unittest.TestCase):
    def setUp(self):
        # Load the real worker with only unavailable ICE/Phyxio imports substituted.
        name = "maya.agents.phyxio_exercise_agent._offline_speech"
        path = Path(__file__).resolve().parents[1] / "framework/integrations/phyxio/service.py"
        spec = importlib.util.spec_from_file_location(name, path)
        self.module = importlib.util.module_from_spec(spec)
        self.enterContext(patch.dict(sys.modules, {
            name: self.module, "Ice": Mock(),
            "phyxio": SimpleNamespace(connect=SimpleNamespace(PhyxioConnector=object)),
        }))
        spec.loader.exec_module(self.module)
        self.connector = object.__new__(self.module.PhyxioService)
        c = self.connector
        c._tts_running = Event()
        c._cancel_recording = Event()
        c._tts_queue = queue.Queue()
        c._requested_agent_state = "idle"
        c._tts = Mock()
        c._tts.speak_text_async.return_value.get.return_value = SimpleNamespace(audio_data=b"audio")
        c._agent_set_state = Mock()
        c._asr_set_state = Mock()
        c.cooldown_s = 0
        c._cooldown_until = 0
        c.client_connected = True
        c._lock = Lock()
        c._recording = False

    def play_queue(self, wait_callback=lambda: None):
        class StopWorker(BaseException):
            pass

        c = self.connector
        pending = [c._tts_queue.get_nowait() for _ in range(c._tts_queue.qsize())]
        with patch.object(c._tts_queue, "get", side_effect=[*pending, StopWorker()]), \
                patch.object(self.module.sf, "read", return_value=([0], 24000)), \
                patch.object(self.module.sd, "play"), \
                patch.object(self.module.sd, "wait", side_effect=wait_callback):
            with self.assertRaises(StopWorker):
                c._tts_loop()

    def test_acknowledgement_restores_thinking_then_final_restores_idle(self):
        c = self.connector
        c.set_agent_state("thinking")
        c.speak_text("Checking information.")
        self.play_queue()
        self.assertEqual([call.args[0] for call in c._agent_set_state.call_args_list],
                         ["thinking", "talking", "thinking"])
        c.set_agent_state("idle")
        c.speak_text("Final answer.")
        self.play_queue()
        self.assertEqual(c._agent_set_state.call_args.args, ("idle",))
        self.assertEqual([call.args[0] for call in c._tts.speak_text_async.call_args_list],
                         ["Checking information.", "Final answer."])

    def test_completion_during_playback_defers_idle_until_audio_finishes(self):
        c = self.connector
        c.set_agent_state("thinking")
        c.speak_text("Checking information.")
        c.speak_text("Final answer.")
        states = []

        def while_playing():
            c.set_agent_state("idle")
            states.append(c._agent_set_state.call_args.args[0])

        self.play_queue(while_playing)
        self.assertEqual(states, ["talking", "talking"])
        self.assertEqual(c._agent_set_state.call_args.args, ("idle",))
        self.assertFalse(c._tts_running.is_set())

    def test_synthesis_failure_restores_state_and_allows_next_message(self):
        c = self.connector
        c.set_agent_state("thinking")
        c.speak_text("Checking information.")
        c.speak_text("Final answer.")
        c._tts.speak_text_async.side_effect = [RuntimeError("offline"), Mock(get=Mock(return_value=SimpleNamespace(audio_data=b"audio")))]
        with self.assertLogs(level="ERROR"):
            self.play_queue()
        self.assertEqual(c._tts.speak_text_async.call_count, 2)
        self.assertFalse(c._tts_running.is_set())
        self.assertEqual(c._agent_set_state.call_args.args, ("thinking",))

    def test_recording_blocked_during_processing_and_queued_speech(self):
        c = self.connector
        c.set_agent_state("thinking")
        self.assertFalse(c._handle_interaction_state_request(True)[0])
        c.set_agent_state("idle")
        c.speak_text("Queued answer.")
        self.assertFalse(c._handle_interaction_state_request(True)[0])


if __name__ == "__main__":
    unittest.main()
