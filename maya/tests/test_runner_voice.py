"""Local runner capture cleanup and retry, without hardware or speech services."""

import unittest
from time import monotonic
from unittest.mock import Mock, patch

import numpy as np

from maya.framework.supervisor import run_with_vad as runner


class RunnerVoiceTests(unittest.TestCase):
    def setUp(self):
        self.voice = runner.VoiceConfig(seconds=0.04, rate=16000, poll_seconds=0.005)
        self.enterContext(patch("builtins.print"))
        self.enterContext(patch.object(runner.os, "environ", {}))

    def read(self):
        return runner.read_text(None, None, Mock(), "test", self.voice)

    def test_device_failure_allows_typed_retry(self):
        with patch("builtins.input", side_effect=["/voice", "Typed retry"]), \
                patch.object(runner.sd, "InputStream", side_effect=runner.sd.PortAudioError("unavailable")):
            self.assertEqual(self.read(), "Typed retry")

    def test_stalled_capture_is_bounded_and_closed(self):
        stream = Mock()
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock()
        stream.active = True
        with patch("builtins.input", side_effect=["/voice", "Typed retry"]), \
                patch.object(runner.sd, "InputStream", return_value=stream):
            started = monotonic()
            self.assertEqual(self.read(), "Typed retry")
            self.assertLess(monotonic() - started, 1)
        stream.__exit__.assert_called_once()

    def test_cancellation_closes_capture_and_allows_retry(self):
        stream = Mock()
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock()
        with patch("builtins.input", side_effect=["/voice", "Typed retry"]), \
                patch.object(runner.sd, "InputStream", return_value=stream), \
                patch.object(runner.queue.Queue, "get", side_effect=KeyboardInterrupt):
            self.assertEqual(self.read(), "Typed retry")
        stream.__exit__.assert_called_once()

    def test_configured_device_and_asr_failure_allow_second_capture(self):
        streams = []
        def capture(**kwargs):
            stream = Mock()
            stream.__enter__ = Mock(return_value=stream)
            stream.__exit__ = Mock()
            kwargs["callback"](np.full((160, 1), 10000, dtype=np.int16), 160, None, None)
            streams.append(stream)
            return stream
        with patch.dict(runner.os.environ, {"MAYA_MIC_DEVICE": "6"}), \
                patch("builtins.input", side_effect=["/voice", "/voice"]), \
                patch.object(runner.sd, "InputStream", side_effect=capture) as device, \
                patch.object(runner.webrtcvad, "Vad") as vad, \
                patch.object(runner, "transcribe", side_effect=[RuntimeError("ASR unavailable"), "Hello Maya"]):
            vad.return_value.is_speech.return_value = True
            self.assertEqual(self.read(), "Hello Maya")
        self.assertEqual(device.call_count, 2)
        self.assertTrue(all(c.kwargs["device"] == 6 for c in device.call_args_list))
        self.assertTrue(all(s.__exit__.call_count == 1 for s in streams))

    def test_invalid_timings_fail_before_capture(self):
        for field in ("seconds", "poll_seconds"):
            for value in (0, -1, float("inf"), float("nan")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    runner.VoiceConfig(**{"seconds": 1, "rate": 16000, field: value})


class RunnerProviderFailureTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(runner.os, "chdir"))
        self.enterContext(patch.object(runner.atexit, "register"))
        self.enterContext(patch.object(runner, "acquire_single_instance_lock", return_value=Mock()))
        self.enterContext(patch.object(runner, "build_supervisor", return_value=Mock()))
        self.enterContext(patch.object(runner, "init_phyxio_io", return_value=Mock()))
        self.read = self.enterContext(patch.object(runner, "read_text"))
        self.speak = self.enterContext(patch.object(runner, "speak_output"))
        self.printed = self.enterContext(patch("builtins.print"))
        self.session = runner.SessionConfig(user_id="synthetic-recovery")
        self.policy = runner.load_response_policy().model_copy(update={"request_unavailable_text": "Configured retry message."})
        self.enterContext(patch.object(runner, "load_response_policy", return_value=self.policy))

    @staticmethod
    def rejected():
        import httpx
        return runner.BadRequestError("private rejected content", response=httpx.Response(
            400, request=httpx.Request("POST", "https://model.invalid")), body={"code": "content_filter"})

    def exercise(self, error, *, resumed=False):
        from types import SimpleNamespace
        from copy import deepcopy
        self.read.side_effect = ["First question", "Clarification reply", "Fresh question", None] if resumed else ["First question", "Fresh question", None]
        seen = []
        outcomes = iter(([(None, [SimpleNamespace(value="Which result?")])] if resumed else []) +
                        [error, (runner.ResponseText("Recovered", "Recovered"), [])])
        def stream(app, payload, config, *args, **kwargs):
            seen.append((payload, deepcopy(config)))
            outcome = next(outcomes)
            if isinstance(outcome, Exception): raise outcome
            return outcome
        with patch.object(runner, "stream_updates", side_effect=stream):
            runner.main(self.session)
        self.assertEqual(len(seen), 3 if resumed else 2)
        self.assertNotEqual(seen[-1][1]["configurable"]["thread_id"], seen[-2][1]["configurable"]["thread_id"])
        self.assertTrue(all(config["configurable"]["user_id"] == self.session.user_id for _, config in seen))
        spoken = [call.args[2] for call in self.speak.call_args_list]
        self.assertEqual(spoken[-2:], [self.policy.request_unavailable_text, "Recovered"])
        self.assertTrue(all(call.kwargs.get("end_interaction") for call in self.speak.call_args_list))
        self.assertNotIn("private rejected content", str(self.printed.call_args_list))

    def test_provider_rejection_does_not_exit_or_automatically_replay(self):
        self.exercise(self.rejected())

    def test_provider_rejection_during_resume_allows_fresh_question(self):
        self.exercise(self.rejected(), resumed=True)

    def test_network_timeout_and_rate_limit_allow_fresh_question(self):
        import httpx
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        request = httpx.Request("POST", "https://model.invalid")
        for error in (APIConnectionError(request=request), APITimeoutError(request=request),
                      RateLimitError("private rejected content", response=httpx.Response(429, request=request), body=None)):
            with self.subTest(error=type(error).__name__):
                self.speak.reset_mock();self.printed.reset_mock()
                self.exercise(error)


if __name__ == "__main__":
    unittest.main()
