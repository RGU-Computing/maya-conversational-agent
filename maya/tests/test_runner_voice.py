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


if __name__ == "__main__":
    unittest.main()
