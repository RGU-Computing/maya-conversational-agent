"""Native callback binding and recorder recovery, without ICE or audio hardware."""

import importlib.util
from io import BytesIO
from pathlib import Path
import queue
import sys
from threading import Event
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from maya.framework.conversation.personalization import SessionConfig


class ConnectorContract:
    """Reproduce 26.5.29's self-bound methods, rather than mocking state setters."""

    def __init__(self):
        self.calls = Mock()
        self.ready = Event()
        self.asr = SimpleNamespace(set_state=self._asr_set_state, set_text=self.calls.asr_text)
        self.agent = SimpleNamespace(set_state=self._agent_set_state, set_text=self.calls.agent_text,
                                     start_interaction=self.calls.start, end_interaction=self.calls.end)

    def _asr_set_state(self, state):
        self.calls.asr_state(state)

    def _agent_set_state(self, state):
        self.calls.agent_state(state)

    def on_client_connected(self):
        self.ready.set()

    def stop(self):
        self.calls.stop()


class InputStream:
    def __init__(self, *, active=True, blocks=(), **kwargs):
        self.active = active
        self.blocks = blocks
        self.callback = kwargs['callback']
        self.closed = False
        self.opened = Event()

    def __enter__(self):
        self.opened.set()
        for block in self.blocks:
            self.callback(block, len(block), None, None)
        return self

    def __exit__(self, *args):
        self.active = False
        self.closed = True


class PhyxioVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = 'maya.agents.phyxio_exercise_agent._voice_contract'
        path = Path(__file__).resolve().parents[1] / 'framework/integrations/phyxio/service.py'
        spec = importlib.util.spec_from_file_location(name, path)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: cls.module, 'Ice': Mock(),
                'phyxio': SimpleNamespace(connect=SimpleNamespace(PhyxioConnector=ConnectorContract))}):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        m = self.module
        with patch.object(m, 'init_asr', return_value=(Mock(), 'test')), \
                patch.object(m, 'init_tts', return_value=Mock()), \
                patch.object(m, '_list_audio_devices_once'), patch.object(m, 'Thread'):
            self.service = m.PhyxioService(SessionConfig(user_id="voice-test"))
        self.service.voice = m.VoiceConfig(max_seconds=.08, poll_seconds=.005, mic_device=7)
        self.connector = self.service._phyxio
        self.connector.on_client_connected()
        self.calls = self.connector.calls
        self.streams = []

    def stream(self, **kwargs):
        stream = InputStream(**kwargs)
        self.streams.append(stream)
        return stream

    def finish_worker(self):
        self.service._record_thread.join(1)
        self.assertFalse(self.service._record_thread.is_alive())
        self.assertFalse(self.service._recording)

    def test_native_state_bindings_and_ready_are_preserved(self):
        self.service._asr_set_state('listening')
        self.service.set_agent_state('thinking')
        self.calls.asr_state.assert_called_once_with('listening')
        self.calls.agent_state.assert_called_once_with('thinking')
        self.assertTrue(self.connector.ready.is_set())
        self.assertTrue(self.service.client_connected)
        overrides = set(type(self.connector).__dict__) & set(ConnectorContract.__dict__)
        self.assertNotIn('_asr_set_state', overrides)
        self.assertNotIn('_agent_set_state', overrides)

    def test_missing_connection_does_not_start_recording(self):
        self.service.client_connected = False
        with patch.object(self.module.sd, 'InputStream') as stream:
            self.assertFalse(self.connector.change_interaction_state_request(True)[0])
        stream.assert_not_called()

    def test_open_failure_resets_ui_and_next_tap_succeeds(self):
        block = np.full((1600, 1), .2, dtype=np.float32)
        with patch.object(self.module.sd, 'InputStream', side_effect=self.module.sd.PortAudioError('Device unavailable', -9985)), \
                self.assertLogs(level='ERROR'):
            self.assertTrue(self.connector.change_interaction_state_request(True)[0])
            self.finish_worker()
        self.calls.asr_state.assert_called_with('stop')
        self.calls.end.assert_called_once()
        self.calls.asr_text.assert_called_with(self.service.voice.error_text)
        self.assertNotIn(call('listening'), self.calls.asr_state.call_args_list)
        with patch.object(self.module.sd, 'InputStream', side_effect=lambda **kw: self.stream(blocks=[block], **kw)) as audio, \
                patch.object(self.service, '_transcribe_wav', return_value='How did I perform?'):
            self.assertTrue(self.connector.change_interaction_state_request(True)[0])
            self.finish_worker()
        self.assertEqual(self.service.asr_queue.get_nowait(), 'How did I perform?')
        self.assertTrue(self.streams[-1].closed)
        self.assertEqual(audio.call_args.kwargs['device'], 7)
        self.calls.end.assert_called_once()  # Successful transcript keeps interaction open.

    def test_rapid_stop_start_keeps_original_cancellation_until_worker_exits(self):
        entered, release = Event(), Event()
        def capture():
            entered.set()
            release.wait(1)
            return BytesIO(), {'peak': 0, 'speech_seen': False}
        with patch.object(self.service, '_record_until_pause', side_effect=capture) as capture_mock:
            self.connector.change_interaction_state_request(True)
            try:
                self.assertTrue(entered.wait(1))
                first = self.service._record_thread
                self.assertTrue(self.connector.change_interaction_state_request(False)[0])
                self.assertFalse(self.connector.change_interaction_state_request(True)[0])
                self.assertIs(self.service._record_thread, first)
                self.assertTrue(self.service._cancel_recording.is_set())
                capture_mock.assert_called_once()
            finally:
                release.set()
                self.finish_worker()

    def test_cancellation_wakes_recorder_with_no_audio_callbacks(self):
        waiting = Event()
        original_get = queue.Queue.get
        def get(q, *args, **kwargs):
            waiting.set()
            return original_get(q, *args, **kwargs)
        with patch.object(self.module.sd, 'InputStream', side_effect=self.stream), \
                patch.object(queue.Queue, 'get', get):
            self.connector.change_interaction_state_request(True)
            self.assertTrue(waiting.wait(1))
            self.connector.change_interaction_state_request(False)
            self.finish_worker()
        self.assertTrue(self.streams[0].closed)
        self.assertTrue(self.service.asr_queue.empty())

    def test_avatar_second_tap_cancels_capture_then_allows_fresh_recording(self):
        entered, release = Event(), Event()
        def capture():
            entered.set()
            release.wait(1)
            return BytesIO(), {'peak': .2, 'speech_seen': True}
        with patch.object(self.service, '_record_until_pause', side_effect=capture), \
                patch.object(self.service, '_transcribe_wav') as transcribe:
            self.assertTrue(self.connector.change_interaction_state_request(True)[0])
            try:
                self.assertTrue(entered.wait(1))
                first = self.service._record_thread
                self.assertTrue(self.connector.change_interaction_state_request(True)[0])
                self.assertTrue(self.service._cancel_recording.is_set())
                self.assertFalse(self.connector.change_interaction_state_request(True)[0])
                self.assertIs(self.service._record_thread, first)
            finally:
                release.set()
                self.finish_worker()
        transcribe.assert_not_called()
        self.assertTrue(self.service.asr_queue.empty())
        self.calls.end.assert_called_once()
        with patch.object(self.service, '_record_until_pause', return_value=(BytesIO(), {'peak': .2, 'speech_seen': True})), \
                patch.object(self.service, '_transcribe_wav', return_value='Hello again'):
            self.assertTrue(self.connector.change_interaction_state_request(True)[0])
            self.finish_worker()
        self.assertEqual(self.service.asr_queue.get_nowait(), 'Hello again')

    def test_stalled_and_inactive_streams_finish_and_release_microphone(self):
        for active in (True, False):
            with self.subTest(active=active), \
                    patch.object(self.module.sd, 'InputStream', side_effect=lambda **kw: self.stream(active=active, **kw)), \
                    self.assertLogs(level='ERROR'):
                self.connector.change_interaction_state_request(True)
                self.finish_worker()
            self.assertTrue(self.streams[-1].closed)
            self.assertTrue(self.service.asr_queue.empty())
            self.calls.end.assert_called()

    def test_cancellation_during_transcription_discards_late_result(self):
        for active in (False, True):
            with self.subTest(active=active):
                entered, release = Event(), Event()
                def transcribe(_wav):
                    entered.set()
                    release.wait(1)
                    return 'A cancelled question'
                with patch.object(self.service, '_record_until_pause', return_value=(BytesIO(), {'peak': .2, 'speech_seen': True})), \
                        patch.object(self.service, '_transcribe_wav', side_effect=transcribe):
                    self.connector.change_interaction_state_request(True)
                    try:
                        self.assertTrue(entered.wait(1))
                        self.assertTrue(self.connector.change_interaction_state_request(active)[0])
                    finally:
                        release.set()
                        self.finish_worker()
                self.assertTrue(self.service.asr_queue.empty())
                self.assertNotIn(call('A cancelled question'), self.calls.asr_text.call_args_list)
        self.assertEqual(self.calls.end.call_count, 2)

    def test_stop_is_accepted_during_cooldown_and_thinking(self):
        self.service._cooldown_until = monotonic() + 10
        self.service._requested_agent_state = 'thinking'
        self.assertTrue(self.connector.change_interaction_state_request(False)[0])
        self.assertTrue(self.service._cancel_recording.is_set())

    def test_worker_start_failure_does_not_leave_busy_state(self):
        with patch.object(self.module, 'Thread') as thread, self.assertLogs(level='ERROR'):
            thread.return_value.start.side_effect = RuntimeError('cannot start thread')
            self.assertFalse(self.connector.change_interaction_state_request(True)[0])
        self.assertFalse(self.service._recording)
        self.calls.end.assert_called_once()

    def test_no_speech_or_transcription_failure_allows_retry(self):
        for result in ('', RuntimeError('ASR offline')):
            with self.subTest(result=repr(result)), \
                    patch.object(self.service, '_record_until_pause', return_value=(BytesIO(), {'peak': .2, 'speech_seen': True})), \
                    patch.object(self.service, '_transcribe_wav', side_effect=[result]), \
                    patch.object(self.module.log, 'exception'):
                self.connector.change_interaction_state_request(True)
                self.finish_worker()
                self.assertTrue(self.service.asr_queue.empty())
        self.assertEqual(self.calls.end.call_count, 2)

    def test_ui_failure_does_not_leave_recording_busy(self):
        self.calls.asr_state.side_effect = RuntimeError('UI offline')
        with patch.object(self.service, '_record_until_pause', return_value=(BytesIO(), {'peak': 0, 'speech_seen': False})), \
                self.assertLogs(level='WARNING'):
            self.connector.change_interaction_state_request(True)
            self.finish_worker()

    def test_invalid_capture_timings_are_rejected(self):
        for field in ('poll_seconds', 'max_seconds'):
            for value in (0, -1, float('inf'), float('nan')):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.module.VoiceConfig(**{field: value})


if __name__ == '__main__':
    unittest.main()
