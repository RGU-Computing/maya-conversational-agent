"""Onboarding and speech contracts, with no network, database, microphone or ICE."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from pydantic import ValidationError

from maya.framework.conversation.personalization import (
    Preferences, SessionConfig, load_session, onboarding_options,
)
from maya.framework.integrations import speech
from maya.framework.supervisor import run_with_vad as runner
from maya.framework.integrations.phyxio import bridge


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_missing_or_blank_identity_is_rejected(self):
        with self.assertRaises(ValidationError):
            load_session()
        for user_id in ("", "   "):
            with patch.dict(os.environ, {"MAYA_RUN_USER_ID": user_id}):
                with self.assertRaises(ValidationError):
                    load_session()

    def test_environment_identity_and_preference_defaults(self):
        with patch.dict(os.environ, {"MAYA_RUN_USER_ID": "participant-a"}):
            session = load_session()
        self.assertEqual(session.user_id, "participant-a")
        self.assertEqual(session.preferences, Preferences())
        self.assertEqual(session.preferences.personality_id, "supportive")
        self.assertEqual(SessionConfig(user_id="a").preferences.personality_id, "supportive")
        with patch.dict(os.environ, {
            "MAYA_RUN_USER_ID": "participant-a", "MAYA_PERSONALITY": "concise",
            "AZURE_SPEECH_VOICE": "en-GB-LibbyNeural", "MAYA_ASR_LANGUAGE": "es",
        }):
            session = load_session()
        self.assertEqual(session.user_id, "participant-a")
        self.assertEqual(session.preferences, Preferences(
            personality_id="concise", tts_voice="en-GB-LibbyNeural", asr_language="es"
        ))

    def test_json_overrides_environment_and_missing_fields_inherit_defaults(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text(json.dumps({
                "user_id": "participant-a", "preferences": {"personality_id": "supportive"}
            }))
            with patch.dict(os.environ, {
                "MAYA_SESSION_CONFIG": str(path), "MAYA_RUN_USER_ID": "other-user",
                "MAYA_PERSONALITY": "concise", "MAYA_ASR_LANGUAGE": "es",
            }):
                session = load_session()
            self.assertEqual(session.user_id, "participant-a")
            self.assertEqual(session.preferences.personality_id, "supportive")
            self.assertEqual(session.preferences.asr_language, "es")

    def test_invalid_onboarding_payloads_fail_at_boundary(self):
        for payload in [
            {}, {"user_id": "  "}, {"user_id": 123},
            {"user_id": "a", "unexpected": True},
            {"user_id": "a", "preferences": {"personality_id": "invented"}},
            {"user_id": "a", "preferences": {"tts_voice": " "}},
            {"user_id": "a", "preferences": {"asr_language": "English"}},
            {"user_id": "a", "preferences": {"voice": "typo"}},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                SessionConfig.model_validate(payload)

    def test_file_cannot_fall_back_to_test_identity(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text('{"preferences": {}}')
            with self.assertRaises(ValidationError):
                load_session(path)
            with self.assertRaises(FileNotFoundError):
                load_session(path.with_name("missing.json"))

    def test_threads_are_fresh_and_explicit_resume_is_scoped_to_participant(self):
        first = SessionConfig(user_id="a")
        second = SessionConfig(user_id="b")
        self.assertNotEqual(first.graph_config(), first.graph_config())
        self.assertEqual(first.graph_config(thread_id="visit"), first.graph_config(thread_id="visit"))
        self.assertNotEqual(first.graph_config(thread_id="visit"), second.graph_config(thread_id="visit"))
        self.assertEqual(first.graph_config()["configurable"]["user_id"], "a")

    def test_configuration_is_immutable_and_exported_config_is_independent(self):
        session = SessionConfig(user_id="a", preferences=Preferences(personality_id="supportive"))
        with self.assertRaises(ValidationError):
            session.preferences.personality_id = "concise"
        config = session.graph_config()
        config["configurable"]["preferences"]["personality_id"] = "concise"
        self.assertEqual(session.preferences.personality_id, "supportive")

    def test_onboarding_schema_matches_supported_personalities(self):
        options = json.loads(json.dumps(onboarding_options()))
        schema = options["schema"]["$defs"]["Preferences"]
        self.assertEqual(set(schema["properties"]["personality_id"]["enum"]), set(options["personalities"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["personality_id"]["default"], "supportive")


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_missing_credentials_allow_local_text_mode(self):
        self.assertEqual(speech.init_asr(), (None, None))
        self.assertIsNone(speech.init_tts(Preferences()))

    def test_openai_asr_takes_precedence(self):
        with patch.dict(os.environ, {
            "OPENAI_WHISPER_API_KEY": "fake", "OPENAI_WHISPER_MODEL": "test-model",
            "AZURE_OPENAI_WHISPER_API_KEY": "fake-azure",
        }), patch.object(speech, "OpenAI") as openai, patch.object(speech, "AzureOpenAI") as azure:
            client, model = speech.init_asr()
        openai.assert_called_once_with(api_key="fake")
        azure.assert_not_called()
        self.assertIs(client, openai.return_value)
        self.assertEqual(model, "test-model")

    def test_azure_asr_deployment_endpoint_and_version(self):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_WHISPER_API_KEY": "fake", "AZURE_OPENAI_DEPLOYMENT_WHISPER": "whisper-test",
            "AZURE_OPENAI_WHISPER_ENDPOINT": "https://example.test/openai/deployments/whisper-test",
            "AZURE_OPENAI_API_VERSION": "test-version",
        }), patch.object(speech, "AzureOpenAI") as azure:
            client, model = speech.init_asr()
        azure.assert_called_once_with(api_key="fake", api_version="test-version", azure_endpoint="https://example.test")
        self.assertIs(client, azure.return_value)
        self.assertEqual(model, "whisper-test")

    def test_selected_voice_reaches_azure(self):
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": "fake", "AZURE_SPEECH_REGION": "test"}), \
                patch.object(speech.speechsdk, "SpeechConfig") as config, \
                patch.object(speech.speechsdk, "SpeechSynthesizer") as synthesizer:
            speech.init_tts(Preferences(tts_voice="en-GB-LibbyNeural"))
        self.assertEqual(config.return_value.speech_synthesis_voice_name, "en-GB-LibbyNeural")
        synthesizer.assert_called_once_with(speech_config=config.return_value, audio_config=None)

    def test_language_is_omitted_for_auto_and_forwarded_when_selected(self):
        client = Mock()
        client.audio.transcriptions.create.return_value.text = " Hello "
        wav = BytesIO(b"audio fixture")
        for language in ("auto", "en", "es"):
            result = speech.transcribe(client, "test", wav, Preferences(asr_language=language))
            kwargs = client.audio.transcriptions.create.call_args.kwargs
            self.assertEqual(result, "Hello")
            self.assertEqual(kwargs["file"], ("input.wav", wav, "audio/wav"))
            self.assertEqual(kwargs.get("language"), None if language == "auto" else language)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.preferences = Preferences(personality_id="supportive", tts_voice="en-GB-LibbyNeural", asr_language="es")
        self.enterContext(patch.dict(os.environ, {"MAYA_ENABLE_PHYXIO_IO": "1", "MAYA_RUN_THREAD_ID": ""}))

    def test_runner_passes_preferences_to_speech_graph_and_clarification_input(self):
        session = SessionConfig(user_id="participant-a", preferences=self.preferences)
        factory = SimpleNamespace(build_supervisor=Mock())
        with patch.dict(sys.modules, {"maya.framework.supervisor.factory": factory}), \
                patch.object(runner.os, "chdir"), patch.object(runner.atexit, "register"), \
                patch.object(runner, "acquire_single_instance_lock"), \
                patch.object(runner, "init_phyxio_io", return_value=None) as mirror, \
                patch.object(runner, "init_asr", return_value=(None, None)), \
                patch.object(runner, "init_tts", return_value=None) as tts, \
                patch.object(runner, "read_text", side_effect=["Hello", "A result", None]) as read, \
                patch.object(runner, "speak_output"), \
                patch.object(runner, "stream_updates", side_effect=[
                    (None, [SimpleNamespace(value="Which result?")]), (runner.response_text("Done.", {}), [])
                ]) as stream, patch("builtins.print"):
            runner.main(session)
        mirror.assert_called_once_with(session)
        tts.assert_called_once_with(self.preferences)
        for call in read.call_args_list:
            self.assertEqual(call.kwargs["preferences"], self.preferences)
        config = stream.call_args_list[0].args[2]["configurable"]
        self.assertEqual(config["user_id"], "participant-a")
        self.assertEqual(config["preferences"], self.preferences.model_dump())
        self.assertEqual(stream.call_args_list[1].args[1].resume, "A result")

    def test_local_vad_transcription_receives_selected_language(self):
        def input_stream(**kwargs):
            kwargs["callback"](np.full((160, 1), 16000, dtype=np.int16), 160, None, None)
            return MagicMock()

        client = Mock()
        client.audio.transcriptions.create.return_value.text = "Hola"
        with patch("builtins.input", side_effect=["/voice"]), patch("builtins.print"), \
                patch.object(runner.sd, "InputStream", side_effect=input_stream), \
                patch.object(runner.webrtcvad, "Vad") as vad:
            vad.return_value.is_speech.return_value = True
            result = runner.read_text(None, None, client, "test", runner.VoiceConfig(.01, 16000), preferences=self.preferences)
        self.assertEqual(result, "Hola")
        self.assertEqual(client.audio.transcriptions.create.call_args.kwargs["language"], "es")

    def test_mirror_singleton_keeps_runner_settings_for_later_tool_calls(self):
        session = SessionConfig(user_id="participant-a", preferences=self.preferences)
        constructor = Mock(side_effect=lambda value: SimpleNamespace(session=value, preferences=value.preferences))
        module = SimpleNamespace(PhyxioService=constructor)
        with patch.object(bridge, "_service", None), \
                patch.dict(sys.modules, {"maya.framework.integrations.phyxio.service": module}):
            with self.assertRaises(RuntimeError):
                bridge.get_phyxio_service()
            constructor.assert_not_called()
            service = runner.init_phyxio_io(session)
            self.assertIs(service, bridge.get_phyxio_service())
            self.assertIs(service, bridge.get_phyxio_service(session))
            with self.assertRaises(ValueError):
                runner.init_phyxio_io(SessionConfig(user_id="participant-a"))
            with self.assertRaises(ValueError):
                runner.init_phyxio_io(SessionConfig(user_id="participant-b", preferences=self.preferences))
        constructor.assert_called_once_with(session)

    def test_actual_connector_uses_shared_speech_settings_without_ice(self):
        class OfflineConnector:
            def __init__(self, config_path=None):
                self.routine = Mock()

        name = "maya.agents.phyxio_exercise_agent._offline_speech_adapter"
        path = Path(__file__).resolve().parents[1] / "framework" / "integrations" / "phyxio" / "service.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            name: module, "Ice": SimpleNamespace(),
            "phyxio": SimpleNamespace(connect=SimpleNamespace(PhyxioConnector=OfflineConnector)),
        }), patch("logging.basicConfig"):
            spec.loader.exec_module(module)
        client = Mock()
        client.audio.transcriptions.create.return_value.text = "Hola"
        with patch.object(module, "init_asr", return_value=(client, "test")), \
                patch.object(module, "init_tts") as tts, patch.object(module, "Thread"), \
                patch.object(module, "_list_audio_devices_once"), patch.object(module, "log"):
            service = module.PhyxioService(SessionConfig(user_id="participant-a", preferences=self.preferences))
            result = service._transcribe_wav(BytesIO(b"fixture"))
        tts.assert_called_once_with(self.preferences)
        self.assertEqual(result, "Hola")
        self.assertEqual(client.audio.transcriptions.create.call_args.kwargs["language"], "es")


if __name__ == "__main__":
    unittest.main()
