"""Reference provenance and separate display/speech delivery, without external services."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from maya.framework.conversation.responses import (
    health_presentation, load_response_policy, opening_response, response_text,
)
from maya.framework.supervisor import run_with_vad as runner


class ResponsePresentationTests(unittest.TestCase):
    def setUp(self):
        load_response_policy.cache_clear()
        self.addCleanup(load_response_policy.cache_clear)
        self.enterContext(patch.dict(os.environ, {"MAYA_RESPONSE_POLICY": ""}))
        self.policy = load_response_policy()
        self.citations = [
            {"title": f"Retrieved study {i}", "source_path": f"/private/corpus/paper-{i}.pdf", "page_number": i}
            for i in range(3)
        ]

    def test_second_instance_preserves_owner_pid_and_lock_releases_on_close(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "runner.lock"
            with patch.dict(os.environ, {"MAYA_RUN_LOCK": str(path)}):
                first = runner.acquire_single_instance_lock()
                self.assertIsNotNone(first)
                try:
                    with patch("builtins.print") as output:
                        self.assertIsNone(runner.acquire_single_instance_lock())
                    self.assertEqual(path.read_text(), str(os.getpid()))
                    self.assertIn(f"pid={os.getpid()}", output.call_args.args[0])
                finally:
                    first.close()
                second = runner.acquire_single_instance_lock()
                self.assertIsNotNone(second)
                second.close()

    def test_all_retrieved_sources_displayed_but_not_spoken(self):
        result = health_presentation("An **answer** [1].", self.citations)
        self.assertEqual(len(result["references"]), 3)
        for i in range(3):
            self.assertIn(f"Retrieved study {i}", result["display_text"])
            self.assertIn(f"PDF page {i + 1}", result["display_text"])
            self.assertNotIn(f"Retrieved study {i}", result["speech_text"])
        self.assertNotIn("/private/", json.dumps(result))
        self.assertNotIn("[1]", result["speech_text"])
        self.assertTrue(result["display_text"].endswith(self.policy.health_disclaimer))
        self.assertTrue(result["speech_text"].endswith(self.policy.health_disclaimer))
        self.assertEqual(result["display_text"].count(self.policy.health_disclaimer), 1)

    def test_duplicates_missing_metadata_and_no_sources_are_honest(self):
        result = health_presentation("Answer.", [self.citations[0], self.citations[0], {}])
        self.assertEqual(len(result["references"]), 2)
        self.assertIsNone(result["references"][1]["page"])
        self.assertIn(self.policy.unknown_source_label, result["display_text"])
        result = health_presentation("Insufficient evidence.", [])
        self.assertEqual(result["references"], [])
        self.assertIn(self.policy.no_references_text, result["display_text"])
        self.assertNotIn(self.policy.no_references_text, result["speech_text"])

    def test_partner_configuration_controls_wording_without_heading_parsing(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            values = self.policy.model_dump(mode="json")
            values.update(references_heading="Fuentes", disclaimer_heading="Aviso",
                          health_disclaimer="Consulta con tu equipo de atención.",
                          ai_disclosure="Soy un asistente de IA.", page_label="Página",
                          privacy_notice_url="https://example.org/pilot-privacy")
            path.write_text(json.dumps(values))
            with patch.dict(os.environ, {"MAYA_RESPONSE_POLICY": str(path)}):
                load_response_policy.cache_clear()
                result = health_presentation("Respuesta.", self.citations)
                self.assertIn("Fuentes:", result["display_text"])
                self.assertEqual(result["speech_text"], "Respuesta.\n\nConsulta con tu equipo de atención.")
                opening = opening_response()
                self.assertIn(values["privacy_notice_url"], opening.display_text)
                self.assertEqual(opening.speech_text, values["ai_disclosure"])
                values["health_disclaimer"] = " "
                path.write_text(json.dumps(values))
                load_response_policy.cache_clear()
                with self.assertRaises(ValidationError):
                    load_response_policy()

    def test_stream_to_local_and_mirror_uses_separate_speech_channel(self):
        presentation = health_presentation("Helpful answer.", self.citations)
        for message in [
            AIMessage(content="Helpful answer.", additional_kwargs={"presentation": presentation}),
            {"role": "assistant", "content": "Helpful answer.", "additional_kwargs": {"presentation": presentation}},
        ]:
            app = Mock()
            app.stream.return_value = [((), "updates", {"health_rag": {"messages": [message]}})]
            reply, interrupts = runner.stream_updates(app, {}, {})
            self.assertEqual(interrupts, [])
            mirror = Mock()
            runner.speak_output(None, mirror, reply.display_text, speech_text=reply.speech_text, end_interaction=True)
            mirror.show_text.assert_called_once_with(presentation["display_text"])
            mirror.speak_text.assert_called_once_with(presentation["speech_text"])
            mirror.end_interaction.assert_called_once()
            with patch.object(runner, "speak") as speak:
                runner.speak_output(None, None, reply.display_text, speech_text=reply.speech_text)
            speak.assert_called_once_with(None, presentation["speech_text"])

    def test_plain_reply_and_interrupt_contract_are_preserved(self):
        self.assertEqual(response_text("Hello!", {}).speech_text, "Hello!")
        interrupt = SimpleNamespace(value="Which result?")
        app = Mock()
        app.stream.return_value = [((), "updates", {"__interrupt__": [interrupt]})]
        reply, interrupts = runner.stream_updates(app, {}, {})
        self.assertIsNone(reply)
        self.assertEqual(interrupts, [interrupt])

    def test_main_waits_for_user_before_speaking_and_prints_full_health_reply(self):
        presentation = health_presentation("Helpful answer.", self.citations)
        reply = response_text("Helpful answer.", {"presentation": presentation})
        events = []

        def read(*args, **kwargs):
            events.append("input")
            return "What is cardiotoxicity?" if events.count("input") == 1 else None

        def speak(*args, **kwargs):
            events.append((args[2], kwargs["speech_text"]))

        with patch.object(runner.os, "chdir"), patch.object(runner.atexit, "register"), \
                patch.object(runner, "acquire_single_instance_lock", return_value=Mock()), \
                patch.object(runner, "build_supervisor"), patch.object(runner, "init_phyxio_io", return_value=None), \
                patch.object(runner, "init_asr", return_value=(None, None)), patch.object(runner, "init_tts"), \
                patch.object(runner, "read_text", side_effect=read), \
                patch.object(runner, "speak_output", side_effect=speak), \
                patch.object(runner, "stream_updates", return_value=(reply, [])), patch("builtins.print") as output:
            runner.main(runner.SessionConfig(user_id="offline-presentation"))
        self.assertEqual(events, ["input", (reply.display_text, reply.speech_text), "input"])
        self.assertFalse(any(self.policy.ai_disclosure in call.args[0] for call in output.call_args_list))
        self.assertTrue(any(reply.display_text in call.args[0] for call in output.call_args_list))

    def test_startup_and_exit_without_input_never_speaks_locally_or_on_mirror(self):
        for mirror in (None, Mock()):
            with self.subTest(mirror=mirror is not None), \
                    patch.object(runner.os, "chdir"), patch.object(runner.atexit, "register"), \
                    patch.object(runner, "acquire_single_instance_lock", return_value=Mock()), \
                    patch.object(runner, "build_supervisor"), \
                    patch.object(runner, "init_phyxio_io", return_value=mirror), \
                    patch.object(runner, "init_asr", return_value=(None, None)), \
                    patch.object(runner, "init_tts"), \
                    patch.object(runner, "read_text", return_value=None), \
                    patch.object(runner, "speak_output") as speak, \
                    patch.object(runner, "stream_updates") as stream, patch("builtins.print"):
                runner.main(runner.SessionConfig(user_id="offline-silent-startup"))
            speak.assert_not_called()
            stream.assert_not_called()
            if mirror is not None:
                mirror.show_text.assert_not_called()
                mirror.speak_text.assert_not_called()


if __name__ == "__main__":
    unittest.main()
