"""Speech setup shared by the local runner and the Phyxio connector."""

from __future__ import annotations

import os
from io import BytesIO

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI, OpenAI

from maya.framework.conversation.personalization import Preferences, load_session


def init_asr() -> tuple[object | None, str | None]:
    if key := os.getenv("OPENAI_WHISPER_API_KEY"):
        return OpenAI(api_key=key), os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")

    key = os.getenv("AZURE_OPENAI_WHISPER_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_WHISPER")
    version = os.getenv("AZURE_OPENAI_WHISPER_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-06-01"
    if key and endpoint and deployment:
        return AzureOpenAI(
            api_key=key, api_version=version, azure_endpoint=endpoint.split("/openai/")[0]
        ), deployment
    return None, None


def init_tts(preferences: Preferences | None = None) -> speechsdk.SpeechSynthesizer | None:
    preferences = preferences or load_session().preferences
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        return None
    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.speech_synthesis_voice_name = preferences.tts_voice
    config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    return speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)


def transcribe(client, model: str, wav: BytesIO, preferences: Preferences) -> str:
    language = {} if preferences.asr_language == "auto" else {"language": preferences.asr_language}
    result = client.audio.transcriptions.create(
        model=model, file=("input.wav", wav, "audio/wav"), **language
    )
    return (getattr(result, "text", None) or "").strip()
