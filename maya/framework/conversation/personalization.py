"""Startup contract shared by onboarding, the runner and speech adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PersonalityPrompts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    default: str
    supportive: str = Field(min_length=1)
    concise: str = Field(min_length=1)


PERSONALITIES = PersonalityPrompts.model_validate_json(
    Path(__file__).with_name("personalities.json").read_text(encoding="utf-8")
).model_dump()


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    personality_id: Literal["default", "supportive", "concise"] = "supportive"
    tts_voice: str = Field(
        default="en-GB-RyanNeural", min_length=1,
        description="Azure voice name available in the deployment's Speech resource.",
        examples=["en-GB-RyanNeural", "en-GB-LibbyNeural"],
    )
    asr_language: str = Field(
        default="auto", pattern=r"^(auto|[a-z]{2})$",
        description="auto, or a Whisper-supported ISO 639-1 language code, e.g. en or es.",
    )

    def response_style(self) -> str:
        if not (instruction := PERSONALITIES[self.personality_id]):
            return ""
        return (
            "\n\nRESPONSE STYLE (user-facing wording only)\n"
            + instruction
            + " Keep all evidence, safety guidance, required detail and tool policies unchanged."
        )


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    user_id: str = Field(min_length=1, description="Participant ID resolved by Phyxio, not a raw card number.")
    preferences: Preferences = Field(default_factory=Preferences)

    def graph_config(self, *, thread_id: str | None = None) -> dict:
        # Scope even an explicitly resumed thread to its participant.
        thread = json.dumps([self.user_id, thread_id or uuid4().hex], separators=(",", ":"))
        return {
            "configurable": {
                "user_id": self.user_id,
                "thread_id": thread,
                "preferences": self.preferences.model_dump(),
            },
            "recursion_limit": 40,
        }


def load_session(path: str | Path | None = None) -> SessionConfig:
    """Explicit JSON fields override environment defaults. A file must supply identity."""
    path = path or os.getenv("MAYA_SESSION_CONFIG")
    payload = json.loads(Path(path).read_text(encoding="utf-8")) if path else {
        "user_id": os.getenv("MAYA_RUN_USER_ID")
    }
    defaults = {
        field: os.environ[variable]
        for field, variable in (
            ("personality_id", "MAYA_PERSONALITY"),
            ("tts_voice", "AZURE_SPEECH_VOICE"),
            ("asr_language", "MAYA_ASR_LANGUAGE"),
        )
        if variable in os.environ
    }
    # Validate the external payload before merging so misspelled/invalid fields fail.
    session = SessionConfig.model_validate(payload)
    preferences = Preferences.model_validate({**defaults, **session.preferences.model_dump(exclude_unset=True)})
    return SessionConfig(user_id=session.user_id, preferences=preferences)


def preferences_from_config(config: dict | None) -> Preferences:
    return Preferences.model_validate((config or {}).get("configurable", {}).get("preferences", {}))


def onboarding_options() -> dict:
    """JSON-serializable schema and personality labels for UCLM's onboarding UI."""
    return {"schema": SessionConfig.model_json_schema(), "personalities": PERSONALITIES.copy()}
