"""Startup contract shared by onboarding, the runner and speech adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PersonalityPrompts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    default: str
    supportive: str = Field(min_length=1)
    concise: str = Field(min_length=1)


PERSONALITIES = PersonalityPrompts.model_validate_json(
    Path(__file__).with_name("personalities.json").read_text(encoding="utf-8")
).model_dump()


class ResponseLanguages(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    default_response_language: str
    languages: dict[str, str]
    instruction_template: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.default_response_language not in self.languages or any(
            not code or not name for code, name in self.languages.items()
        ):
            raise ValueError("The default response language must have a supported code and full name.")
        if "{language}" not in self.instruction_template:
            raise ValueError("The response-language instruction must contain {language}.")
        try:
            self.instruction_template.format(language=self.languages[self.default_response_language])
        except (KeyError, IndexError, ValueError):
            raise ValueError("Invalid response-language template; use the {language} placeholder.") from None
        return self


RESPONSE_LANGUAGES = ResponseLanguages.model_validate_json(
    Path(__file__).with_name("response_languages.json").read_text(encoding="utf-8")
)


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    personality_id: Literal["default", "supportive", "concise"] = "supportive"
    response_language: str = Field(
        default=RESPONSE_LANGUAGES.default_response_language,
        description="Language of generated user-facing text; independent of ASR language and TTS voice.",
        json_schema_extra={"enum": list(RESPONSE_LANGUAGES.languages)},
    )
    tts_voice: str = Field(
        default="en-GB-RyanNeural", min_length=1,
        description="Azure voice name available in the deployment's Speech resource.",
        examples=["en-GB-RyanNeural", "en-GB-LibbyNeural"],
    )
    asr_language: str = Field(
        default="auto", pattern=r"^(auto|[a-z]{2})$",
        description="auto, or a Whisper-supported ISO 639-1 language code, e.g. en or es.",
    )
    emergency_contact: str | None = Field(
        default=None, pattern=r"^\+?[0-9](?:[ -]?[0-9]){2,14}$",
        description="Pilot-approved emergency phone number for urgent guidance.",
    )

    @field_validator("response_language")
    @classmethod
    def supported_response_language(cls, value: str) -> str:
        if value not in RESPONSE_LANGUAGES.languages:
            raise ValueError("Unsupported response_language; supported codes: "
                             + ", ".join(RESPONSE_LANGUAGES.languages))
        return value

    def language_instruction(self) -> str:
        return RESPONSE_LANGUAGES.instruction_template.format(
            language=RESPONSE_LANGUAGES.languages[self.response_language]
        )

    def response_style(self) -> str:
        """Shared user-facing instructions, also included in the answer cache scope."""
        instruction = PERSONALITIES[self.personality_id]
        tone = (
            "\n\nRESPONSE STYLE (user-facing wording only)\n"
            + instruction
            + " Keep all evidence, safety guidance, required detail and tool policies unchanged."
        ) if instruction else ""
        return tone + "\n\n" + self.language_instruction()


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
    """JSON-serializable schema and preference options for UCLM's onboarding UI."""
    return {"schema": SessionConfig.model_json_schema(), "personalities": PERSONALITIES.copy(),
            "languages": RESPONSE_LANGUAGES.languages.copy(),
            "default_response_language": RESPONSE_LANGUAGES.default_response_language}
