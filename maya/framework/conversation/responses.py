"""Configurable presentation shared by the conversational UI and speech runner."""

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from maya.framework.conversation.personalization import Preferences, RESPONSE_LANGUAGES
from maya.framework.conversation.text import strip_citations_and_references, strip_markdown
from maya.framework.integrations.models import get_chat_model, model_call


class ResponseMessages(BaseModel):
    """Only application-owned, user-facing text is eligible for translation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, str_min_length=1)

    ai_disclosure: str
    health_disclaimer: str
    references_heading: str
    disclaimer_heading: str
    no_references_text: str
    unknown_source_label: str
    page_label: str
    privacy_notice_label: str
    clarification_unavailable_text: str
    request_unavailable_text: str
    health_processing_status: str


class ResponsePolicy(ResponseMessages):
    source_language: str
    localization_instruction: str
    privacy_notice_url: HttpUrl | None
    health_output_instruction: str
    health_processing_delay_seconds: float = Field(ge=0, allow_inf_nan=False)


@lru_cache(maxsize=1)
def load_response_policy() -> ResponsePolicy:
    """Load once per process; partners can supply their own complete JSON file."""
    path = Path(os.environ.get("MAYA_RESPONSE_POLICY") or Path(__file__).with_name("response_policy.json"))
    return ResponsePolicy.model_validate_json(path.read_text(encoding="utf-8"))


class Notice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, str_min_length=1)
    text: str


class TranslatedMessages(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, str_min_length=1)
    texts: list[str]


Messages = TypeVar("Messages", bound=BaseModel)


@lru_cache(maxsize=128)
def _translated_messages(source: tuple[str, ...], instruction: str,
                         model_key: tuple[str, str, str]) -> tuple[str, ...]:
    # Cache immutable strings, not a mutable model shared between sessions. Source
    # wording, language instruction and deployment identity form the cache key.
    translated = model_call(get_chat_model(), TranslatedMessages, strict=True).invoke([
        SystemMessage(content=instruction),
        HumanMessage(content=json.dumps(source, ensure_ascii=False)),
    ])
    texts = TranslatedMessages.model_validate(translated).texts
    if len(texts) != len(source):
        raise ValueError("Translation must return exactly one text per source message.")
    return tuple(texts)


def localize_messages(messages: Messages, response_language: str | None = None) -> Messages:
    """Translate configured notices with the configured model, once per text/language.

    The policy declares the source language shared by application message files.
    Failures propagate: never silently substitute an English notice. Call this
    before work that needs a localized error message, so an outage needs no LLM.
    """
    preferences = Preferences(**({"response_language": response_language} if response_language is not None else {}))
    policy = load_response_policy()
    if preferences.response_language == policy.source_language:
        return messages
    # These values are notices, not retrieved evidence or generated answers. Do
    # not mix the chat prompt's quoted-evidence preservation rule into translation.
    instruction = (policy.localization_instruction + "\n\nSource message language: " + policy.source_language
                   + "\nTarget response language: "
                   + RESPONSE_LANGUAGES.languages[preferences.response_language])
    source = messages.model_dump(mode="json")
    model = get_chat_model()
    model_key = (model.azure_endpoint, model.deployment_name, model.openai_api_version)
    texts = _translated_messages(tuple(source.values()), instruction, model_key)
    return type(messages).model_validate(dict(zip(source, texts, strict=True)))


def localize_text(text: str, response_language: str | None = None) -> str:
    return localize_messages(Notice(text=text), response_language).text if text else text


def response_policy(response_language: str | None = None) -> ResponsePolicy:
    policy = load_response_policy()
    messages = ResponseMessages.model_validate(policy.model_dump(include=ResponseMessages.model_fields.keys()))
    return policy.model_copy(update=localize_messages(messages, response_language).model_dump())


@dataclass(frozen=True)
class ResponseText:
    display_text: str
    speech_text: str


def response_text(content: str, extra: dict[str, Any]) -> ResponseText:
    """Use the explicit speech channel so reference headings need no text parsing."""
    if presentation := extra.get("presentation"):
        return ResponseText(presentation["display_text"], presentation["speech_text"])
    return ResponseText(content, strip_citations_and_references(strip_markdown(content)))


def opening_response(response_language: str | None = None) -> ResponseText:
    policy = response_policy(response_language)
    display = policy.ai_disclosure
    if policy.privacy_notice_url:
        display += f"\n\n{policy.privacy_notice_label}: {policy.privacy_notice_url}"
    return ResponseText(display, policy.ai_disclosure)


def health_presentation(answer: str, citations: list[dict[str, Any]], response_language: str | None = None) -> dict[str, Any]:
    """Render only retriever metadata, without exposing local paths or inventing links."""
    policy = response_policy(response_language)
    references = []
    for citation in citations:
        filename = Path(citation.get("source_path") or "").name
        page = citation.get("page_number")
        reference = {
            "title": citation.get("title") or filename or policy.unknown_source_label,
            "file_name": filename,
            "page": page + 1 if isinstance(page, int) and page >= 0 else None,
        }
        if reference not in references:
            references.append(reference)

    lines = []
    for number, reference in enumerate(references, 1):
        details = [reference["title"]]
        if reference["file_name"] and reference["file_name"] != reference["title"]:
            details.append(reference["file_name"])
        if reference["page"] is not None:
            details.append(f"{policy.page_label} {reference['page']}")
        lines.append(f"[{number}] " + " — ".join(details))

    # The model supplies the answer; application-owned references and notices are
    # attached afterwards and never sent through reference-heading stripping.
    answer = strip_citations_and_references(strip_markdown(answer))
    source_text = "\n".join(lines) or policy.no_references_text
    return {
        "answer_text": answer,
        "references": references,
        "disclaimer": policy.health_disclaimer,
        "display_text": f"{answer}\n\n{policy.references_heading}:\n{source_text}\n\n{policy.disclaimer_heading}:\n{policy.health_disclaimer}",
        "speech_text": f"{answer}\n\n{policy.health_disclaimer}",
    }
