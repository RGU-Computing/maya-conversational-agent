"""Configurable presentation shared by the conversational UI and speech runner."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from maya.framework.conversation.text import strip_citations_and_references, strip_markdown


class ResponsePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, str_min_length=1)

    ai_disclosure: str
    health_disclaimer: str
    references_heading: str
    disclaimer_heading: str
    no_references_text: str
    unknown_source_label: str
    page_label: str
    privacy_notice_label: str
    privacy_notice_url: HttpUrl | None
    clarification_unavailable_text: str
    request_unavailable_text: str
    health_output_instruction: str
    health_processing_status: str
    health_processing_delay_seconds: float = Field(ge=0, allow_inf_nan=False)


@lru_cache(maxsize=1)
def load_response_policy() -> ResponsePolicy:
    """Load once per process; partners can supply their own complete JSON file."""
    path = Path(os.environ.get("MAYA_RESPONSE_POLICY") or Path(__file__).with_name("response_policy.json"))
    return ResponsePolicy.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ResponseText:
    display_text: str
    speech_text: str


def response_text(content: str, extra: dict[str, Any]) -> ResponseText:
    """Use the explicit speech channel so reference headings need no text parsing."""
    if presentation := extra.get("presentation"):
        return ResponseText(presentation["display_text"], presentation["speech_text"])
    return ResponseText(content, strip_citations_and_references(strip_markdown(content)))


def opening_response() -> ResponseText:
    policy = load_response_policy()
    display = policy.ai_disclosure
    if policy.privacy_notice_url:
        display += f"\n\n{policy.privacy_notice_label}: {policy.privacy_notice_url}"
    return ResponseText(display, policy.ai_disclosure)


def health_presentation(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """Render only retriever metadata, without exposing local paths or inventing links."""
    policy = load_response_policy()
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
