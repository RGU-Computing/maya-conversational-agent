"""Protect known identifiers in model requests, keeping real identities local.

This is deliberately not free-text name recognition. Identity values come from
the session, saved profile and explicitly labelled data. Unknown names are not
guessed. There is no NLP model, network request or shared identity cache.
"""

from __future__ import annotations

import json
import re

from langchain.agents.middleware import AgentMiddleware, PIIMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import ensure_config
from pydantic import BaseModel


def privacy_config(config, identities):
    """Forward this participant's known identity without mutating shared config."""
    config = ensure_config() if config is None else config
    return {**config, "configurable": {**config.get("configurable", {}),
                                      "privacy_context": dict(identities or {})}}


def _map_strings(value, transform):
    if isinstance(value, BaseModel):
        return type(value).model_validate(_map_strings(value.model_dump(), transform))
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_strings(child, transform) for key, child in value.items()}
    if isinstance(value, list):
        return [_map_strings(child, transform) for child in value]
    return value


def _json_parts(text):
    """Yield ordinary text and decoded JSON embedded in a prompt or tool result."""
    decoder = json.JSONDecoder()
    offset = 0
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        if start < offset:
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        yield text[offset:start], None
        yield text[start:end], value
        offset = end
    yield text[offset:], None


class _Pseudonyms:
    def __init__(self):
        self.prefix = "PRIVATE"
        self.known: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.originals: dict[str, str] = {}
        self.pattern = None

    def add(self, value):
        if isinstance(value, str) and value.strip():
            self.known.add(value)

    def structured(self, value):
        if isinstance(value, dict):
            profile = bool({"connections", "interests", "job"} & value.keys())
            for key, child in value.items():
                if key in {"user_id", "username", "patient_id", "fhirId", "email", "phone", "address", "birthDate"}:
                    _map_strings(child, self.add)
                elif profile and key in {"name", "location"}:
                    self.add(child)
                self.structured(child)
        elif isinstance(value, list):
            for child in value:
                self.structured(child)

    def collect(self, text, detectors):
        # Short aliases avoid generating long random IDs. Choose a prefix absent
        # from every input, so user-supplied alias-like text cannot be restored.
        while self.prefix + "_" in text:
            self.prefix += "_"
        for raw, value in _json_parts(text):
            if value is not None:
                self.structured(value)
                _map_strings(value, lambda s: self.collect(s, detectors))
            else:
                # Include the resource type: a numeric patient ID must not mask
                # the same number when it occurs as a clinical measurement.
                for match in re.finditer(r"Patient/[\w.\-]+", raw):
                    self.add(match.group())
                for detector in detectors:
                    for match in detector.detector(raw):
                        self.add(match["value"])
        return text

    def alias(self, original):
        if original not in self.aliases:
            token = f"{self.prefix}_{len(self.aliases)}"
            self.aliases[original] = token
            self.originals[token] = original
        return self.aliases[original]

    def redact(self, text):
        if self.pattern is None:
            # Numeric identifiers are masked in labelled fields/references only;
            # treating every occurrence of e.g. '49' as identity would erase HR.
            values = [v for v in self.known if not v.isdecimal()]
            self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(
                re.escape(v) for v in sorted(values, key=len, reverse=True)
            ) + r")(?!\w)", re.IGNORECASE) if values else re.compile(r"(?!x)x")
        replace = lambda value: self.pattern.sub(lambda match: self.alias(match.group()), value)

        def structured(value):
            if isinstance(value, dict):
                return {key: (self.alias(child) if isinstance(child, str) and child in self.known
                              and key in {"user_id", "username", "patient_id", "fhirId", "phone"}
                              else structured(child)) for key, child in value.items()}
            if isinstance(value, list):
                return [structured(child) for child in value]
            return replace(value) if isinstance(value, str) else value

        return "".join(json.dumps(structured(value), ensure_ascii=False) if value is not None
                       else replace(raw) for raw, value in _json_parts(text))

    def restore(self, value):
        if isinstance(value, BaseModel):
            return type(value).model_validate(self.restore(value.model_dump()))
        if not self.originals:
            return value
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(token) for token in self.originals) + r")(?!\w)")

        def text(original):
            return "".join(json.dumps(_map_strings(decoded, text), ensure_ascii=False) if decoded is not None
                           else pattern.sub(lambda m: self.originals[m.group()], raw)
                           for raw, decoded in _json_parts(original))

        return _map_strings(value, text)


class PrivacyMiddleware(AgentMiddleware):
    """Last model wrapper, after dynamic prompts; state and tool identity stay local."""

    def __init__(self):
        self.detectors = (
            PIIMiddleware("email"),
            PIIMiddleware("phone", detector=r"(?<![\w+])\+[1-9](?:[ ()-]*\d){7,14}(?!\d)"),
        )

    def _prepare(self, request):
        aliases = _Pseudonyms()
        config = ensure_config().get("configurable", {})
        aliases.add(config.get("user_id"))
        aliases.add(config.get("thread_id"))
        for value in (config.get("privacy_context") or {}).values():
            aliases.add(value)
        messages = ([request.system_message] if request.system_message else []) + request.messages
        for message in messages:
            if isinstance(message.content, list) and any(
                not isinstance(block, str) and (not isinstance(block, dict) or block.get("type") != "text")
                for block in message.content
            ):
                raise ValueError("Privacy protection supports text messages only.")
            _map_strings(message.content, lambda s: aliases.collect(s, self.detectors))
            _map_strings(message.additional_kwargs, lambda s: aliases.collect(s, self.detectors))
            if isinstance(message, HumanMessage) and message.name:
                aliases.collect(message.name, self.detectors)
                aliases.add(message.name)
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    aliases.structured(call["args"])
                    _map_strings(call["args"], lambda s: aliases.collect(s, self.detectors))

        def copy(message):
            update = {"content": _map_strings(message.content, aliases.redact),
                      "additional_kwargs": _map_strings(message.additional_kwargs, aliases.redact)}
            if isinstance(message, HumanMessage) and message.name:
                update["name"] = aliases.alias(message.name)
            if isinstance(message, AIMessage):
                update["tool_calls"] = [dict(call, args=json.loads(aliases.redact(json.dumps(call["args"]))))
                                        for call in message.tool_calls]
            return message.model_copy(update=update, deep=True)

        system = copy(request.system_message) if request.system_message else None
        copied = [copy(message) for message in request.messages]
        if aliases.originals:
            instruction = "\n\nCopy PRIVATE aliases for personal identifiers exactly, including in structured output."
            system = SystemMessage(content=(system.text if system else "") + instruction)
        return request.override(messages=copied, system_message=system), aliases

    @staticmethod
    def _restore(response, aliases):
        return ModelResponse(
            result=[message.model_copy(update={
                "content": aliases.restore(message.content),
                "additional_kwargs": aliases.restore(message.additional_kwargs),
                **({"tool_calls": [dict(call, args=aliases.restore(call["args"])) for call in message.tool_calls]}
                   if isinstance(message, AIMessage) else {}),
            }, deep=True) for message in response.result],
            structured_response=aliases.restore(response.structured_response),
        )

    def wrap_model_call(self, request, handler):
        protected, aliases = self._prepare(request)
        return self._restore(handler(protected), aliases)

    async def awrap_model_call(self, request, handler):
        protected, aliases = self._prepare(request)
        return self._restore(await handler(protected), aliases)
