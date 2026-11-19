"""Shared language-model construction from environment configuration."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import convert_to_messages
from langchain_core.runnables import RunnableLambda

from .privacy import PrivacyMiddleware


@lru_cache(maxsize=1)
def get_chat_model() -> AzureChatOpenAI:
    """Return the project's configured deterministic Azure chat model."""

    load_dotenv()
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0,
        timeout=float(os.environ["MAYA_MODEL_TIMEOUT_SECONDS"]),
        max_retries=int(os.environ["MAYA_MODEL_MAX_RETRIES"]),
    )


def model_call(model, schema=None, *, method="json_schema", strict=None):
    """One normal LangChain model turn, protected also for typed/direct calls.

    No tools, retries or additional LLM calls are introduced. The outer MAYA
    graph still owns routing, tools and memory; this boundary owns privacy.
    """
    if method not in {"json_schema", "function_calling"}:
        raise ValueError("Unsupported structured-output method.")
    if method == "function_calling" and strict is not None:
        raise ValueError("ToolStrategy uses the provider's default tool strictness.")
    if strict not in (None, True):
        raise ValueError("Native Pydantic responses use strict schema validation.")
    native_schema = schema is not None and method == "json_schema"
    # Bind the Pydantic class directly, preserving the SDK's strict conversion
    # and parsed response. ProviderStrategy in LangChain 1.1 instead binds an
    # empty tools list and serializes the schema before SDK normalization.
    bound_model = model.bind(response_format=schema) if native_schema else model
    response_format = ToolStrategy(schema, handle_errors=False) if schema is not None and not native_schema else None
    # This is a single model turn, not another conversation. Do not inherit the
    # supervisor's database checkpointer and add writes around every LLM call.
    agent = create_agent(bound_model, tools=[], middleware=[PrivacyMiddleware()],
                         response_format=response_format, checkpointer=False)

    def inputs(value):
        messages = value.to_messages() if hasattr(value, "to_messages") else convert_to_messages(value)
        # LangGraph assigns IDs to messages without one. Work on copies so a
        # model turn cannot mutate the supervisor's history or evidence IDs.
        return {"messages": [message.model_copy(deep=True) for message in messages]}

    def result(state):
        message = state["messages"][-1]
        if native_schema:
            return schema.model_validate(message.additional_kwargs["parsed"])
        return state["structured_response"] if schema is not None else message

    def invoke(value, config):
        return result(agent.invoke(inputs(value), config=config))

    async def ainvoke(value, config):
        return result(await agent.ainvoke(inputs(value), config=config))

    return RunnableLambda(invoke, afunc=ainvoke, name="private_model_call")


__all__ = ["get_chat_model", "model_call"]
