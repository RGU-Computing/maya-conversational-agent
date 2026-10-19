"""Shared language-model construction from environment configuration."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI


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


__all__ = ["get_chat_model"]
