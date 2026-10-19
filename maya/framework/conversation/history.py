"""Bounded dialogue for resolving follow-up requests, without replaying tool calls."""

import json
import os
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages
from pydantic import BaseModel, Field


class ConversationConfig(BaseModel):
    message_limit: int = Field(gt=0)


@lru_cache(maxsize=1)
def conversation_config():
    path = Path(os.getenv("MAYA_CONVERSATION_CONFIG") or Path(__file__).with_name("conversation.json"))
    return ConversationConfig.model_validate(json.loads(path.read_text()))


def recent_dialogue(messages):
    dialogue = [m for m in convert_to_messages(messages)
                if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    return [type(m)(content=m.content) for m in dialogue[-conversation_config().message_limit:]]
