"""Validated partner-editable HealthRAG instructions and retrieval settings."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True,
                              str_strip_whitespace=True, str_min_length=1)


class Prompts(ConfigSection):
    assessment: str
    assessment_repair: str
    clarification: str
    query_rewrite: str
    answer: str
    urgent_answer: str


class Retrieval(ConfigSection):
    k: int = Field(gt=0)
    citation_limit: int = Field(gt=0)
    excerpt_chars: int = Field(gt=3)
    score_threshold: float = Field(ge=0, le=1)


class AnswerCache(ConfigSection):
    semantic_score_threshold: float = Field(ge=0, le=1)
    token_overlap_threshold: float = Field(ge=0, le=1)


class Messages(ConfigSection):
    empty_response: str
    no_evidence: str
    cached_answer_prefix: str


class HealthRagConfig(ConfigSection):
    prompts: Prompts
    retrieval: Retrieval
    cache: AnswerCache
    messages: Messages


def load_health_rag_config(path: str | Path | None = None) -> HealthRagConfig:
    """Build-time snapshot. Relative config_path values are resolved by the agent loader."""
    path = Path(path) if path else Path(__file__).with_name("config.json")
    return HealthRagConfig.model_validate_json(path.read_text(encoding="utf-8"))
