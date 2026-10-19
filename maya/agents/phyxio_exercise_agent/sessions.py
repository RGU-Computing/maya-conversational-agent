"""Read-only session comparison; the provider owns transport and identity mapping."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Literal, Protocol

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import AliasChoices, AwareDatetime, BaseModel, ConfigDict, Field


class SessionMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    value: float = Field(allow_inf_nan=False)
    unit: str
    target: float | None = Field(default=None, allow_inf_nan=False)
    target_kind: str | None = None


class ExerciseSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str
    session_id: str | None = None
    routine_id: str
    routine_name: str | None = None
    comparison_key: str = Field(min_length=1, description="Comparable recorded exercises and targets, not proof of an unchanged prescription.")
    recorded_at: AwareDatetime = Field(validation_alias=AliasChoices("recorded_at", "ended_at"))
    status: str | None = None
    source: str
    metrics: dict[str, SessionMetric]
    exercise_results: list[dict] = Field(default_factory=list)

    def is_finished(self, accepted_statuses: list[str]) -> bool:
        statuses = [self.status] if self.status else [r["status"] for r in self.exercise_results]
        return bool(statuses) and all(status in accepted_statuses for status in statuses)


class PhyxioMetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    unit: str
    target_field: str | None = None
    target_kind: str
    unset_target_values: list[float] = Field(default_factory=list)


class PhyxioSessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_pages: int = Field(gt=0)
    timeout_ms: int = Field(gt=0)
    end_of_history_error: str | None = None
    metric_statuses: list[str]
    metrics: dict[str, PhyxioMetricConfig]


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["disabled", "dummy", "phyxio"]
    first_page: int = Field(ge=0)
    fixture_path: str | None = None
    phyxio: PhyxioSessionConfig | None = None
    comparison_limit: int = Field(ge=0)
    accepted_statuses: list[str]
    fallback_text: str
    subject_clarification: str


class SessionProvider(Protocol):
    synthetic: bool

    def fetch_sessions(self, user_id: str) -> list[ExerciseSession]: ...


class DummySessionProvider:
    synthetic = True

    def __init__(self, path: Path):
        self.sessions = [ExerciseSession.model_validate(row) for row in json.loads(path.read_text())]

    def fetch_sessions(self, user_id: str) -> list[ExerciseSession]:
        return [session for session in self.sessions if session.user_id == user_id]


class ExerciseSessionService:
    def __init__(self, config: SessionConfig, provider: SessionProvider | None):
        self.config, self.provider = config, provider

    def latest(self, user_id: str) -> dict:
        result = {"status": "unavailable", "synthetic": bool(self.provider and self.provider.synthetic),
                  "latest": None, "previous": [], "comparison": {},
                  "comparison_limit": self.config.comparison_limit, "fallback_text": self.config.fallback_text}
        if not user_id or self.provider is None:
            return result
        try:
            sessions = self.provider.fetch_sessions(user_id)
            ids = [s.session_id for s in sessions if s.session_id is not None]
            if any(s.user_id != user_id for s in sessions) or len(set(ids)) != len(ids):
                return {**result, "status": "invalid_response"}
            now = datetime.now(timezone.utc)
            sessions = sorted((s for s in sessions if s.is_finished(self.config.accepted_statuses)
                               and s.recorded_at <= now), key=lambda s: s.recorded_at, reverse=True)
        except (OSError, ValueError, TimeoutError):
            return result
        if not sessions:
            return {**result, "status": "no_data"}
        latest = sessions[0]
        if len(sessions) > 1 and sessions[1].recorded_at == latest.recorded_at:
            return {**result, "status": "ambiguous_latest"}
        previous = [s for s in sessions[1:] if s.routine_id == latest.routine_id
                    and s.comparison_key == latest.comparison_key][:self.config.comparison_limit]
        comparison = {}
        for key, metric in latest.metrics.items():
            values = [s.metrics[key].value for s in previous if key in s.metrics
                      and s.metrics[key].unit == metric.unit and s.metrics[key].target == metric.target
                      and s.metrics[key].target_kind == metric.target_kind]
            if values:
                average = mean(values)
                comparison[key] = {"label": metric.label, "unit": metric.unit, "latest": metric.value,
                                   "previous_count": len(values), "previous_mean": average,
                                   "difference": metric.value - average}
        return {**result, "status": "ok", "latest": latest.model_dump(mode="json", exclude={"user_id"}, exclude_none=True),
                "previous": [s.model_dump(mode="json", exclude={"user_id"}, exclude_none=True) for s in previous],
                "comparison": comparison}


def load_session_service() -> ExerciseSessionService:
    default = Path(__file__).with_name("session_config.json")
    path = Path(os.getenv("MAYA_EXERCISE_SESSION_CONFIG") or default)
    config = SessionConfig.model_validate({**json.loads(default.read_text()), **json.loads(path.read_text())})
    provider = None
    if config.provider == "dummy":
        if not config.fixture_path:
            raise ValueError("The dummy session provider requires fixture_path.")
        provider = DummySessionProvider(path.parent / config.fixture_path)
    elif config.provider == "phyxio":
        from .phyxio_sessions import PhyxioSessionProvider

        if config.phyxio is None:
            raise ValueError("The Phyxio session provider requires phyxio settings.")
        provider = PhyxioSessionProvider(config.phyxio, config.first_page)
    return ExerciseSessionService(config, provider)


@tool
def get_latest_exercise_session(config: RunnableConfig) -> dict:
    """Review the current participant's latest recorded exercise session and comparable previous sessions.

    Read-only: for feedback on performance or progress, not for starting an exercise.
    Identity and the comparison limit come from application configuration, never model arguments.
    """
    return load_session_service().latest(str(config.get("configurable", {}).get("user_id") or ""))
