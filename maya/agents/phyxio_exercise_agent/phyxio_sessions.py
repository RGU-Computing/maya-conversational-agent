"""Read-only session history adapter for libphyxio-connect 26.5.29."""

import json

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from maya.framework.integrations.phyxio.bridge import get_phyxio_service
from .sessions import ExerciseSession, PhyxioSessionConfig, SessionMetric


class ExerciseResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    exercise_id: str | int
    status: str
    goal_iter: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    done_iter: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    limit_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    elapsed_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class RecordedSession(BaseModel):
    date: AwareDatetime
    routine_id: str | int
    results: list[ExerciseResult]


def _decode(value) -> dict:
    data = json.loads(value) if isinstance(value, (str, bytes)) else value
    if not isinstance(data, dict):
        raise ValueError("Invalid Phyxio response.")
    return data


def _read(reader, method: str, *args):
    # ICE exceptions are not OSError subclasses. Translate only the transport boundary.
    try:
        return _decode(getattr(reader, method)(*args))
    except Exception:
        raise OSError("Phyxio history request failed.") from None


class PhyxioSessionProvider:
    synthetic = False

    def __init__(self, config: PhyxioSessionConfig, first_page: int):
        fields = set(config.metrics) | {metric.target_field for metric in config.metrics.values() if metric.target_field}
        if not fields <= ExerciseResult.model_fields.keys():
            raise ValueError("Phyxio metric mappings must use fields from the documented exercise result.")
        self.config = config
        self.first_page = first_page

    def _normalize(self, row: RecordedSession, user_id: str, names: dict) -> ExerciseSession:
        results = [result.model_dump() for result in row.results]
        if len({str(result.exercise_id) for result in row.results}) != len(row.results):
            raise ValueError("Repeated exercise IDs cannot be compared unambiguously.")
        metrics = {}
        for result in results:
            if result["status"] not in self.config.metric_statuses:
                continue
            exercise_id = str(result["exercise_id"])
            name = names.get("exercises", {}).get(exercise_id) or exercise_id
            for field, setting in self.config.metrics.items():
                if result.get(field) is not None:
                    target = result.get(setting.target_field)
                    if target in setting.unset_target_values:
                        target = None
                    metrics[f"{exercise_id}:{field}"] = SessionMetric(
                        label=f"{name}: {setting.label}", value=result[field], unit=setting.unit,
                        target=target, target_kind=setting.target_kind if target is not None else None)
        # Derived comparison grouping, not a source session ID or prescription version.
        setup = sorted((str(r.exercise_id), r.goal_iter, r.limit_time) for r in row.results)
        routine_id = str(row.routine_id)
        return ExerciseSession(user_id=user_id, routine_id=routine_id,
            routine_name=names.get("routines", {}).get(routine_id),
            recorded_at=row.date, comparison_key=json.dumps(setup), source="Phyxio session history",
            metrics=metrics, exercise_results=results)

    def _reader(self, user_id: str):
        try:
            return get_phyxio_service().session_history_reader(user_id, self.config.timeout_ms)
        except Exception:
            raise OSError("Exercise history requires the active participant's Phyxio connection.") from None

    def fetch_page(self, user_id: str, page: int) -> tuple[dict, dict]:
        reader = self._reader(user_id)
        data = _read(reader, "getSessionList", page)
        try:
            catalog = _read(reader, "getRoutineList")
        except OSError:
            catalog = {}
        return data, catalog

    def fetch_sessions(self, user_id: str) -> list[ExerciseSession]:
        reader = self._reader(user_id)
        try:
            catalog = _read(reader, "getRoutineList")
            names = {section: {str(key): value["name"] for key, value in catalog.get(section, {}).items()
                              if isinstance(value, dict) and isinstance(value.get("name"), str)}
                     for section in ("routines", "exercises")}
        except (OSError, AttributeError, TypeError):
            names = {}  # Historical results remain usable when current routine names are unavailable.
        sessions, seen = [], set()
        for page in range(self.first_page, self.first_page + self.config.max_pages):
            data = _read(reader, "getSessionList", page)
            if "error" in data:
                if (page > self.first_page and self.config.end_of_history_error
                        and data["error"] == self.config.end_of_history_error):
                    return sessions
                raise OSError("Phyxio could not return session history.")
            if data.get("page") != page or not isinstance(data.get("sessions"), list):
                raise ValueError("Invalid Phyxio history page.")
            if not data["sessions"]:
                return sessions
            for item in data["sessions"]:
                row = RecordedSession.model_validate(item)
                key = (str(row.routine_id), row.date)
                if key in seen:
                    raise ValueError("Phyxio returned ambiguous or repeated session records.")
                seen.add(key)
                sessions.append(self._normalize(row, user_id, names))
        # A bounded partial history cannot establish the latest record without confirmed ordering.
        raise OSError("Phyxio history exceeded the configured page limit.")
