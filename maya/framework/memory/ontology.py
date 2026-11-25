"""Ontology-aligned long-term memory for MAYA agents."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from maya.framework.ontology.registry import OntologyRegistry
from maya.framework.integrations.models import model_call


logger = logging.getLogger(__name__)


class OntologyMemoryNotReady(RuntimeError):
    """Relevant session input has not completed ontology verification yet."""


class OntologyMemoryRecord(BaseModel):
    """A user fact instantiated against one active OntoModel memory binding."""

    model_id: str = Field(description="Selected OntoModel identifier.")
    model_version: str = Field(default="", description="OntoModel version used for this record.")
    binding_path: str = Field(description="Exact active memory binding path.")
    node_id: str = Field(description="Ontology node represented by this record.")
    properties: dict[str, Any] = Field(description="Only fields allowed by the binding capture contract.")
    source_type: str = ""
    source_message_id: str = ""
    evidence_text: str = Field(description="Exact complete assertion copied from the source text.")


class OntologyMemoryExtraction(BaseModel):
    """Structured result of one ontology-memory extraction call."""

    should_write: bool = False
    records: list[OntologyMemoryRecord] = Field(default_factory=list)


class OntologyMemoryCheck(BaseModel):
    """One LLM-evaluated verification check."""

    check_id: str
    passed: bool
    reason: str


class OntologyPropertyCheck(BaseModel):
    """LLM judgment about support for one proposed ontology property."""

    property_name: str
    explicitly_supported: bool
    reason: str


class OntologyMemoryDecision(BaseModel):
    """Independent verification checks for one proposed record."""

    record_index: int
    checks: list[OntologyMemoryCheck] = Field(default_factory=list)
    property_checks: list[OntologyPropertyCheck] = Field(default_factory=list)


class OntologyMemoryVerification(BaseModel):
    """Verification decisions for proposed ontology-memory records."""

    decisions: list[OntologyMemoryDecision] = Field(default_factory=list)


ExtractorFactory = Callable[[Any], Any]


def _create_extractor(model: Any) -> Any:
    return model_call(model,
        OntologyMemoryExtraction,
        method="function_calling",
    )


def _create_verifier(model: Any) -> Any:
    return model_call(model,
        OntologyMemoryVerification,
        method="function_calling",
    )


class OntologyMemoryService:
    """Load ontology memories and schedule validated background enrichment."""

    def __init__(
        self,
        *,
        model: Any,
        store: BaseStore,
        registry: OntologyRegistry,
        extractor_factory: ExtractorFactory = _create_extractor,
        verifier_factory: ExtractorFactory = _create_verifier,
        executor: Executor | None = None,
        register_shutdown: bool = True,
    ):
        self.model = model
        self.store = store
        self.pending_store = InMemoryStore()
        self.registry = registry
        self.extractor_factory = extractor_factory
        self.verifier_factory = verifier_factory
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="maya-ontology-memory")
        self._owns_executor = executor is None
        self._extractors: dict[tuple[str, ...], Any] = {}
        self._verifiers: dict[tuple[str, ...], Any] = {}
        self._scheduled_jobs: set[tuple[str, str]] = set()
        self._recovered_users: set[str] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        if register_shutdown and self._owns_executor:
            atexit.register(self.close)

    @staticmethod
    def memory_namespace(user_id: str) -> tuple[str, ...]:
        return ("ontology", user_id)

    @staticmethod
    def pending_namespace(user_id: str) -> tuple[str, ...]:
        return ("ontology_pending", user_id)

    def load_records(
        self,
        user_id: str,
        model_ids: Sequence[str],
        *,
        limit_per_model: int = 8,
    ) -> list[dict[str, Any]]:
        """Load exact model-filtered records for prompt rendering."""

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        namespace = self.memory_namespace(user_id)
        for model_id in self.registry.resolve_model_ids(model_ids):
            items = self.store.search(
                namespace,
                filter={"type": "ontology_memory_record", "model_id": model_id},
                limit=limit_per_model,
            )
            for item in items:
                if item.key in seen:
                    continue
                record = item.value.get("record")
                if isinstance(record, dict):
                    records.append(record)
                    seen.add(item.key)
        return records

    def render_context(
        self,
        *,
        user_id: str,
        agent_id: str,
        model_ids: Sequence[str],
    ) -> str:
        return self.load_context(user_id=user_id, agent_id=agent_id, model_ids=model_ids)[0]

    def load_context(
        self,
        *,
        user_id: str,
        agent_id: str,
        model_ids: Sequence[str],
    ) -> tuple[str, str]:
        """Return full ontology and record-only context from the same loaded records."""

        selected = self.registry.resolve_model_ids(model_ids, agent_id=agent_id)
        self._recover_pending(user_id)
        pending = self._relevant_pending(user_id, selected)
        if pending:
            with self._condition:
                self._condition.wait_for(
                    lambda: not any((user_id, item.key) in self._scheduled_jobs for item in pending),
                    timeout=self.registry.runtime["memory_policy"]["recovery_wait_seconds"],
                )
            if self._relevant_pending(user_id, selected):
                raise OntologyMemoryNotReady("Relevant ontology updates are still pending.")
        memories = self.load_records(user_id, selected)
        return (
            self.registry.render_context(agent_id, selected, memories),
            self.registry.render_memory_context(selected, memories),
        )

    def _relevant_pending(self, user_id: str, model_ids: Sequence[str]) -> list[Any]:
        selected = set(model_ids)
        return [item for item in self.pending_store.search(
            self.pending_namespace(user_id), filter={"type": "ontology_memory_job", "user_id": user_id},
            limit=self.registry.runtime["memory_policy"]["pending_job_limit"],
        ) if set(item.value.get("selected_model_ids") or []) & selected]

    def schedule(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_message_id: str,
        source_text: str,
        model_ids: Sequence[str],
        privacy_context: dict | None = None,
    ) -> bool:
        """Queue a session-only pending job and return without waiting for its LLM work."""

        selected = self.registry.resolve_model_ids(model_ids)
        selected = [model_id for model_id in selected if self.registry.has_active_bindings([model_id])]
        if self._closed or not source_text.strip() or not selected:
            return False

        self._recover_pending(user_id)
        job_key = _job_key(thread_id, source_message_id, source_text)
        payload = {
            "type": "ontology_memory_job",
            "user_id": user_id,
            "thread_id": thread_id,
            "source_message_id": source_message_id,
            "source_text": source_text.strip(),
            "selected_model_ids": selected,
            "privacy_context": dict(privacy_context or {}),
        }
        self.pending_store.put(self.pending_namespace(user_id), job_key, payload, index=False)
        self._submit(job_key, payload)
        return True

    def process_now(self, job_key: str, payload: dict[str, Any]) -> int:
        """Process one job synchronously; useful for tests and recovery."""

        return self._process_job(job_key, payload)

    def _recover_pending(self, user_id: str) -> None:
        with self._lock:
            if user_id in self._recovered_users:
                return
            self._recovered_users.add(user_id)
        for item in self.pending_store.search(
            self.pending_namespace(user_id), limit=self.registry.runtime["memory_policy"]["pending_job_limit"],
        ):
            if isinstance(item.value, dict) and item.value.get("type") == "ontology_memory_job":
                self._submit(item.key, item.value)

    def _submit(self, job_key: str, payload: dict[str, Any]) -> None:
        user_id = str(payload["user_id"])
        scheduled_key = (user_id, job_key)
        with self._lock:
            if self._closed or scheduled_key in self._scheduled_jobs:
                return
            self._scheduled_jobs.add(scheduled_key)
        future = self.executor.submit(self._process_job, job_key, payload)
        future.add_done_callback(lambda completed: self._job_finished(scheduled_key, completed))

    def _job_finished(self, scheduled_key: tuple[str, str], future: Future[Any]) -> None:
        error = None if future.cancelled() else future.exception()
        with self._condition:
            self._scheduled_jobs.discard(scheduled_key)
            if error is not None or future.cancelled():
                self._recovered_users.discard(scheduled_key[0])
            self._condition.notify_all()
        if error is not None:
            logger.warning("Ontology memory background update failed: %s", type(error).__name__)

    def _extractor(self, model_ids: Sequence[str]) -> Any:
        key = tuple(model_ids)
        with self._lock:
            extractor = self._extractors.get(key)
            if extractor is None:
                extractor = self.extractor_factory(self.model)
                self._extractors[key] = extractor
            return extractor

    def _verifier(self, model_ids: Sequence[str]) -> Any:
        key = tuple(model_ids)
        with self._lock:
            verifier = self._verifiers.get(key)
            if verifier is None:
                verifier = self.verifier_factory(self.model)
                self._verifiers[key] = verifier
            return verifier

    def _verified_records(
        self,
        *,
        records: Sequence[OntologyMemoryRecord],
        source_text: str,
        source_message_id: str,
        model_ids: Sequence[str],
        config: dict[str, Any],
    ) -> list[OntologyMemoryRecord]:
        if not records:
            return []
        required_checks = set(self.registry.memory_verification_check_ids)
        decisions: dict[int, OntologyMemoryDecision] = {}
        # Retry omitted/incomplete decisions once. They are not semantic rejections.
        for _ in range(2):
            pending = {index: record for index, record in enumerate(records) if index not in decisions}
            verification = self._verifier(model_ids).invoke(
                [
                    SystemMessage(content=self.registry.memory_verification_instructions(model_ids)),
                    HumanMessage(content=json.dumps({
                        "source_message": {"id": source_message_id, "text": source_text},
                        "proposed_records": [
                            {"record_index": index, "record": record.model_dump(mode="json")}
                            for index, record in pending.items()
                        ],
                    }, ensure_ascii=False, separators=(",", ":")), id=source_message_id),
                ],
                config=config,
            )
            verification = OntologyMemoryVerification.model_validate(verification)
            indices = [decision.record_index for decision in verification.decisions]
            for decision in verification.decisions:
                index = decision.record_index
                if index not in pending or indices.count(index) != 1:
                    continue
                checks = [check.check_id for check in decision.checks]
                properties = [check.property_name for check in decision.property_checks]
                if (len(checks) == len(required_checks) and set(checks) == required_checks
                        and len(properties) == len(pending[index].properties)
                        and set(properties) == set(pending[index].properties)):
                    decisions[index] = decision
            if len(decisions) == len(records):
                break
        if len(decisions) != len(records):
            raise ValueError("Ontology memory verification omitted required record or property checks.")

        approved: dict[int, OntologyMemoryRecord] = {}
        for decision in decisions.values():
            results = {check.check_id: check.passed for check in decision.checks}
            proposed = records[decision.record_index]
            proposed_property_names = set(proposed.properties)
            supported_property_names = {
                check.property_name
                for check in decision.property_checks
                if check.explicitly_supported and check.property_name in proposed_property_names
            }
            if all(results.values()):
                approved[decision.record_index] = proposed.model_copy(
                    update={
                        "properties": {
                            name: value
                            for name, value in proposed.properties.items()
                            if name in supported_property_names
                        }
                    }
                )
        return [approved[index] for index in sorted(approved)]

    def _process_job(self, job_key: str, payload: dict[str, Any]) -> int:
        user_id = str(payload["user_id"])
        source_text = str(payload["source_text"])
        source_message_id = str(payload["source_message_id"])
        model_ids = self.registry.resolve_model_ids(payload.get("selected_model_ids") or [])
        namespace = self.memory_namespace(user_id)

        existing_items = self.store.search(
            namespace,
            filter={"type": "ontology_memory_record"},
            limit=100,
        )
        existing: list[dict[str, Any]] = []
        for item in existing_items:
            raw_record = item.value.get("record")
            if not isinstance(raw_record, dict) or raw_record.get("model_id") not in model_ids:
                continue
            try:
                record = OntologyMemoryRecord.model_validate({**raw_record, "evidence_text": ""})
                existing.append(
                    {
                        "record_id": item.key,
                        "record": record.model_dump(mode="json", exclude={"evidence_text"}),
                    }
                )
            except ValueError:
                continue

        extractor = self._extractor(model_ids)
        identity = payload.get("privacy_context")
        if identity is None:
            # Jobs created before privacy context was persisted may be recovered
            # after an upgrade. Resolve only this participant's existing profile.
            profiles = self.store.search(("profile", user_id), limit=1)
            identity = {key: profiles[0].value.get(key) for key in ("name", "location")} if profiles else {}
        config = {
            "configurable": {
                "thread_id": str(payload["thread_id"]),
                "user_id": user_id,
                "privacy_context": identity,
            }
        }
        extraction = extractor.invoke(
            [
                SystemMessage(content=self.registry.memory_instructions(model_ids)),
                HumanMessage(
                    content=json.dumps(
                        {
                            "existing_records": existing,
                            "source_message": {
                                "id": source_message_id,
                                "text": source_text,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    id=source_message_id,
                ),
            ],
            config=config,
        )
        if not isinstance(extraction, OntologyMemoryExtraction):
            extraction = OntologyMemoryExtraction.model_validate(extraction)

        writes = 0
        proposed = extraction.records if extraction.should_write else []
        records = self._verified_records(
            records=proposed,
            source_text=source_text,
            source_message_id=source_message_id,
            model_ids=model_ids,
            config=config,
        )
        for content in records:
            model_version = str(self.registry.models.get(content.model_id, {}).get("version") or "")
            record = content.model_copy(
                update={
                    "model_version": model_version,
                    "source_type": self.registry.user_message_source,
                    "source_message_id": source_message_id,
                }
            )
            raw = record.model_dump(mode="json")
            try:
                self.registry.validate_record(
                    raw,
                    source_text=source_text,
                    selected_model_ids=model_ids,
                )
            except ValueError:
                continue

            key = _record_key(self.registry, raw)
            del raw["evidence_text"]
            search_text = _search_text(raw)
            self.store.put(
                namespace,
                key,
                {
                    "type": "ontology_memory_record",
                    "model_id": raw["model_id"],
                    "binding_path": raw["binding_path"],
                    "record": raw,
                    "search_text": search_text,
                },
                index=["search_text"],
            )
            writes += 1

        self.pending_store.delete(self.pending_namespace(user_id), job_key)
        return writes

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._owns_executor:
            self.executor.shutdown(wait=True, cancel_futures=False)
        self.pending_store = InMemoryStore()


def _job_key(thread_id: str, source_message_id: str, source_text: str) -> str:
    raw = "|".join([thread_id, source_message_id, source_text.strip()])
    return f"job:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _record_key(registry: OntologyRegistry, record: dict[str, Any]) -> str:
    _, binding = registry.binding(str(record["binding_path"]))
    identity_fields = (binding.get("capture") or {}).get("identity_fields") or []
    properties = record.get("properties") or {}
    identity = [_canonical_identity(properties.get(field)) for field in identity_fields]
    if not any(identity):
        identity = [str(record.get("source_message_id") or ""), str(record.get("evidence_text") or "")]
    raw = json.dumps(
        [record["model_id"], record["binding_path"], identity],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"record:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _canonical_identity(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _search_text(record: dict[str, Any]) -> str:
    properties = " ".join(str(value) for value in (record.get("properties") or {}).values())
    return " ".join(
        part
        for part in [
            str(record.get("model_id") or ""),
            str(record.get("binding_path") or ""),
            properties,
        ]
        if part
    )


__all__ = [
    "OntologyMemoryDecision",
    "OntologyMemoryExtraction",
    "OntologyMemoryCheck",
    "OntologyPropertyCheck",
    "OntologyMemoryRecord",
    "OntologyMemoryService",
    "OntologyMemoryVerification",
]
