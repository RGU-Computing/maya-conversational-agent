from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import TestCase

from maya.cli.inspect_memory import (
    _jsonable,
    discover_users,
    inspect_user,
)


class FakeStore:
    def __init__(self):
        timestamp = datetime(2026, 7, 17, 10, 30, tzinfo=timezone.utc)
        self.namespaces = [
            ("profile", "alice"),
            ("ontology", "alice"),
            ("ontology_pending", "bob"),
        ]
        self.items = {
            ("profile", "alice"): [
                SimpleNamespace(
                    key="identity",
                    value={"name": "Alice"},
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            ],
            ("ontology", "alice"): [
                SimpleNamespace(
                    key="medication-1",
                    value={"medication": "Bisoprolol", "dose": "2.5 mg"},
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            ],
            ("ontology_pending", "bob"): [],
        }

    def list_namespaces(self, *, limit):
        return self.namespaces[:limit]

    def search(self, namespace, *, limit):
        return self.items[namespace][:limit]


class FakeCheckpointer:
    def get_tuple(self, config):
        return SimpleNamespace(
            config=config,
            checkpoint={"channel_values": {"status": "ready"}},
            metadata={"source": "loop"},
            parent_config=None,
            pending_writes=[],
        )


class MemoryInspectorTests(TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.checkpointer = FakeCheckpointer()

    def test_jsonable_truncates_nested_strings(self):
        result = _jsonable(
            {"nested": ["abcdefgh"]}, max_string_length=4
        )

        self.assertEqual(
            result, {"nested": ["abcd... [4 characters omitted]"]}
        )

    def test_discover_users_groups_namespaces(self):
        self.assertEqual(
            discover_users(self.store),
            [
                {"user_id": "alice", "namespaces": ["ontology", "profile"]},
                {"user_id": "bob", "namespaces": ["ontology_pending"]},
            ],
        )

    def test_inspect_user_filters_namespaces_and_includes_checkpoint(self):
        result = inspect_user(
            self.store,
            self.checkpointer,
            user_id="alice",
            namespaces={"ontology"},
            thread_id="demo-thread",
        )

        self.assertEqual(result["namespaces"][0]["path"], ["ontology", "alice"])
        self.assertEqual(result["namespaces"][0]["item_count"], 1)
        self.assertEqual(
            result["namespaces"][0]["items"][0]["value"]["dose"], "2.5 mg"
        )
        self.assertEqual(
            result["checkpoint"]["config"]["configurable"]["thread_id"],
            "demo-thread",
        )
