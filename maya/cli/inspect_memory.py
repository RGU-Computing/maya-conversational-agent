"""Read-only command-line inspection of MAYA's persistent memory.

Examples:
    python -m maya.cli.inspect_memory --list-users
    python -m maya.cli.inspect_memory --user-id test1
    python -m maya.cli.inspect_memory --user-id test1 --namespace ontology
    python -m maya.cli.inspect_memory --user-id test1 --thread-id session-1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

from dotenv import load_dotenv

from maya.framework.memory import get_postgres_memory


def _jsonable(value: Any, *, max_string_length: int | None) -> Any:
    """Convert store and checkpoint values into readable JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if max_string_length is None or len(value) <= max_string_length:
            return value
        omitted = len(value) - max_string_length
        return f"{value[:max_string_length]}... [{omitted} characters omitted]"

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, max_string_length=max_string_length)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _jsonable(item, max_string_length=max_string_length) for item in value
        ]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(), max_string_length=max_string_length)

    return str(value)


def _all_namespaces(store: Any) -> list[tuple[str, ...]]:
    return [tuple(namespace) for namespace in store.list_namespaces(limit=1000)]


def discover_users(store: Any) -> list[dict[str, Any]]:
    """Return users and their available top-level memory namespaces."""
    namespaces_by_user: dict[str, set[str]] = {}
    for namespace in _all_namespaces(store):
        if len(namespace) < 2:
            continue
        namespaces_by_user.setdefault(namespace[1], set()).add(namespace[0])

    return [
        {"user_id": user_id, "namespaces": sorted(namespaces)}
        for user_id, namespaces in sorted(namespaces_by_user.items())
    ]


def _serialize_items(
    items: Iterable[Any], *, max_string_length: int | None
) -> list[dict[str, Any]]:
    return [
        {
            "key": item.key,
            "value": _jsonable(
                item.value, max_string_length=max_string_length
            ),
            "created_at": _jsonable(
                getattr(item, "created_at", None),
                max_string_length=max_string_length,
            ),
            "updated_at": _jsonable(
                getattr(item, "updated_at", None),
                max_string_length=max_string_length,
            ),
        }
        for item in items
    ]


def _serialize_checkpoint(
    checkpointer: Any,
    *,
    user_id: str,
    thread_id: str,
    max_string_length: int | None,
) -> dict[str, Any] | None:
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    checkpoint = checkpointer.get_tuple(config)
    if checkpoint is None:
        return None

    return _jsonable(
        {
            "config": checkpoint.config,
            "checkpoint": checkpoint.checkpoint,
            "metadata": checkpoint.metadata,
            "parent_config": checkpoint.parent_config,
            "pending_writes": checkpoint.pending_writes,
        },
        max_string_length=max_string_length,
    )


def inspect_user(
    store: Any,
    checkpointer: Any,
    *,
    user_id: str,
    namespaces: set[str] | None = None,
    limit: int = 100,
    thread_id: str | None = None,
    max_string_length: int | None = 500,
) -> dict[str, Any]:
    """Inspect one user's store namespaces and, optionally, one checkpoint."""
    selected = []
    for namespace in _all_namespaces(store):
        if len(namespace) < 2 or namespace[1] != user_id:
            continue
        if namespaces and namespace[0] not in namespaces:
            continue

        items = store.search(namespace, limit=limit)
        serialized_items = _serialize_items(
            items, max_string_length=max_string_length
        )
        selected.append(
            {
                "path": list(namespace),
                "item_count": len(serialized_items),
                "items": serialized_items,
            }
        )

    result: dict[str, Any] = {
        "user_id": user_id,
        "namespaces": selected,
    }
    if thread_id:
        result["checkpoint"] = _serialize_checkpoint(
            checkpointer,
            user_id=user_id,
            thread_id=thread_id,
            max_string_length=max_string_length,
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect MAYA's persistent memory without modifying it."
    )
    subject = parser.add_mutually_exclusive_group(required=True)
    subject.add_argument(
        "--list-users",
        action="store_true",
        help="List discovered user IDs and their memory namespaces.",
    )
    subject.add_argument("--user-id", help="Inspect memory for this user ID.")
    parser.add_argument(
        "--namespace",
        action="append",
        dest="namespaces",
        help="Include only this top-level namespace; may be repeated.",
    )
    parser.add_argument(
        "--thread-id",
        help="Look up a process-local checkpoint (unavailable from a separate runner process).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum items shown per namespace (default: 100).",
    )
    parser.add_argument(
        "--max-string-length",
        type=int,
        default=500,
        help="Truncate long strings after this many characters (default: 500).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show complete strings without truncation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.max_string_length < 1:
        raise SystemExit("--max-string-length must be at least 1")

    load_dotenv()
    try:
        store, checkpointer = get_postgres_memory()
        if args.list_users:
            output: Any = {"users": discover_users(store)}
        else:
            output = inspect_user(
                store,
                checkpointer,
                user_id=args.user_id,
                namespaces=set(args.namespaces) if args.namespaces else None,
                limit=args.limit,
                thread_id=args.thread_id,
                max_string_length=None if args.full else args.max_string_length,
            )
    except Exception as exc:
        raise SystemExit(f"Memory inspection failed: {exc}") from exc

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
