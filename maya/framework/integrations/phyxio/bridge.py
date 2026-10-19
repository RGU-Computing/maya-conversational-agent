from __future__ import annotations

from maya.framework.conversation.personalization import SessionConfig


_service = None


def get_phyxio_service(session: SessionConfig | None = None):
    """One connection per process, configured by the runner before any tool calls."""
    global _service
    if _service is None:
        if session is None:
            raise RuntimeError("Initialize Phyxio with the participant session before using its tools.")
        from .service import PhyxioService

        _service = PhyxioService(session)
    elif session is not None and session != _service.session:
        raise ValueError("Phyxio is already configured for another session; restart MAYA to change it.")
    return _service
