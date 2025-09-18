"""Helpers for broadcasting realtime note events via WebSockets."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

from services.websocket_manager import get_connection_manager


async def notify_note_update(user_id: int, action: str, note: Dict[str, Any]) -> None:
    """Send a realtime payload to all of a user's active WebSocket connections."""
    manager = get_connection_manager()
    await manager.send_to_user(
        user_id,
        {
            "type": "note_update",
            "action": action,
            "note": note,
            "timestamp": datetime.now().isoformat(),
        },
    )


def schedule_note_update(user_id: int, action: str, note: Dict[str, Any]) -> None:
    """Schedule a realtime note update outside of an async context."""

    async def _send() -> None:
        await notify_note_update(user_id, action, note)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_send())
    else:
        loop.create_task(_send())
