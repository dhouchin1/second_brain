"""
Schema utilities for transitioning from legacy fields to core schema.

Use these SQL snippets to standardize selection and ordering across modules.
"""

def note_ts(alias: str = "notes") -> str:
    return f"COALESCE({alias}.timestamp, {alias}.created_at)"

def note_body(alias: str = "notes") -> str:
    return f"COALESCE({alias}.body, {alias}.content, '')"

def note_updated(alias: str = "notes") -> str:
    return f"COALESCE({alias}.updated_at, {note_ts(alias)})"

