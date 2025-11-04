#!/usr/bin/env python3
"""
Second Brain MCP Server

An MCP (Model Context Protocol) server that exposes Second Brain functionality
to Claude Code and other MCP clients.

Provides tools for:
- Searching notes
- Creating notes
- Retrieving note details
- Analyzing vault statistics
- Managing tags

Installation:
    pip install mcp

Usage:
    python mcp_server.py

Configuration (in Claude Code settings):
    {
        "mcpServers": {
            "second-brain": {
                "command": "python",
                "args": ["/Users/dhouchin/mvp-setup/second_brain/mcp_server.py"],
                "env": {}
            }
        }
    }
"""

import asyncio
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: mcp package not installed")
    print("Install with: pip install mcp")
    exit(1)

# Database path
DB_PATH = Path(__file__).parent / "notes.db"

# Initialize MCP server
app = Server("second-brain")


def get_db():
    """Get database connection"""
    return sqlite3.connect(str(DB_PATH))


# ============================================================================
# MCP Tool: Search Notes
# ============================================================================

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools"""
    return [
        Tool(
            name="search_notes",
            description="Search through Second Brain notes using full-text search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    },
                    "user_id": {
                        "type": "number",
                        "description": "User ID to search for (default: 1)",
                        "default": 1
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="create_note",
            description="Create a new note in Second Brain",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Note content"
                    },
                    "title": {
                        "type": "string",
                        "description": "Note title (optional, will be auto-generated if not provided)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the note (optional)"
                    },
                    "user_id": {
                        "type": "number",
                        "description": "User ID (default: 1)",
                        "default": 1
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="get_note",
            description="Get details of a specific note by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "number",
                        "description": "Note ID"
                    },
                    "user_id": {
                        "type": "number",
                        "description": "User ID (default: 1)",
                        "default": 1
                    }
                },
                "required": ["note_id"]
            }
        ),
        Tool(
            name="get_vault_stats",
            description="Get statistics and analytics about the Second Brain vault",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "number",
                        "description": "User ID (default: 1)",
                        "default": 1
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_tags",
            description="Get all tags and their frequencies",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "number",
                        "description": "User ID (default: 1)",
                        "default": 1
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of tags to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_recent_notes",
            description="Get recently created notes",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "number",
                        "description": "User ID (default: 1)",
                        "default": 1
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of notes to return (default: 10)",
                        "default": 10
                    },
                    "days": {
                        "type": "number",
                        "description": "Number of days to look back (default: 7)",
                        "default": 7
                    }
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls"""

    if name == "search_notes":
        return await search_notes_tool(arguments)
    elif name == "create_note":
        return await create_note_tool(arguments)
    elif name == "get_note":
        return await get_note_tool(arguments)
    elif name == "get_vault_stats":
        return await get_vault_stats_tool(arguments)
    elif name == "get_tags":
        return await get_tags_tool(arguments)
    elif name == "get_recent_notes":
        return await get_recent_notes_tool(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ============================================================================
# Tool Implementations
# ============================================================================

async def search_notes_tool(args: Dict[str, Any]) -> List[TextContent]:
    """Search notes using FTS5"""
    query = args["query"]
    limit = args.get("limit", 10)
    user_id = args.get("user_id", 1)

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Search using FTS5
        cursor.execute("""
            SELECT n.id, n.title, n.content, n.tags, n.created_at, n.type
            FROM notes n
            JOIN notes_fts fts ON n.id = fts.rowid
            WHERE fts MATCH ? AND n.user_id = ?
            ORDER BY fts.rank
            LIMIT ?
        """, (query, user_id, limit))

        results = cursor.fetchall()

        if not results:
            return [TextContent(
                type="text",
                text=f"No results found for '{query}'"
            )]

        # Format results
        output = f"Found {len(results)} results for '{query}':\n\n"
        for i, (note_id, title, content, tags, created_at, note_type) in enumerate(results, 1):
            output += f"{i}. **{title or 'Untitled'}** (ID: {note_id})\n"
            output += f"   Type: {note_type or 'text'}\n"
            if tags:
                output += f"   Tags: {tags}\n"
            output += f"   Created: {created_at}\n"
            # Truncate content
            preview = content[:200] if content else ""
            if len(content or "") > 200:
                preview += "..."
            output += f"   Preview: {preview}\n\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error searching notes: {str(e)}")]
    finally:
        conn.close()


async def create_note_tool(args: Dict[str, Any]) -> List[TextContent]:
    """Create a new note"""
    content = args["content"]
    title = args.get("title")
    tags = args.get("tags", [])
    user_id = args.get("user_id", 1)

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Generate title if not provided
        if not title:
            # Simple title generation (first 50 chars)
            title = content[:50].strip()
            if len(content) > 50:
                title += "..."

        # Format tags
        tags_str = ",".join(tags) if tags else None

        # Insert note
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO notes (
                user_id, title, content, body, tags,
                type, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, title, content, content, tags_str,
            'text', 'completed', now, now
        ))

        note_id = cursor.lastrowid
        conn.commit()

        output = f"✅ Note created successfully!\n\n"
        output += f"**ID**: {note_id}\n"
        output += f"**Title**: {title}\n"
        if tags:
            output += f"**Tags**: {', '.join(tags)}\n"
        output += f"**Created**: {now}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error creating note: {str(e)}")]
    finally:
        conn.close()


async def get_note_tool(args: Dict[str, Any]) -> List[TextContent]:
    """Get note details"""
    note_id = args["note_id"]
    user_id = args.get("user_id", 1)

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT id, title, content, tags, type, status, created_at, updated_at
            FROM notes
            WHERE id = ? AND user_id = ?
        """, (note_id, user_id))

        result = cursor.fetchone()

        if not result:
            return [TextContent(type="text", text=f"Note #{note_id} not found")]

        note_id, title, content, tags, note_type, status, created_at, updated_at = result

        output = f"# {title or 'Untitled'}\n\n"
        output += f"**ID**: {note_id}\n"
        output += f"**Type**: {note_type or 'text'}\n"
        output += f"**Status**: {status or 'unknown'}\n"
        if tags:
            output += f"**Tags**: {tags}\n"
        output += f"**Created**: {created_at}\n"
        output += f"**Updated**: {updated_at}\n\n"
        output += f"## Content\n\n{content}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting note: {str(e)}")]
    finally:
        conn.close()


async def get_vault_stats_tool(args: Dict[str, Any]) -> List[TextContent]:
    """Get vault statistics"""
    user_id = args.get("user_id", 1)

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Total notes
        total = cursor.execute(
            "SELECT COUNT(*) FROM notes WHERE user_id = ?",
            (user_id,)
        ).fetchone()[0]

        # Notes by type
        cursor.execute("""
            SELECT type, COUNT(*) as count
            FROM notes
            WHERE user_id = ?
            GROUP BY type
            ORDER BY count DESC
        """, (user_id,))
        types = cursor.fetchall()

        # Recent activity (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        recent = cursor.execute("""
            SELECT COUNT(*) FROM notes
            WHERE user_id = ? AND created_at > ?
        """, (user_id, thirty_days_ago)).fetchone()[0]

        # Tags
        cursor.execute(
            "SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL",
            (user_id,)
        )
        all_tags = []
        for (tags_str,) in cursor.fetchall():
            if tags_str:
                all_tags.extend([t.strip() for t in tags_str.replace('#', '').split(',') if t.strip()])

        tag_counts = Counter(all_tags)
        top_tags = tag_counts.most_common(5)

        # Format output
        output = "# 📊 Second Brain Statistics\n\n"
        output += f"**Total Notes**: {total}\n"
        output += f"**Notes (Last 30 Days)**: {recent}\n\n"

        output += "## Notes by Type\n"
        for note_type, count in types:
            output += f"- {note_type or 'unknown'}: {count}\n"

        output += "\n## Top Tags\n"
        for tag, count in top_tags:
            output += f"- {tag}: {count} notes\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting stats: {str(e)}")]
    finally:
        conn.close()


async def get_tags_tool(args: Dict[str, Any]) -> List[TextContent]:
    """Get all tags"""
    user_id = args.get("user_id", 1)
    limit = args.get("limit", 20)

    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT tags FROM notes WHERE user_id = ? AND tags IS NOT NULL",
            (user_id,)
        )

        all_tags = []
        for (tags_str,) in cursor.fetchall():
            if tags_str:
                all_tags.extend([t.strip() for t in tags_str.replace('#', '').split(',') if t.strip()])

        tag_counts = Counter(all_tags)
        top_tags = tag_counts.most_common(limit)

        output = f"# 🏷️ Tags (Top {len(top_tags)})\n\n"
        for tag, count in top_tags:
            output += f"- **{tag}**: {count} notes\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting tags: {str(e)}")]
    finally:
        conn.close()


async def get_recent_notes_tool(args: Dict[str, Any]) -> List[TextContent]:
    """Get recent notes"""
    user_id = args.get("user_id", 1)
    limit = args.get("limit", 10)
    days = args.get("days", 7)

    conn = get_db()
    cursor = conn.cursor()

    try:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT id, title, content, tags, created_at, type
            FROM notes
            WHERE user_id = ? AND created_at > ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, cutoff, limit))

        results = cursor.fetchall()

        if not results:
            return [TextContent(
                type="text",
                text=f"No notes found in the last {days} days"
            )]

        output = f"# 📝 Recent Notes (Last {days} Days)\n\n"
        for note_id, title, content, tags, created_at, note_type in results:
            output += f"## {title or 'Untitled'} (ID: {note_id})\n"
            output += f"**Type**: {note_type or 'text'}\n"
            if tags:
                output += f"**Tags**: {tags}\n"
            output += f"**Created**: {created_at}\n"
            preview = (content or "")[:150]
            if len(content or "") > 150:
                preview += "..."
            output += f"{preview}\n\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting recent notes: {str(e)}")]
    finally:
        conn.close()


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
