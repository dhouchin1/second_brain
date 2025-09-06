# ──────────────────────────────────────────────────────────────────────────────
# File: services/enhanced_discord_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced Discord Integration Service

Advanced Discord bot features including thread capture, reaction-based workflows,
team collaboration, and intelligent content processing.
"""

import discord
from discord.ext import commands
import json
import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import sqlite3

from config import settings
from services.embeddings import Embeddings
from services.advanced_capture_service import get_advanced_capture_service, CaptureOptions
from llm_utils import ollama_summarize, ollama_generate_title

logger = logging.getLogger(__name__)

@dataclass
class ThreadCapture:
    """Data structure for Discord thread capture."""
    thread_id: int
    thread_name: str
    messages: List[Dict[str, Any]]
    participants: List[str]
    start_time: datetime
    end_time: datetime
    message_count: int

@dataclass
class DiscordContext:
    """Discord-specific context information."""
    guild_id: Optional[int]
    guild_name: Optional[str]
    channel_id: int
    channel_name: str
    user_id: int
    username: str
    message_id: Optional[int] = None

class EnhancedDiscordService:
    """Enhanced Discord integration service."""
    
    def __init__(self, get_conn_func):
        """Initialize with database connection function."""
        self.get_conn = get_conn_func
        self.embedder = Embeddings()
        self.advanced_capture = None
        
        # Bot configuration
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.intents.guilds = True
        self.intents.members = True
        
        self.bot = commands.Bot(command_prefix='!sb ', intents=self.intents)
        
        # Register event handlers
        self._register_events()
        self._register_commands()
    
    def _get_advanced_capture(self):
        """Lazy load advanced capture service."""
        if not self.advanced_capture:
            self.advanced_capture = get_advanced_capture_service(self.get_conn)
        return self.advanced_capture
    
    def _register_events(self):
        """Register Discord event handlers."""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Enhanced Discord bot logged in as {self.bot.user}")
        
        @self.bot.event
        async def on_reaction_add(reaction, user):
            """Handle reaction-based workflows."""
            if user.bot:
                return
            
            # 🧠 - Save message to Second Brain
            if str(reaction.emoji) == "🧠":
                await self._handle_brain_reaction(reaction, user)
            
            # 📝 - Start thread summary
            elif str(reaction.emoji) == "📝":
                await self._handle_summary_reaction(reaction, user)
            
            # ⭐ - Mark as important
            elif str(reaction.emoji) == "⭐":
                await self._handle_star_reaction(reaction, user)
    
    def _register_commands(self):
        """Register Discord slash commands."""
        
        @self.bot.hybrid_command(name="capture", description="Capture a note to Second Brain")
        async def capture_note(ctx, *, content: str):
            """Capture a quick note."""
            try:
                discord_context = DiscordContext(
                    guild_id=ctx.guild.id if ctx.guild else None,
                    guild_name=ctx.guild.name if ctx.guild else None,
                    channel_id=ctx.channel.id,
                    channel_name=ctx.channel.name,
                    user_id=ctx.author.id,
                    username=ctx.author.name
                )
                
                result = await self.capture_text_note(
                    content=content,
                    discord_context=discord_context,
                    note_type="quick_note"
                )
                
                if result["success"]:
                    embed = discord.Embed(
                        title="✅ Note Captured",
                        description=f"**Title:** {result['title']}\n**Tags:** {', '.join(result.get('tags', []))[:100]}",
                        color=0x00ff00
                    )
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"❌ Failed to capture note: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Capture command failed: {e}")
                await ctx.send(f"❌ Error: {str(e)}")
        
        @self.bot.hybrid_command(name="thread_summary", description="Summarize a thread conversation")
        async def thread_summary(ctx, messages: int = 50):
            """Summarize thread conversation."""
            try:
                if not isinstance(ctx.channel, discord.Thread):
                    await ctx.send("❌ This command can only be used in threads.")
                    return
                
                await ctx.defer()
                
                thread_capture = await self._capture_thread_messages(ctx.channel, messages)
                result = await self.process_thread_summary(thread_capture)
                
                if result["success"]:
                    embed = discord.Embed(
                        title=f"📝 Thread Summary: {thread_capture.thread_name}",
                        description=result["summary"][:2000],
                        color=0x0099ff
                    )
                    
                    if result.get("key_points"):
                        embed.add_field(
                            name="Key Points",
                            value="\n".join([f"• {point}" for point in result["key_points"][:5]]),
                            inline=False
                        )
                    
                    if result.get("action_items"):
                        embed.add_field(
                            name="Action Items", 
                            value="\n".join([f"• {item}" for item in result["action_items"][:5]]),
                            inline=False
                        )
                    
                    embed.add_field(
                        name="Note ID",
                        value=str(result["note_id"]),
                        inline=True
                    )
                    
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"❌ Failed to summarize thread: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Thread summary failed: {e}")
                await ctx.send(f"❌ Error: {str(e)}")
        
        @self.bot.hybrid_command(name="search", description="Search Second Brain notes")
        async def search_notes(ctx, *, query: str):
            """Search notes in Second Brain."""
            try:
                results = await self.search_notes(query, limit=5)
                
                if results:
                    embed = discord.Embed(
                        title=f"🔍 Search Results: '{query}'",
                        color=0x9932cc
                    )
                    
                    for i, note in enumerate(results[:5], 1):
                        title = note.get('title', 'Untitled')[:100]
                        content = note.get('body', '')[:150] + "..." if len(note.get('body', '')) > 150 else note.get('body', '')
                        tags = note.get('tags', '')[:50] + "..." if len(note.get('tags', '')) > 50 else note.get('tags', '')
                        
                        embed.add_field(
                            name=f"{i}. {title}",
                            value=f"{content}\n*Tags: {tags}*",
                            inline=False
                        )
                    
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"🔍 No results found for '{query}'")
                    
            except Exception as e:
                logger.error(f"Search command failed: {e}")
                await ctx.send(f"❌ Search error: {str(e)}")
        
        @self.bot.hybrid_command(name="meeting_notes", description="Start meeting notes template")
        async def meeting_notes(ctx, *, meeting_topic: str):
            """Create meeting notes template."""
            try:
                discord_context = DiscordContext(
                    guild_id=ctx.guild.id if ctx.guild else None,
                    guild_name=ctx.guild.name if ctx.guild else None,
                    channel_id=ctx.channel.id,
                    channel_name=ctx.channel.name,
                    user_id=ctx.author.id,
                    username=ctx.author.name
                )
                
                # Create meeting notes template
                meeting_content = f"""# Meeting: {meeting_topic}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Channel:** #{ctx.channel.name}
**Started by:** {ctx.author.name}

## Attendees
- {ctx.author.name}

## Agenda
- {meeting_topic}

## Discussion
[Meeting discussion will be captured here]

## Action Items
- [ ] [Action items will be added here]

## Next Steps
[Next steps and follow-ups]
"""
                
                result = await self.capture_text_note(
                    content=meeting_content,
                    discord_context=discord_context,
                    note_type="meeting",
                    title=f"Meeting: {meeting_topic}"
                )
                
                if result["success"]:
                    embed = discord.Embed(
                        title="📝 Meeting Notes Created",
                        description=f"**Topic:** {meeting_topic}\n**Note ID:** {result['note_id']}",
                        color=0xffa500
                    )
                    embed.add_field(
                        name="Next Steps",
                        value="• React with 🧠 to save important messages\n• Use `/thread_summary` to summarize discussion\n• Add attendees as they join",
                        inline=False
                    )
                    await ctx.send(embed=embed)
                else:
                    await ctx.send(f"❌ Failed to create meeting notes: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Meeting notes command failed: {e}")
                await ctx.send(f"❌ Error: {str(e)}")
        
        @self.bot.hybrid_command(name="stats", description="Show Second Brain statistics")
        async def show_stats(ctx):
            """Show usage statistics."""
            try:
                stats = await self.get_discord_usage_stats(ctx.guild.id if ctx.guild else None)
                
                embed = discord.Embed(
                    title="📊 Second Brain Statistics",
                    color=0x00aaff
                )
                
                embed.add_field(
                    name="Total Notes from Discord",
                    value=str(stats.get("total_discord_notes", 0)),
                    inline=True
                )
                
                embed.add_field(
                    name="This Server",
                    value=str(stats.get("server_notes", 0)),
                    inline=True
                )
                
                embed.add_field(
                    name="Most Used Channel",
                    value=stats.get("top_channel", "None"),
                    inline=True
                )
                
                if stats.get("recent_notes"):
                    recent_list = "\n".join([
                        f"• {note['title'][:40]}..." if len(note['title']) > 40 else f"• {note['title']}"
                        for note in stats["recent_notes"][:5]
                    ])
                    embed.add_field(
                        name="Recent Captures",
                        value=recent_list,
                        inline=False
                    )
                
                await ctx.send(embed=embed)
                
            except Exception as e:
                logger.error(f"Stats command failed: {e}")
                await ctx.send(f"❌ Error: {str(e)}")
    
    async def _handle_brain_reaction(self, reaction, user):
        """Handle 🧠 reaction to save message."""
        try:
            message = reaction.message
            
            discord_context = DiscordContext(
                guild_id=message.guild.id if message.guild else None,
                guild_name=message.guild.name if message.guild else None,
                channel_id=message.channel.id,
                channel_name=message.channel.name,
                user_id=user.id,
                username=user.name,
                message_id=message.id
            )
            
            # Capture message content
            content = message.content
            if message.attachments:
                content += "\n\n**Attachments:**\n"
                for attachment in message.attachments:
                    content += f"- {attachment.filename} ({attachment.url})\n"
            
            result = await self.capture_text_note(
                content=content,
                discord_context=discord_context,
                note_type="saved_message",
                title=f"Message from {message.author.name}",
                original_author=message.author.name
            )
            
            if result["success"]:
                # React with checkmark to confirm
                await message.add_reaction("✅")
            else:
                # React with X to indicate failure
                await message.add_reaction("❌")
                
        except Exception as e:
            logger.error(f"Brain reaction handler failed: {e}")
            await reaction.message.add_reaction("❌")
    
    async def _handle_summary_reaction(self, reaction, user):
        """Handle 📝 reaction to start thread summary."""
        try:
            message = reaction.message
            
            if isinstance(message.channel, discord.Thread):
                # Summarize the thread
                thread_capture = await self._capture_thread_messages(message.channel, 50)
                result = await self.process_thread_summary(thread_capture)
                
                if result["success"]:
                    embed = discord.Embed(
                        title=f"📝 Auto-Summary: {thread_capture.thread_name}",
                        description=result["summary"][:1000] + "..." if len(result["summary"]) > 1000 else result["summary"],
                        color=0x0099ff
                    )
                    await message.channel.send(embed=embed)
                    await message.add_reaction("✅")
                else:
                    await message.add_reaction("❌")
            else:
                # Just save the individual message with summary intent
                await self._handle_brain_reaction(reaction, user)
                
        except Exception as e:
            logger.error(f"Summary reaction handler failed: {e}")
            await reaction.message.add_reaction("❌")
    
    async def _handle_star_reaction(self, reaction, user):
        """Handle ⭐ reaction to mark as important."""
        try:
            message = reaction.message
            
            discord_context = DiscordContext(
                guild_id=message.guild.id if message.guild else None,
                guild_name=message.guild.name if message.guild else None,
                channel_id=message.channel.id,
                channel_name=message.channel.name,
                user_id=user.id,
                username=user.name,
                message_id=message.id
            )
            
            result = await self.capture_text_note(
                content=message.content,
                discord_context=discord_context,
                note_type="important_message",
                title=f"⭐ Important: {message.author.name}",
                original_author=message.author.name,
                tags=["important", "starred"]
            )
            
            if result["success"]:
                await message.add_reaction("✅")
            else:
                await message.add_reaction("❌")
                
        except Exception as e:
            logger.error(f"Star reaction handler failed: {e}")
            await reaction.message.add_reaction("❌")
    
    async def _capture_thread_messages(self, thread: discord.Thread, limit: int = 50) -> ThreadCapture:
        """Capture messages from a Discord thread."""
        messages = []
        participants = set()
        
        async for message in thread.history(limit=limit, oldest_first=True):
            if not message.author.bot:  # Skip bot messages
                messages.append({
                    "id": message.id,
                    "author": message.author.name,
                    "content": message.content,
                    "timestamp": message.created_at.isoformat(),
                    "attachments": [att.url for att in message.attachments]
                })
                participants.add(message.author.name)
        
        return ThreadCapture(
            thread_id=thread.id,
            thread_name=thread.name,
            messages=messages,
            participants=list(participants),
            start_time=messages[0]["timestamp"] if messages else datetime.now(),
            end_time=messages[-1]["timestamp"] if messages else datetime.now(),
            message_count=len(messages)
        )
    
    async def capture_text_note(
        self,
        content: str,
        discord_context: DiscordContext,
        note_type: str = "discord_note",
        title: str = None,
        original_author: str = None,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Capture text note from Discord."""
        try:
            # Generate title if not provided
            if not title:
                title = ollama_generate_title(content) or f"Discord {note_type.replace('_', ' ').title()}"
            
            # Base tags
            base_tags = ["discord", note_type]
            if discord_context.guild_name:
                base_tags.append(f"server-{discord_context.guild_name.lower()}")
            base_tags.append(f"channel-{discord_context.channel_name}")
            
            if tags:
                base_tags.extend(tags)
            
            # Process with AI
            summary = ""
            ai_tags = []
            actions = []
            
            try:
                ai_result = ollama_summarize(content)
                if ai_result.get("summary"):
                    summary = ai_result["summary"]
                if ai_result.get("tags"):
                    ai_tags.extend(ai_result["tags"][:3])
                if ai_result.get("actions"):
                    actions.extend(ai_result["actions"])
            except Exception as e:
                logger.warning(f"AI processing failed: {e}")
            
            all_tags = base_tags + ai_tags
            
            # Format content with Discord context
            formatted_content = content
            if original_author and original_author != discord_context.username:
                formatted_content = f"**Original Author:** {original_author}\n\n{content}"
            
            if summary:
                formatted_content = f"**Summary:** {summary}\n\n{formatted_content}"
            
            formatted_content += f"\n\n**Discord Context:**\n"
            formatted_content += f"- Server: {discord_context.guild_name or 'DM'}\n"
            formatted_content += f"- Channel: #{discord_context.channel_name}\n"
            formatted_content += f"- Captured by: {discord_context.username}\n"
            formatted_content += f"- Timestamp: {datetime.now().isoformat()}\n"
            
            if actions:
                formatted_content += f"\n**Action Items:**\n" + "\n".join([f"- {action}" for action in actions])
            
            # Save to database
            note_id = await self._save_note(
                title=title,
                content=formatted_content,
                tags=all_tags,
                metadata={
                    "content_type": note_type,
                    "source": "discord",
                    "discord_context": {
                        "guild_id": discord_context.guild_id,
                        "guild_name": discord_context.guild_name,
                        "channel_id": discord_context.channel_id,
                        "channel_name": discord_context.channel_name,
                        "user_id": discord_context.user_id,
                        "username": discord_context.username,
                        "message_id": discord_context.message_id
                    },
                    "original_author": original_author,
                    "has_ai_summary": bool(summary),
                    "action_items_count": len(actions)
                }
            )
            
            return {
                "success": True,
                "note_id": note_id,
                "title": title,
                "summary": summary,
                "tags": all_tags,
                "action_items": actions,
                "message": "Note captured successfully"
            }
            
        except Exception as e:
            logger.error(f"Discord note capture failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_thread_summary(self, thread_capture: ThreadCapture) -> Dict[str, Any]:
        """Process and summarize a Discord thread."""
        try:
            # Format thread content for AI processing
            thread_content = f"Thread: {thread_capture.thread_name}\n"
            thread_content += f"Participants: {', '.join(thread_capture.participants)}\n"
            thread_content += f"Messages: {thread_capture.message_count}\n\n"
            
            for msg in thread_capture.messages:
                thread_content += f"[{msg['timestamp']}] {msg['author']}: {msg['content']}\n"
                if msg['attachments']:
                    thread_content += f"  Attachments: {', '.join(msg['attachments'])}\n"
                thread_content += "\n"
            
            # AI processing for summary
            ai_prompt = """Summarize this Discord thread conversation. Extract:
1. Main topic and key discussion points
2. Important decisions made
3. Action items and assignments
4. Key insights or conclusions
Provide a structured summary."""
            
            ai_result = ollama_summarize(thread_content[:5000], ai_prompt)  # Limit content
            
            summary = ai_result.get("summary", "Thread discussion captured")
            key_points = []
            action_items = ai_result.get("actions", [])
            
            # Extract key points from summary
            if summary:
                sentences = summary.split('. ')
                key_points = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10][:5]
            
            # Save thread summary
            title = f"Thread Summary: {thread_capture.thread_name}"
            tags = ["discord", "thread-summary", "conversation"] + [f"participants-{len(thread_capture.participants)}"]
            
            formatted_content = f"**Thread:** {thread_capture.thread_name}\n"
            formatted_content += f"**Participants:** {', '.join(thread_capture.participants)}\n"
            formatted_content += f"**Messages:** {thread_capture.message_count}\n"
            formatted_content += f"**Duration:** {thread_capture.start_time} to {thread_capture.end_time}\n\n"
            formatted_content += f"**Summary:**\n{summary}\n\n"
            
            if key_points:
                formatted_content += f"**Key Points:**\n" + "\n".join([f"• {point}" for point in key_points]) + "\n\n"
            
            if action_items:
                formatted_content += f"**Action Items:**\n" + "\n".join([f"• {item}" for item in action_items]) + "\n\n"
            
            formatted_content += f"**Full Conversation:**\n{thread_content}"
            
            note_id = await self._save_note(
                title=title,
                content=formatted_content,
                tags=tags,
                metadata={
                    "content_type": "thread_summary",
                    "source": "discord",
                    "thread_id": thread_capture.thread_id,
                    "thread_name": thread_capture.thread_name,
                    "message_count": thread_capture.message_count,
                    "participants": thread_capture.participants,
                    "action_items_count": len(action_items)
                }
            )
            
            return {
                "success": True,
                "note_id": note_id,
                "title": title,
                "summary": summary,
                "key_points": key_points,
                "action_items": action_items,
                "message": "Thread summarized successfully"
            }
            
        except Exception as e:
            logger.error(f"Thread summary processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_notes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search notes in Second Brain."""
        try:
            # Use existing search service
            from services.search_adapter import SearchService
            search_service = SearchService("notes.db")  # Use default database
            
            results = search_service.search(query, mode='hybrid', k=limit)
            
            return [
                {
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "body": result.get("body"),
                    "tags": result.get("tags"),
                    "created_at": result.get("created_at"),
                    "score": result.get("score", 0)
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Note search failed: {e}")
            return []
    
    async def get_discord_usage_stats(self, guild_id: Optional[int] = None) -> Dict[str, Any]:
        """Get Discord usage statistics."""
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            
            # Total Discord notes
            cursor.execute("SELECT COUNT(*) FROM notes WHERE json_extract(metadata, '$.source') = 'discord'")
            total_discord_notes = cursor.fetchone()[0]
            
            # Server-specific notes
            server_notes = 0
            if guild_id:
                cursor.execute("""
                    SELECT COUNT(*) FROM notes 
                    WHERE json_extract(metadata, '$.source') = 'discord'
                    AND json_extract(metadata, '$.discord_context.guild_id') = ?
                """, (guild_id,))
                server_notes = cursor.fetchone()[0]
            
            # Most used channel
            cursor.execute("""
                SELECT json_extract(metadata, '$.discord_context.channel_name') as channel, COUNT(*) as count
                FROM notes 
                WHERE json_extract(metadata, '$.source') = 'discord'
                AND channel IS NOT NULL
                GROUP BY channel
                ORDER BY count DESC
                LIMIT 1
            """)
            
            top_channel_row = cursor.fetchone()
            top_channel = f"#{top_channel_row[0]} ({top_channel_row[1]} notes)" if top_channel_row else "None"
            
            # Recent notes
            cursor.execute("""
                SELECT title, created_at FROM notes 
                WHERE json_extract(metadata, '$.source') = 'discord'
                ORDER BY created_at DESC
                LIMIT 5
            """)
            
            recent_notes = [
                {"title": row[0], "created_at": row[1]}
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                "total_discord_notes": total_discord_notes,
                "server_notes": server_notes,
                "top_channel": top_channel,
                "recent_notes": recent_notes
            }
            
        except Exception as e:
            logger.error(f"Failed to get Discord stats: {e}")
            return {}
    
    async def _save_note(
        self,
        title: str,
        content: str,
        tags: List[str],
        metadata: Dict[str, Any]
    ) -> int:
        """Save note to database with embeddings."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        try:
            # Clean tags
            tags_str = ", ".join(set(tag.strip() for tag in tags if tag.strip()))
            
            # Insert note
            cursor.execute("""
                INSERT INTO notes (title, body, tags, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                title,
                content,
                tags_str,
                json.dumps(metadata),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            note_id = cursor.lastrowid
            
            # Generate embeddings
            try:
                embedding_text = f"{title}\n\n{content}"
                embedding = self.embedder.embed(embedding_text)
                
                # Store in vector table if available
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'")
                    if cursor.fetchone():
                        cursor.execute(
                            "INSERT OR REPLACE INTO note_vecs(note_id, embedding) VALUES (?, ?)",
                            (note_id, json.dumps(embedding))
                        )
                except Exception as e:
                    logger.debug(f"Vector storage not available: {e}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
            
            conn.commit()
            return note_id
            
        finally:
            conn.close()
    
    def start_bot(self):
        """Start the Discord bot."""
        if not settings.discord_bot_token or settings.discord_bot_token == "your-discord-bot-token":
            logger.warning("Discord bot token not configured. Skipping bot startup.")
            return
        
        try:
            self.bot.run(settings.discord_bot_token)
        except Exception as e:
            logger.error(f"Discord bot startup failed: {e}")


def get_enhanced_discord_service(get_conn_func):
    """Factory function to get enhanced Discord service."""
    return EnhancedDiscordService(get_conn_func)