# ──────────────────────────────────────────────────────────────────────────────
# File: services/enhanced_apple_shortcuts_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced Apple Shortcuts Integration for Second Brain

Advanced shortcuts for iOS/macOS including voice memos, photo OCR, location-based notes,
quick capture workflows, and deep integration with iOS features.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import base64
import hashlib

from config import settings
from services.embeddings import Embeddings
from services.advanced_capture_service import get_advanced_capture_service, CaptureOptions
from llm_utils import ollama_summarize, ollama_generate_title

logger = logging.getLogger(__name__)

class EnhancedAppleShortcutsService:
    """Enhanced Apple Shortcuts integration service."""
    
    def __init__(self, get_conn_func):
        """Initialize with database connection function."""
        self.get_conn = get_conn_func
        self.embedder = Embeddings()
        self.advanced_capture = None
    
    def _get_advanced_capture(self):
        """Lazy load advanced capture service."""
        if not self.advanced_capture:
            self.advanced_capture = get_advanced_capture_service(self.get_conn)
        return self.advanced_capture
    
    async def process_voice_memo(
        self, 
        audio_data: str = None,
        audio_url: str = None,
        transcription: str = None,
        location_data: Dict = None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """
        Process voice memo from iOS Shortcuts.
        
        Args:
            audio_data: Base64 encoded audio data
            audio_url: URL to audio file (if hosted)
            transcription: Pre-transcribed text from iOS
            location_data: GPS location info
            context: Additional context (time, app, etc.)
        """
        try:
            # Use transcription if provided (iOS can do this locally)
            if not transcription and audio_data:
                # Would need audio transcription service here
                # For now, use placeholder
                transcription = "Voice memo captured (transcription not available)"
            
            if not transcription:
                return {"success": False, "error": "No transcription or audio data provided"}
            
            # Generate title
            title = ollama_generate_title(transcription) or "Voice Memo"
            
            # Process with AI
            tags = ["voice-memo", "audio", "ios-shortcut"]
            summary = ""
            actions = []
            
            try:
                ai_result = ollama_summarize(transcription)
                if ai_result.get("summary"):
                    summary = ai_result["summary"]
                if ai_result.get("tags"):
                    tags.extend(ai_result["tags"])
                if ai_result.get("actions"):
                    actions.extend(ai_result["actions"])
            except Exception as e:
                logger.warning(f"AI processing failed: {e}")
            
            # Add location context if available
            location_text = ""
            if location_data:
                lat = location_data.get("latitude")
                lng = location_data.get("longitude")
                address = location_data.get("address", "")
                
                if lat and lng:
                    location_text = f"\n**Location:** {address} ({lat:.4f}, {lng:.4f})"
                    tags.append("location")
            
            # Add context information
            context_text = ""
            if context:
                timestamp = context.get("timestamp")
                app_name = context.get("app_name")
                device = context.get("device")
                
                context_parts = []
                if timestamp:
                    context_parts.append(f"Recorded: {timestamp}")
                if app_name:
                    context_parts.append(f"App: {app_name}")
                if device:
                    context_parts.append(f"Device: {device}")
                
                if context_parts:
                    context_text = f"\n**Context:** {' | '.join(context_parts)}"
            
            # Format content
            content = transcription
            if summary:
                content = f"**Summary:** {summary}\n\n**Transcription:**\n{transcription}"
            
            content += location_text + context_text
            
            if actions:
                content += f"\n\n**Action Items:**\n" + "\n".join([f"- {action}" for action in actions])
            
            # Save to database
            note_id = await self._save_note(
                title=title,
                content=content,
                tags=tags,
                metadata={
                    "content_type": "voice_memo",
                    "source": "ios_shortcuts",
                    "has_audio": bool(audio_data or audio_url),
                    "transcription_method": "ios_builtin" if transcription else "server",
                    "location": location_data,
                    "context": context,
                    "action_items_count": len(actions)
                }
            )
            
            return {
                "success": True,
                "note_id": note_id,
                "title": title,
                "summary": summary,
                "action_items": actions,
                "tags": tags,
                "message": "Voice memo processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Voice memo processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_photo_ocr(
        self,
        image_data: str,
        location_data: Dict = None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """
        Process photo with OCR from iOS Shortcuts.
        
        Args:
            image_data: Base64 encoded image
            location_data: GPS location info
            context: Additional context
        """
        try:
            options = CaptureOptions(
                enable_ocr=True,
                enable_ai_processing=True,
                custom_tags=["photo", "ocr", "ios-shortcut"]
            )
            
            # Add location tag if available
            if location_data:
                options.custom_tags.append("location")
            
            # Use advanced capture service for OCR
            advanced_capture = self._get_advanced_capture()
            result = await advanced_capture.capture_screenshot_with_ocr(image_data, options)
            
            if result.success:
                # Add location and context info
                additional_content = ""
                
                if location_data:
                    lat = location_data.get("latitude")
                    lng = location_data.get("longitude")
                    address = location_data.get("address", "")
                    
                    if lat and lng:
                        additional_content += f"\n**Location:** {address} ({lat:.4f}, {lng:.4f})"
                
                if context:
                    timestamp = context.get("timestamp")
                    if timestamp:
                        additional_content += f"\n**Captured:** {timestamp}"
                
                if additional_content:
                    # Update the content in database
                    conn = self.get_conn()
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        "UPDATE notes SET body = ? WHERE id = ?",
                        (result.content + additional_content, result.note_id)
                    )
                    conn.commit()
                    conn.close()
                
                return {
                    "success": True,
                    "note_id": result.note_id,
                    "title": result.title,
                    "extracted_text": result.content,
                    "tags": result.tags,
                    "message": "Photo OCR processed successfully"
                }
            else:
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            logger.error(f"Photo OCR processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_quick_note(
        self,
        text: str,
        note_type: str = "thought",
        location_data: Dict = None,
        context: Dict = None,
        auto_tag: bool = True
    ) -> Dict[str, Any]:
        """
        Process quick note from iOS Shortcuts.
        
        Args:
            text: Note content
            note_type: Type of note (thought, task, idea, meeting, etc.)
            location_data: GPS location info
            context: Additional context
            auto_tag: Whether to auto-generate tags
        """
        try:
            # Generate title
            title = ollama_generate_title(text) or f"{note_type.title()} Note"
            
            # Base tags
            tags = [note_type, "ios-shortcut", "quick-note"]
            
            # Add location tag
            if location_data:
                tags.append("location")
            
            # Auto-generate tags if enabled
            if auto_tag:
                try:
                    ai_result = ollama_summarize(text)
                    if ai_result.get("tags"):
                        tags.extend(ai_result["tags"][:5])  # Limit to 5 AI tags
                except Exception as e:
                    logger.warning(f"Auto-tagging failed: {e}")
            
            # Add location and context
            content = text
            
            if location_data:
                lat = location_data.get("latitude")
                lng = location_data.get("longitude")
                address = location_data.get("address", "")
                
                if lat and lng:
                    content += f"\n\n**Location:** {address} ({lat:.4f}, {lng:.4f})"
            
            if context:
                timestamp = context.get("timestamp")
                app_name = context.get("source_app")
                
                context_parts = []
                if timestamp:
                    context_parts.append(f"Created: {timestamp}")
                if app_name:
                    context_parts.append(f"From: {app_name}")
                
                if context_parts:
                    content += f"\n**Context:** {' | '.join(context_parts)}"
            
            # Save note
            note_id = await self._save_note(
                title=title,
                content=content,
                tags=tags,
                metadata={
                    "content_type": f"quick_{note_type}",
                    "source": "ios_shortcuts",
                    "location": location_data,
                    "context": context,
                    "auto_tagged": auto_tag
                }
            )
            
            return {
                "success": True,
                "note_id": note_id,
                "title": title,
                "tags": tags,
                "message": f"{note_type.title()} note saved successfully"
            }
            
        except Exception as e:
            logger.error(f"Quick note processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_web_clip(
        self,
        url: str,
        selected_text: str = None,
        page_title: str = None,
        context: Dict = None
    ) -> Dict[str, Any]:
        """
        Process web clip from iOS Safari Share Sheet.
        
        Args:
            url: Web page URL
            selected_text: Selected text from page
            page_title: Page title
            context: Additional context
        """
        try:
            # Use web ingestion service
            from services.web_ingestion_service import WebIngestionService
            web_service = WebIngestionService()
            
            # If there's selected text, prioritize that
            if selected_text:
                title = page_title or ollama_generate_title(selected_text) or "Web Clip"
                
                content = f"**Source:** {url}\n\n"
                if page_title and page_title != title:
                    content += f"**Page:** {page_title}\n\n"
                content += f"**Selected Text:**\n{selected_text}"
                
                # Add context
                if context:
                    timestamp = context.get("timestamp")
                    if timestamp:
                        content += f"\n\n**Clipped:** {timestamp}"
                
                tags = ["web-clip", "ios-shortcut", "selected-text"]
                
                # Auto-generate tags from content
                try:
                    ai_result = ollama_summarize(selected_text)
                    if ai_result.get("tags"):
                        tags.extend(ai_result["tags"][:3])
                except Exception as e:
                    logger.warning(f"Auto-tagging failed: {e}")
                
                note_id = await self._save_note(
                    title=title,
                    content=content,
                    tags=tags,
                    metadata={
                        "content_type": "web_clip_selection",
                        "source": "ios_shortcuts",
                        "source_url": url,
                        "page_title": page_title,
                        "context": context,
                        "has_selection": True
                    }
                )
                
                return {
                    "success": True,
                    "note_id": note_id,
                    "title": title,
                    "content_type": "selection",
                    "message": "Web selection clipped successfully"
                }
            
            else:
                # Process full page
                web_result = await web_service.extract_and_process_url(url)
                
                if web_result:
                    # Add iOS context
                    additional_content = ""
                    if context:
                        timestamp = context.get("timestamp")
                        if timestamp:
                            additional_content = f"\n\n**Clipped via iOS:** {timestamp}"
                    
                    tags = ["web-clip", "ios-shortcut", "full-page"]
                    
                    note_id = await self._save_note(
                        title=web_result.title,
                        content=web_result.content + additional_content,
                        tags=tags,
                        metadata={
                            "content_type": "web_clip_full",
                            "source": "ios_shortcuts",
                            "source_url": url,
                            "context": context,
                            "has_selection": False
                        }
                    )
                    
                    return {
                        "success": True,
                        "note_id": note_id,
                        "title": web_result.title,
                        "content_type": "full_page",
                        "message": "Web page clipped successfully"
                    }
                else:
                    return {"success": False, "error": "Failed to extract web content"}
            
        except Exception as e:
            logger.error(f"Web clip processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_shortcut_templates(self) -> List[Dict[str, Any]]:
        """Get pre-built iOS Shortcuts templates."""
        return [
            {
                "name": "Quick Voice Memo",
                "description": "Record and transcribe voice memos with location",
                "endpoint": "/api/shortcuts/voice-memo",
                "method": "POST",
                "parameters": {
                    "transcription": "Quick Note dictated text",
                    "location_data": {
                        "latitude": 37.7749,
                        "longitude": -122.4194,
                        "address": "San Francisco, CA"
                    },
                    "context": {
                        "timestamp": "2024-12-15T10:30:00Z",
                        "device": "iPhone"
                    }
                },
                "shortcut_url": f"{settings.base_dir}/shortcuts/voice_memo.shortcut"
            },
            {
                "name": "Photo OCR Capture",
                "description": "Take photo and extract text with OCR",
                "endpoint": "/api/shortcuts/photo-ocr",
                "method": "POST",
                "parameters": {
                    "image_data": "base64_image_data_here",
                    "location_data": {
                        "latitude": 37.7749,
                        "longitude": -122.4194
                    }
                },
                "shortcut_url": f"{settings.base_dir}/shortcuts/photo_ocr.shortcut"
            },
            {
                "name": "Quick Thought Capture",
                "description": "Quickly capture thoughts and ideas",
                "endpoint": "/api/shortcuts/quick-note",
                "method": "POST",
                "parameters": {
                    "text": "Your thought or idea here",
                    "note_type": "thought",
                    "auto_tag": True
                },
                "shortcut_url": f"{settings.base_dir}/shortcuts/quick_thought.shortcut"
            },
            {
                "name": "Web Clip from Safari",
                "description": "Clip web content from Safari share sheet",
                "endpoint": "/api/shortcuts/web-clip",
                "method": "POST",
                "parameters": {
                    "url": "https://example.com",
                    "selected_text": "Selected text from page",
                    "page_title": "Page Title"
                },
                "shortcut_url": f"{settings.base_dir}/shortcuts/web_clip.shortcut"
            },
            {
                "name": "Meeting Notes Starter",
                "description": "Quick meeting notes with attendees and agenda",
                "endpoint": "/api/shortcuts/quick-note", 
                "method": "POST",
                "parameters": {
                    "text": "Meeting with [attendees] about [topic]",
                    "note_type": "meeting",
                    "auto_tag": True
                },
                "shortcut_url": f"{settings.base_dir}/shortcuts/meeting_notes.shortcut"
            }
        ]
    
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


def get_enhanced_apple_shortcuts_service(get_conn_func):
    """Factory function to get enhanced Apple shortcuts service."""
    return EnhancedAppleShortcutsService(get_conn_func)