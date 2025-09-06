"""
Content Deduplication Service

Provides intelligent content deduplication with configurable similarity thresholds
and efficient database queries.
"""

import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

@dataclass
class DuplicationResult:
    """Result of content duplication check."""
    is_duplicate: bool
    existing_note_id: Optional[int] = None
    similarity_score: float = 0.0
    match_type: str = "none"  # "exact", "fuzzy", "semantic"
    existing_title: Optional[str] = None
    existing_content_preview: Optional[str] = None

class ContentDeduplicationService:
    """Service for intelligent content deduplication."""
    
    def __init__(self, get_conn_func):
        """Initialize with database connection function."""
        self.get_conn = get_conn_func
    
    def compute_content_hash(self, title: str, content: str) -> str:
        """Compute a normalized content hash for exact duplicate detection."""
        # Normalize content for hashing
        normalized = self._normalize_content(title, content)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _normalize_content(self, title: str, content: str) -> str:
        """Normalize content for consistent hashing."""
        combined = f"{title or ''}\n\n{content or ''}"
        
        # Remove extra whitespace but preserve structure
        lines = []
        for line in combined.split('\n'):
            # Strip each line but keep empty lines for structure
            normalized_line = ' '.join(line.split()) if line.strip() else ''
            lines.append(normalized_line)
        
        # Join and convert to lowercase for case-insensitive comparison
        return '\n'.join(lines).strip().lower()
    
    def check_for_duplicates(
        self, 
        title: str, 
        content: str,
        user_id: int,
        window_days: int = 30,
        fuzzy_threshold: float = 0.85
    ) -> DuplicationResult:
        """
        Check for duplicate content using multiple strategies.
        
        Args:
            title: Note title
            content: Note content
            user_id: User ID to scope search
            window_days: Look back window in days (0 = no limit)
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
            
        Returns:
            DuplicationResult with match information
        """
        try:
            # Step 1: Exact hash match (fastest)
            content_hash = self.compute_content_hash(title, content)
            exact_match = self._check_exact_hash_match(content_hash, user_id, window_days)
            if exact_match:
                return exact_match
            
            # Step 2: Fuzzy content matching (more expensive)
            if fuzzy_threshold > 0:
                fuzzy_match = self._check_fuzzy_match(
                    title, content, user_id, window_days, fuzzy_threshold
                )
                if fuzzy_match:
                    return fuzzy_match
            
            # No duplicates found
            return DuplicationResult(is_duplicate=False)
            
        except Exception as e:
            log.error("Error checking for duplicates: %s", e)
            # On error, assume no duplicates to avoid blocking content creation
            return DuplicationResult(is_duplicate=False)
    
    def _check_exact_hash_match(
        self, 
        content_hash: str, 
        user_id: int, 
        window_days: int
    ) -> Optional[DuplicationResult]:
        """Check for exact content hash matches."""
        conn = self.get_conn()
        try:
            # Build query with optional time window
            query = """
                SELECT id, title, 
                       substr(content, 1, 100) as content_preview
                FROM notes 
                WHERE content_hash = ? AND user_id = ?
            """
            params = [content_hash, user_id]
            
            if window_days > 0:
                cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
                query += " AND created_at >= ?"
                params.append(cutoff)
            
            query += " ORDER BY created_at DESC LIMIT 1"
            
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return DuplicationResult(
                    is_duplicate=True,
                    existing_note_id=row[0],
                    similarity_score=1.0,
                    match_type="exact",
                    existing_title=row[1],
                    existing_content_preview=row[2]
                )
            
            return None
            
        finally:
            conn.close()
    
    def _check_fuzzy_match(
        self,
        title: str,
        content: str,
        user_id: int,
        window_days: int,
        threshold: float
    ) -> Optional[DuplicationResult]:
        """Check for fuzzy content matches using similarity scoring."""
        conn = self.get_conn()
        try:
            # Get recent notes for comparison
            query = """
                SELECT id, title, content 
                FROM notes 
                WHERE user_id = ? AND content IS NOT NULL
            """
            params = [user_id]
            
            if window_days > 0:
                cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
                query += " AND created_at >= ?"
                params.append(cutoff)
            
            # Limit to reasonable number for performance
            query += " ORDER BY created_at DESC LIMIT 50"
            
            cursor = conn.execute(query, params)
            candidates = cursor.fetchall()
            
            # Compare with each candidate
            normalized_input = self._normalize_content(title, content)
            best_match = None
            best_score = 0.0
            
            for note_id, existing_title, existing_content in candidates:
                normalized_existing = self._normalize_content(existing_title, existing_content)
                
                # Calculate similarity score
                similarity = SequenceMatcher(None, normalized_input, normalized_existing).ratio()
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = (note_id, existing_title, existing_content)
            
            if best_match:
                note_id, existing_title, existing_content = best_match
                return DuplicationResult(
                    is_duplicate=True,
                    existing_note_id=note_id,
                    similarity_score=best_score,
                    match_type="fuzzy",
                    existing_title=existing_title,
                    existing_content_preview=existing_content[:100] if existing_content else None
                )
            
            return None
            
        finally:
            conn.close()
    
    def update_existing_note(
        self, 
        note_id: int, 
        new_title: str = None, 
        new_content: str = None
    ) -> bool:
        """Update an existing note's timestamp and optionally content."""
        conn = self.get_conn()
        try:
            if new_title or new_content:
                # Update content and timestamp
                updates = []
                params = []
                
                if new_title:
                    updates.append("title = ?")
                    params.append(new_title)
                
                if new_content:
                    updates.append("content = ?")
                    params.append(new_content)
                    
                    # Update content hash too
                    content_hash = self.compute_content_hash(new_title or "", new_content)
                    updates.append("content_hash = ?")
                    params.append(content_hash)
                
                updates.append("updated_at = ?")
                params.append(datetime.now().isoformat())
                params.append(note_id)
                
                query = f"UPDATE notes SET {', '.join(updates)} WHERE id = ?"
                conn.execute(query, params)
            else:
                # Just update timestamp
                conn.execute(
                    "UPDATE notes SET updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), note_id)
                )
            
            conn.commit()
            return True
            
        except Exception as e:
            log.error("Error updating existing note %s: %s", note_id, e)
            conn.rollback()
            return False
        finally:
            conn.close()

# Global instance
_deduplication_service = None

def get_deduplication_service(get_conn_func):
    """Get global deduplication service instance."""
    global _deduplication_service
    if _deduplication_service is None:
        _deduplication_service = ContentDeduplicationService(get_conn_func)
    return _deduplication_service