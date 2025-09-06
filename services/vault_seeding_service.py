"""
Vault Seeding Service for Second Brain

This service provides programmatic access to vault seeding functionality,
allowing the web application to initialize vaults with starter content.
"""

import logging
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import the seeding script functionality
import sys
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

try:
    from seed_starter_vault import (
        seed_active_vault, SeedCfg, SEED_NOTES, SEED_BOOKMARKS,
        try_ollama_embed, ensure_embeddings_schema, db_conn
    )
except ImportError as e:
    logging.warning("Could not import seeding functionality: %s", e)
    seed_active_vault = None

from config import settings

log = logging.getLogger(__name__)

@dataclass
class SeedingResult:
    """Result of a seeding operation."""
    success: bool
    message: str
    notes_created: int = 0
    embeddings_created: int = 0
    files_written: int = 0
    error: Optional[str] = None

@dataclass
class SeedingOptions:
    """Configuration options for vault seeding."""
    namespace: str = settings.auto_seeding_namespace
    force_overwrite: bool = False
    include_embeddings: bool = True
    embed_model: str = settings.auto_seeding_embed_model
    ollama_url: str = "http://localhost:11434"
    refresh_search_indices: bool = settings.auto_seeding_refresh_indices

class VaultSeedingService:
    """Service for seeding vaults with starter content."""
    
    def __init__(self, get_conn_func):
        """Initialize with database connection function."""
        self.get_conn = get_conn_func
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> bool:
        """Check if all seeding dependencies are available."""
        if seed_active_vault is None:
            log.error("Seeding script not available")
            return False
        
        # Check if vault path exists
        vault_path = Path(settings.vault_path)
        if not vault_path.exists():
            log.warning("Vault path does not exist: %s", vault_path)
            vault_path.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def get_seeding_status(self, user_id: int) -> Dict[str, Any]:
        """Check if vault has been seeded and get status."""
        conn = self.get_conn()
        try:
            # Check for seed notes in database (by tag and namespace)
            cursor = conn.execute(
                """
                SELECT COUNT(*) as seed_count
                FROM notes
                WHERE user_id = ?
                  AND (
                      tags LIKE '%seed%'
                      OR tags LIKE ?
                      OR content LIKE ?
                      OR title LIKE ?
                  )
                """,
                (
                    user_id,
                    f"%ns:{settings.auto_seeding_namespace}%",
                    f"%{settings.auto_seeding_namespace}%",
                    f"%{settings.auto_seeding_namespace}%",
                ),
            )
            
            seed_count = cursor.fetchone()[0]
            
            # Check for seed files in vault
            vault_path = Path(settings.vault_path)
            seed_namespace = vault_path / settings.auto_seeding_namespace
            files_exist = seed_namespace.exists() and any(seed_namespace.rglob("*.md"))
            
            return {
                "is_seeded": seed_count > 0 or files_exist,
                "seed_notes_count": seed_count,
                "seed_files_exist": files_exist,
                "vault_path": str(vault_path),
                "seed_namespace": str(seed_namespace) if seed_namespace.exists() else None
            }
            
        except Exception as e:
            log.error("Error checking seeding status: %s", e)
            return {
                "is_seeded": False,
                "error": str(e)
            }
        finally:
            conn.close()
    
    def get_available_seed_content(self) -> Dict[str, Any]:
        """Get information about available seed content."""
        if not self.is_seeding_available():
            return {}
        
        try:
            notes_data = [
                {
                    "id": note["id"],
                    "title": note["title"],
                    "type": note["type"],
                    "tags": note["tags"].split(", "),
                    "summary": note["summary"]
                }
                for note in SEED_NOTES
            ]
            
            bookmarks_data = [
                {
                    "id": bookmark["id"],
                    "title": bookmark["title"],
                    "url": bookmark["url"],
                    "tags": bookmark["tags"].split(", "),
                    "summary": bookmark["summary"]
                }
                for bookmark in SEED_BOOKMARKS
            ]
            
            return {
                "notes": {
                    "items": notes_data,
                    "count": len(notes_data)
                },
                "bookmarks": {
                    "items": bookmarks_data,
                    "count": len(bookmarks_data)
                },
                "total_items": len(notes_data) + len(bookmarks_data)
            }
        except (NameError, AttributeError):
            # SEED_NOTES or SEED_BOOKMARKS not available
            return {}
    
    def seed_vault(self, user_id: int, options: SeedingOptions) -> SeedingResult:
        """Seed the vault with starter content."""
        if not self._validate_dependencies():
            return SeedingResult(
                success=False,
                message="Seeding dependencies not available",
                error="Missing seeding script or configuration"
            )
        
        try:
            # Create seeding configuration
            cfg = SeedCfg(
                db_path=Path(settings.db_path),
                vault_path=Path(settings.vault_path),
                namespace=options.namespace,
                force=options.force_overwrite,
                no_embed=not options.include_embeddings,
                embed_model=options.embed_model,
                ollama_url=options.ollama_url,
                user_id=user_id
            )
            
            # Count existing content and embeddings before seeding
            initial_count = self._count_seeded_notes(options.namespace)
            initial_embeddings = self._count_seeded_embeddings(options.namespace)
            
            # Perform the seeding
            seed_active_vault(cfg)
            
            # Count content after seeding
            final_count = self._count_seeded_notes(options.namespace)
            final_embeddings = self._count_seeded_embeddings(options.namespace)

            notes_created = final_count - initial_count
            embeddings_created = final_embeddings - initial_embeddings
            
            # Count files written
            seed_path = Path(settings.vault_path) / options.namespace
            files_written = len(list(seed_path.rglob("*.md"))) if seed_path.exists() else 0
            
            return SeedingResult(
                success=True,
                message=f"Successfully seeded vault with {notes_created} notes and {files_written} files",
                notes_created=notes_created,
                files_written=files_written,
                embeddings_created=embeddings_created if options.include_embeddings else 0,
            )
            
            # Optionally refresh search indexes (best-effort)
            if options.refresh_search_indices:
                try:
                    self._refresh_search_indexes()
                except Exception as e:
                    log.warning("Search index refresh skipped/failed: %s", e)
            
        except Exception as e:
            log.error("Vault seeding failed: %s", e)
            return SeedingResult(
                success=False,
                message="Vault seeding failed",
                error=str(e)
            )
    
    def clear_seed_content(self, user_id: int, namespace: str = settings.auto_seeding_namespace) -> SeedingResult:
        """Remove seed content from vault and database."""
        try:
            # Remove from database
            conn = self.get_conn()
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM notes
                    WHERE user_id = ?
                      AND (
                          tags LIKE '%seed%'
                          OR tags LIKE ?
                          OR content LIKE ?
                          OR title LIKE ?
                      )
                    """,
                    (
                        user_id,
                        f"%ns:{namespace}%",
                        f"%{namespace}%",
                        f"%{namespace}%",
                    ),
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
            finally:
                conn.close()
            
            # Remove files
            seed_path = Path(settings.vault_path) / namespace
            files_removed = 0
            if seed_path.exists():
                for file_path in seed_path.rglob("*.md"):
                    file_path.unlink()
                    files_removed += 1
                
                # Remove empty directories
                try:
                    seed_path.rmdir()
                except OSError:
                    pass  # Directory not empty, leave it
            
            return SeedingResult(
                success=True,
                message=f"Removed {deleted_count} seed notes and {files_removed} files",
                notes_created=-deleted_count,  # Negative indicates removal
                files_written=-files_removed
            )
            
        except Exception as e:
            log.error("Failed to clear seed content: %s", e)
            return SeedingResult(
                success=False,
                message="Failed to clear seed content",
                error=str(e)
            )
    
    def test_ollama_connection(self, url: str = "http://localhost:11434") -> Dict[str, Any]:
        """Test connection to Ollama for embeddings."""
        try:
            test_result = try_ollama_embed(["test"], url=url)
            return {
                "available": test_result is not None,
                "url": url,
                "message": "Ollama connection successful" if test_result else "Ollama connection failed"
            }
        except Exception as e:
            return {
                "available": False,
                "url": url,
                "message": f"Ollama connection error: {str(e)}"
            }
    
    def preview_seeding_impact(self, user_id: int) -> Dict[str, Any]:
        """Preview what would happen if seeding is performed."""
        current_status = self.get_seeding_status(user_id)
        available_content = self.get_available_seed_content()
        ollama_status = self.test_ollama_connection()
        
        return {
            "current_status": current_status,
            "available_content": available_content,
            "ollama_status": ollama_status,
            "estimated_notes": available_content["total_items"],
            "estimated_files": available_content["total_items"],
            "estimated_embeddings": available_content["total_items"] if ollama_status["available"] else 0,
            "will_overwrite": current_status["is_seeded"]
        }
    
    def is_seeding_available(self) -> bool:
        """Check if seeding functionality is available."""
        return seed_active_vault is not None
    
    def seed_vault_with_content(self, options: SeedingOptions) -> SeedingResult:
        """Seed the vault with content using provided options."""
        if not self.is_seeding_available():
            return SeedingResult(
                success=False,
                message="Seeding functionality not available",
                error="Seeding dependencies not installed or configured"
            )
        
        try:
            # Use default user_id of 1 if not specified in options
            user_id = getattr(options, 'user_id', 1)
            
            # Create seeding configuration
            cfg = SeedCfg(
                db_path=Path(settings.db_path),
                vault_path=Path(settings.vault_path),
                namespace=options.namespace,
                force=options.force_overwrite,
                no_embed=not options.include_embeddings,
                embed_model=options.embed_model,
                ollama_url=options.ollama_url,
                user_id=user_id
            )
            
            # Count existing content before seeding
            initial_notes = self._count_seeded_notes(options.namespace)
            initial_embeddings = self._count_seeded_embeddings(options.namespace)
            
            # Perform the seeding
            seed_active_vault(cfg)
            
            # Count content after seeding
            final_notes = self._count_seeded_notes(options.namespace)
            final_embeddings = self._count_seeded_embeddings(options.namespace)
            
            notes_created = final_notes - initial_notes
            embeddings_created = final_embeddings - initial_embeddings
            
            result = SeedingResult(
                success=True,
                message=f"Successfully seeded vault with {notes_created} notes",
                notes_created=notes_created,
                embeddings_created=embeddings_created
            )
            
            # Log the operation
            self._log_seeding_operation(result)
            
            return result
            
        except Exception as e:
            log.error("Vault seeding failed: %s", e)
            result = SeedingResult(
                success=False,
                message="Vault seeding failed",
                error=str(e)
            )
            self._log_seeding_operation(result)
            return result
    
    def get_seeding_history(self) -> List[Dict[str, Any]]:
        """Get history of seeding operations."""
        try:
            conn = self.get_conn()
            try:
                cursor = conn.execute("""
                    SELECT id, created_at, success, notes_created, embeddings_created, message
                    FROM seeding_log 
                    ORDER BY created_at DESC
                """)
                
                rows = cursor.fetchall()
                history = []
                
                for row in rows:
                    try:
                        history.append({
                            "id": row[0],
                            "created_at": row[1],
                            "success": bool(row[2]),
                            "notes_created": row[3],
                            "embeddings_created": row[4],
                            "message": row[5]
                        })
                    except (IndexError, TypeError):
                        # Skip malformed rows
                        continue
                
                return history
                
            except sqlite3.OperationalError:
                # Table doesn't exist
                return []
            finally:
                conn.close()
                
        except Exception as e:
            log.error("Error getting seeding history: %s", e)
            return []
    
    def get_seeding_stats(self) -> Dict[str, Any]:
        """Get statistics about seeding operations."""
        try:
            conn = self.get_conn()
            try:
                # Get aggregate statistics
                cursor = conn.execute("SELECT COUNT(*) FROM seeding_log")
                total_operations = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM seeding_log WHERE success = 1")
                successful_operations = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COALESCE(SUM(notes_created), 0) FROM seeding_log WHERE success = 1")
                total_notes_created = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COALESCE(SUM(embeddings_created), 0) FROM seeding_log WHERE success = 1")
                total_embeddings_created = cursor.fetchone()[0]
                
                success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0.0
                
                return {
                    "total_operations": total_operations,
                    "successful_operations": successful_operations,
                    "success_rate": success_rate,
                    "total_notes_created": total_notes_created,
                    "total_embeddings_created": total_embeddings_created
                }
                
            except sqlite3.OperationalError:
                # Table doesn't exist or query failed
                return {
                    "total_operations": 0,
                    "successful_operations": 0,
                    "success_rate": 0.0,
                    "total_notes_created": 0,
                    "total_embeddings_created": 0
                }
            finally:
                conn.close()
                
        except Exception as e:
            log.error("Error getting seeding stats: %s", e)
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "success_rate": 0.0,
                "total_notes_created": 0,
                "total_embeddings_created": 0
            }
    
    def _count_seeded_notes(self, namespace: str = settings.auto_seeding_namespace) -> int:
        """Count notes that were created by seeding."""
        try:
            conn = self.get_conn()
            try:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM notes
                    WHERE (
                        tags LIKE '%seed%'
                        OR tags LIKE ?
                        OR content LIKE ?
                        OR title LIKE ?
                    )
                    """,
                    (f"%ns:{namespace}%", f"%{namespace}%", f"%{namespace}%"),
                )
                
                return cursor.fetchone()[0]
            finally:
                conn.close()
        except Exception as e:
            log.error("Error counting seeded notes: %s", e)
            return 0
    
    def _count_seeded_embeddings(self, namespace: str = settings.auto_seeding_namespace) -> int:
        """Count embeddings that were created by seeding (JSON fallback + sqlite-vec note_vecs)."""
        try:
            conn = self.get_conn()
            try:
                total = 0
                # JSON fallback table
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
                )
                if cursor.fetchone():
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM embeddings e
                        JOIN notes n ON e.note_id = n.id
                        WHERE (
                            n.tags LIKE '%seed%'
                            OR n.tags LIKE ?
                            OR n.content LIKE ?
                            OR n.title LIKE ?
                        )
                        """,
                        (f"%ns:{namespace}%", f"%{namespace}%", f"%{namespace}%"),
                    )
                    total += cursor.fetchone()[0]

                # sqlite-vec table used by SearchService
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'"
                )
                if cursor.fetchone():
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM note_vecs v
                        JOIN notes n ON v.note_id = n.id
                        WHERE (
                            n.tags LIKE '%seed%'
                            OR n.tags LIKE ?
                            OR n.content LIKE ?
                            OR n.title LIKE ?
                        )
                        """,
                        (f"%ns:{namespace}%", f"%{namespace}%", f"%{namespace}%"),
                    )
                    total += cursor.fetchone()[0]

                return total
            finally:
                conn.close()
        except Exception as e:
            log.error("Error counting seeded embeddings: %s", e)
            return 0
    
    def _log_seeding_operation(self, result: SeedingResult):
        """Log a seeding operation to the database."""
        try:
            conn = self.get_conn()
            try:
                # Ensure seeding_log table exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS seeding_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        notes_created INTEGER DEFAULT 0,
                        embeddings_created INTEGER DEFAULT 0,
                        message TEXT,
                        error TEXT
                    )
                """)
                
                # Format message with error if applicable
                message = result.message
                if result.error:
                    message = f"{message}: {result.error}"
                
                # Insert log entry
                conn.execute("""
                    INSERT INTO seeding_log 
                    (created_at, success, notes_created, embeddings_created, message)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    result.success,
                    result.notes_created,
                    result.embeddings_created,
                    message
                ))
                
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            log.error("Error logging seeding operation: %s", e)

    def _refresh_search_indexes(self) -> None:
        """Best-effort refresh of keyword search index if present.
        Rebuilds notes_fts from notes table to ensure seeded content is searchable immediately.
        """
        conn = self.get_conn()
        try:
            cur = conn.cursor()
            # Check for FTS over notes
            row = cur.execute(
                "SELECT name FROM sqlite_master WHERE name='notes_fts'"
            ).fetchone()
            if not row:
                return
            # Clear existing
            cur.execute("DELETE FROM notes_fts")
            # Try SearchService schema (title, body, tags)
            try:
                cur.execute(
                    """
                    INSERT INTO notes_fts(rowid, title, body, tags)
                    SELECT id, title, body, tags FROM notes
                    """
                )
            except Exception:
                # Try application schema (title, content, tags)
                cur.execute(
                    """
                    INSERT INTO notes_fts(rowid, title, body, tags)
                    SELECT id, title, content, tags FROM notes
                    """
                )
            conn.commit()
            log.info("notes_fts rebuilt after seeding")
        finally:
            conn.close()

# Global instance - initialized when first imported
_seeding_service = None

def get_seeding_service(get_conn_func):
    """Get global seeding service instance."""
    global _seeding_service
    if _seeding_service is None:
        _seeding_service = VaultSeedingService(get_conn_func)
    return _seeding_service

# Export the service class and functions
__all__ = ["VaultSeedingService", "SeedingResult", "SeedingOptions", "get_seeding_service"]
