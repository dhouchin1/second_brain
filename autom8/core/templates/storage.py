"""
Template Storage System

This module provides persistent storage capabilities for templates, template libraries,
and template metadata using SQLite with JSON serialization for complex data types.
"""

import json
import sqlite3
import asyncio
import aiosqlite
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Set, Tuple
from pathlib import Path
import uuid

from autom8.models.template import (
    ContextTemplate,
    TemplateType,
    TemplateStatus,
    TemplateLibrary,
    TemplateAnalytics,
    TemplateVariable,
    TemplateSource,
    TemplateMetadata,
)
from autom8.config.settings import get_settings
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateStorageEngine:
    """
    Core storage engine for templates using SQLite with async support.
    """
    
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self._connection_pool = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the storage engine and create tables."""
        if self._initialized:
            return True
        
        try:
            # Ensure database directory exists
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create database and tables
            await self._create_tables()
            
            self._initialized = True
            logger.info(f"Template storage initialized: {self.database_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize template storage: {e}")
            return False
    
    async def _create_tables(self):
        """Create database tables for template storage."""
        async with aiosqlite.connect(self.database_path) as db:
            # Templates table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    template_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    category TEXT,
                    tags TEXT,  -- JSON array
                    version TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    created_by TEXT,
                    variables TEXT,  -- JSON array
                    sources TEXT,    -- JSON array
                    composition TEXT, -- JSON object
                    validation TEXT,  -- JSON object
                    metadata TEXT,    -- JSON object (full metadata)
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP
                )
            """)
            
            # Template libraries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS template_libraries (
                    library_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    author TEXT,
                    version TEXT,
                    license TEXT,
                    homepage TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            # Library membership table (many-to-many)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS library_templates (
                    library_id TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (library_id, template_id),
                    FOREIGN KEY (library_id) REFERENCES template_libraries(library_id),
                    FOREIGN KEY (template_id) REFERENCES templates(template_id)
                )
            """)
            
            # Template analytics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS template_analytics (
                    template_id TEXT PRIMARY KEY,
                    total_executions INTEGER DEFAULT 0,
                    successful_executions INTEGER DEFAULT 0,
                    failed_executions INTEGER DEFAULT 0,
                    avg_render_time_ms REAL DEFAULT 0.0,
                    avg_token_count REAL DEFAULT 0.0,
                    avg_source_count REAL DEFAULT 0.0,
                    most_common_variables TEXT,  -- JSON object
                    execution_frequency TEXT,    -- JSON object
                    user_adoption TEXT,          -- JSON object
                    avg_quality_score REAL DEFAULT 0.0,
                    validation_error_rate REAL DEFAULT 0.0,
                    first_used TIMESTAMP,
                    last_used TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (template_id) REFERENCES templates(template_id)
                )
            """)
            
            # Create indexes for better query performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_templates_type ON templates(type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_templates_status ON templates(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_templates_created_by ON templates(created_by)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_templates_usage_count ON templates(usage_count)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_templates_last_used ON templates(last_used)")
            
            await db.commit()
            logger.debug("Database tables created successfully")
    
    async def store_template(self, template: ContextTemplate) -> bool:
        """Store or update a template."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                # Convert template to database row
                row_data = self._template_to_row(template)
                
                # Insert or replace template
                await db.execute("""
                    INSERT OR REPLACE INTO templates (
                        template_id, type, status, title, description, author,
                        category, tags, version, created_at, updated_at, created_by,
                        variables, sources, composition, validation, metadata,
                        usage_count, last_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row_data)
                
                await db.commit()
                logger.debug(f"Stored template: {template.template_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store template {template.template_id}: {e}")
            return False
    
    async def get_template(self, template_id: str) -> Optional[ContextTemplate]:
        """Retrieve a template by ID."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(
                    "SELECT * FROM templates WHERE template_id = ?",
                    (template_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return self._row_to_template(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get template {template_id}: {e}")
            return None
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                # Remove from libraries first
                await db.execute(
                    "DELETE FROM library_templates WHERE template_id = ?",
                    (template_id,)
                )
                
                # Delete analytics
                await db.execute(
                    "DELETE FROM template_analytics WHERE template_id = ?",
                    (template_id,)
                )
                
                # Delete template
                cursor = await db.execute(
                    "DELETE FROM templates WHERE template_id = ?",
                    (template_id,)
                )
                
                deleted = cursor.rowcount > 0
                await db.commit()
                
                if deleted:
                    logger.debug(f"Deleted template: {template_id}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False
    
    async def list_templates(
        self,
        template_type: Optional[TemplateType] = None,
        status: Optional[TemplateStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ContextTemplate]:
        """List templates with filtering and pagination."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Build query
                conditions = []
                params = []
                
                if template_type:
                    conditions.append("type = ?")
                    params.append(template_type.value)
                
                if status:
                    conditions.append("status = ?")
                    params.append(status.value)
                
                if created_by:
                    conditions.append("created_by = ?")
                    params.append(created_by)
                
                if category:
                    conditions.append("category = ?")
                    params.append(category)
                
                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)
                
                query = f"""
                    SELECT * FROM templates 
                    {where_clause}
                    ORDER BY updated_at DESC 
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                templates = []
                for row in rows:
                    template = self._row_to_template(row)
                    
                    # Filter by tags if specified
                    if tags and not any(tag in template.metadata.tags for tag in tags):
                        continue
                    
                    templates.append(template)
                
                return templates
                
        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return []
    
    async def search_templates(
        self,
        query: str,
        template_type: Optional[TemplateType] = None,
        limit: int = 20
    ) -> List[ContextTemplate]:
        """Search templates by content."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Build search query
                search_conditions = [
                    "title LIKE ? OR description LIKE ? OR tags LIKE ?"
                ]
                search_params = [f"%{query}%", f"%{query}%", f"%{query}%"]
                
                if template_type:
                    search_conditions.append("type = ?")
                    search_params.append(template_type.value)
                
                where_clause = "WHERE " + " AND ".join(search_conditions)
                
                sql = f"""
                    SELECT * FROM templates 
                    {where_clause}
                    ORDER BY usage_count DESC, updated_at DESC 
                    LIMIT ?
                """
                search_params.append(limit)
                
                cursor = await db.execute(sql, search_params)
                rows = await cursor.fetchall()
                
                return [self._row_to_template(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search templates: {e}")
            return []
    
    async def update_usage_stats(
        self,
        template_id: str,
        increment_count: bool = True
    ) -> bool:
        """Update template usage statistics."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                if increment_count:
                    await db.execute("""
                        UPDATE templates 
                        SET usage_count = usage_count + 1, 
                            last_used = CURRENT_TIMESTAMP 
                        WHERE template_id = ?
                    """, (template_id,))
                else:
                    await db.execute("""
                        UPDATE templates 
                        SET last_used = CURRENT_TIMESTAMP 
                        WHERE template_id = ?
                    """, (template_id,))
                
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update usage stats for {template_id}: {e}")
            return False
    
    # Library management methods
    
    async def store_library(self, library: TemplateLibrary) -> bool:
        """Store or update a template library."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO template_libraries (
                        library_id, name, description, author, version,
                        license, homepage, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    library.library_id,
                    library.name,
                    library.description,
                    library.author,
                    library.version,
                    library.license,
                    library.homepage,
                    library.created_at,
                    library.updated_at
                ))
                
                await db.commit()
                logger.debug(f"Stored library: {library.library_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store library {library.library_id}: {e}")
            return False
    
    async def add_template_to_library(
        self,
        library_id: str,
        template_id: str
    ) -> bool:
        """Add a template to a library."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    INSERT OR IGNORE INTO library_templates 
                    (library_id, template_id, added_at) 
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (library_id, template_id))
                
                await db.commit()
                logger.debug(f"Added template {template_id} to library {library_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add template to library: {e}")
            return False
    
    # Analytics methods
    
    async def store_template_analytics(self, analytics: TemplateAnalytics) -> bool:
        """Store template analytics."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO template_analytics (
                        template_id, total_executions, successful_executions,
                        failed_executions, avg_render_time_ms, avg_token_count,
                        avg_source_count, most_common_variables, execution_frequency,
                        user_adoption, avg_quality_score, validation_error_rate,
                        first_used, last_used, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analytics.template_id,
                    analytics.total_executions,
                    analytics.successful_executions,
                    analytics.failed_executions,
                    analytics.avg_render_time_ms,
                    analytics.avg_token_count,
                    analytics.avg_source_count,
                    json.dumps(analytics.most_common_variables),
                    json.dumps(analytics.execution_frequency),
                    json.dumps(analytics.user_adoption),
                    analytics.avg_quality_score,
                    analytics.validation_error_rate,
                    analytics.first_used,
                    analytics.last_used,
                    analytics.updated_at
                ))
                
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store analytics for {analytics.template_id}: {e}")
            return False
    
    async def get_template_analytics(self, template_id: str) -> Optional[TemplateAnalytics]:
        """Get template analytics."""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute(
                    "SELECT * FROM template_analytics WHERE template_id = ?",
                    (template_id,)
                )
                row = await cursor.fetchone()
                
                if row:
                    return TemplateAnalytics(
                        template_id=row["template_id"],
                        total_executions=row["total_executions"],
                        successful_executions=row["successful_executions"],
                        failed_executions=row["failed_executions"],
                        avg_render_time_ms=row["avg_render_time_ms"],
                        avg_token_count=row["avg_token_count"],
                        avg_source_count=row["avg_source_count"],
                        most_common_variables=json.loads(row["most_common_variables"] or "{}"),
                        execution_frequency=json.loads(row["execution_frequency"] or "{}"),
                        user_adoption=json.loads(row["user_adoption"] or "{}"),
                        avg_quality_score=row["avg_quality_score"],
                        validation_error_rate=row["validation_error_rate"],
                        first_used=row["first_used"],
                        last_used=row["last_used"],
                        updated_at=row["updated_at"]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get analytics for {template_id}: {e}")
            return None
    
    # Helper methods for data conversion
    
    def _template_to_row(self, template: ContextTemplate) -> tuple:
        """Convert template object to database row."""
        return (
            template.template_id,
            template.type.value,
            template.status.value,
            template.metadata.title,
            template.metadata.description,
            template.metadata.author,
            template.metadata.category,
            json.dumps(template.metadata.tags),
            template.metadata.version,
            template.created_at,
            template.updated_at,
            template.created_by,
            json.dumps([var.dict() for var in template.variables]),
            json.dumps([source.dict() for source in template.sources]),
            json.dumps(template.composition.dict() if template.composition else None),
            json.dumps(template.validation.dict() if template.validation else None),
            json.dumps(template.metadata.dict()),
            template.metadata.usage_count,
            template.metadata.last_used
        )
    
    def _row_to_template(self, row) -> ContextTemplate:
        """Convert database row to template object."""
        # Parse JSON fields
        variables_data = json.loads(row["variables"] or "[]")
        sources_data = json.loads(row["sources"] or "[]")
        composition_data = json.loads(row["composition"] or "null")
        validation_data = json.loads(row["validation"] or "null")
        metadata_data = json.loads(row["metadata"])
        
        # Rebuild objects
        variables = [TemplateVariable(**var_data) for var_data in variables_data]
        sources = [TemplateSource(**source_data) for source_data in sources_data]
        
        from autom8.models.template import TemplateComposition, TemplateValidation
        composition = TemplateComposition(**composition_data) if composition_data else None
        validation = TemplateValidation(**validation_data) if validation_data else None
        metadata = TemplateMetadata(**metadata_data)
        
        return ContextTemplate(
            template_id=row["template_id"],
            type=TemplateType(row["type"]),
            status=TemplateStatus(row["status"]),
            metadata=metadata,
            variables=variables,
            sources=sources,
            composition=composition,
            validation=validation,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"]
        )


class TemplateStorage:
    """
    Main template storage interface providing high-level storage operations.
    """
    
    def __init__(self, database_path: Optional[str] = None):
        if database_path is None:
            settings = get_settings()
            database_path = settings.get_data_path() / "templates" / "templates.db"
        
        self.storage_engine = TemplateStorageEngine(str(database_path))
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize template storage."""
        if self._initialized:
            return True
        
        try:
            success = await self.storage_engine.initialize()
            if success:
                self._initialized = True
                logger.info("Template storage initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize template storage: {e}")
            return False
    
    # Delegate methods to storage engine
    
    async def store_template(self, template: ContextTemplate) -> bool:
        """Store a template."""
        return await self.storage_engine.store_template(template)
    
    async def get_template(self, template_id: str) -> Optional[ContextTemplate]:
        """Get a template by ID."""
        template = await self.storage_engine.get_template(template_id)
        if template:
            # Update last accessed time
            await self.storage_engine.update_usage_stats(template_id, increment_count=False)
        return template
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        return await self.storage_engine.delete_template(template_id)
    
    async def list_templates(
        self,
        template_type: Optional[TemplateType] = None,
        status: Optional[TemplateStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ContextTemplate]:
        """List templates with filtering."""
        return await self.storage_engine.list_templates(
            template_type, status, created_by, tags, None, limit, offset
        )
    
    async def search_templates(
        self,
        query: str,
        template_type: Optional[TemplateType] = None,
        limit: int = 20
    ) -> List[ContextTemplate]:
        """Search templates."""
        return await self.storage_engine.search_templates(query, template_type, limit)
    
    async def store_library(self, library: TemplateLibrary) -> bool:
        """Store a template library."""
        return await self.storage_engine.store_library(library)
    
    async def add_template_to_library(self, library_id: str, template_id: str) -> bool:
        """Add template to library."""
        return await self.storage_engine.add_template_to_library(library_id, template_id)
    
    async def record_template_execution(self, template_id: str) -> bool:
        """Record template execution for usage tracking."""
        return await self.storage_engine.update_usage_stats(template_id, increment_count=True)
    
    async def get_popular_templates(self, limit: int = 10) -> List[ContextTemplate]:
        """Get most popular templates by usage."""
        try:
            return await self.storage_engine.list_templates(
                template_type=None,
                status=TemplateStatus.ACTIVE,
                limit=limit,
                offset=0
            )
        except Exception as e:
            logger.error(f"Failed to get popular templates: {e}")
            return []
    
    async def backup_templates(self, backup_path: str) -> bool:
        """Create a backup of all templates."""
        try:
            templates = await self.list_templates(limit=1000)  # Get all templates
            
            backup_data = {
                "backup_timestamp": datetime.utcnow().isoformat(),
                "template_count": len(templates),
                "templates": [template.dict() for template in templates]
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Backed up {len(templates)} templates to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup templates: {e}")
            return False
    
    async def restore_templates(self, backup_path: str, merge_strategy: str = "skip") -> bool:
        """Restore templates from backup."""
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            templates_data = backup_data.get("templates", [])
            restored_count = 0
            
            for template_data in templates_data:
                try:
                    template = ContextTemplate(**template_data)
                    
                    # Check if template already exists
                    existing = await self.get_template(template.template_id)
                    
                    if existing and merge_strategy == "skip":
                        continue
                    elif existing and merge_strategy == "error":
                        raise ValueError(f"Template already exists: {template.template_id}")
                    
                    # Store template
                    if await self.store_template(template):
                        restored_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to restore template: {e}")
                    continue
            
            logger.info(f"Restored {restored_count} templates from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore templates: {e}")
            return False
    
    async def cleanup_old_templates(self, days_old: int = 90) -> int:
        """Clean up old unused templates."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # This would need to be implemented in the storage engine
            # For now, return 0
            logger.info(f"Template cleanup not implemented yet")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup templates: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            all_templates = await self.list_templates(limit=1000)
            
            stats = {
                "total_templates": len(all_templates),
                "templates_by_type": {},
                "templates_by_status": {},
                "total_size_estimate": 0,
                "most_used_templates": [],
                "recent_templates": []
            }
            
            # Calculate statistics
            for template in all_templates:
                # Count by type
                type_key = template.type.value
                stats["templates_by_type"][type_key] = stats["templates_by_type"].get(type_key, 0) + 1
                
                # Count by status
                status_key = template.status.value
                stats["templates_by_status"][status_key] = stats["templates_by_status"].get(status_key, 0) + 1
                
                # Estimate size (rough)
                template_json = template.json()
                stats["total_size_estimate"] += len(template_json)
            
            # Get most used (top 5)
            sorted_by_usage = sorted(
                all_templates, 
                key=lambda t: t.metadata.usage_count, 
                reverse=True
            )
            stats["most_used_templates"] = [
                {"id": t.template_id, "title": t.metadata.title, "usage": t.metadata.usage_count}
                for t in sorted_by_usage[:5]
            ]
            
            # Get recent (top 5)
            sorted_by_date = sorted(
                all_templates,
                key=lambda t: t.updated_at,
                reverse=True
            )
            stats["recent_templates"] = [
                {"id": t.template_id, "title": t.metadata.title, "updated": t.updated_at.isoformat()}
                for t in sorted_by_date[:5]
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}


# Global instance
_template_storage = None


async def get_template_storage() -> TemplateStorage:
    """Get global template storage instance."""
    global _template_storage
    
    if _template_storage is None:
        _template_storage = TemplateStorage()
        await _template_storage.initialize()
    
    return _template_storage