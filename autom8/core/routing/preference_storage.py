"""
Preference Storage and Persistence System

This module handles storing, retrieving, and managing learned user preferences
and behavioral patterns for the adaptive routing system.
"""

import asyncio
import json
import pickle
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import asdict
import aiofiles
from pathlib import Path
import uuid

from autom8.core.routing.preference_learning import (
    UserProfile, LearningPattern, UserInteraction
)
from autom8.utils.logging import get_logger

logger = get_logger(__name__)


class PreferenceStorageBackend:
    """Abstract base class for preference storage backends"""
    
    async def store_user_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Store user profile"""
        raise NotImplementedError
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile"""
        raise NotImplementedError
    
    async def store_learning_pattern(self, pattern_id: str, pattern: LearningPattern) -> bool:
        """Store learning pattern"""
        raise NotImplementedError
    
    async def get_learning_patterns(self, user_id: Optional[str] = None) -> List[LearningPattern]:
        """Retrieve learning patterns"""
        raise NotImplementedError
    
    async def store_interaction(self, interaction: UserInteraction) -> bool:
        """Store user interaction"""
        raise NotImplementedError
    
    async def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Retrieve user interactions"""
        raise NotImplementedError
    
    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """Clean up old data, return number of records cleaned"""
        raise NotImplementedError


class SQLiteStorageBackend(PreferenceStorageBackend):
    """SQLite-based storage backend for preferences"""
    
    def __init__(self, db_path: str = "autom8_preferences.db"):
        self.db_path = db_path
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the SQLite database"""
        if self._initialized:
            return True
        
        try:
            # Create database and tables
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            
            # Learning patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    pattern_data TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    support INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL
                )
            """)
            
            # User interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    interaction_data TEXT NOT NULL,
                    outcome_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    model_name TEXT
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_user_id ON learning_patterns(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON learning_patterns(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON user_interactions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON user_interactions(timestamp)")
            
            conn.commit()
            conn.close()
            
            self._initialized = True
            logger.info(f"SQLite preference storage initialized at {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {e}")
            return False
    
    async def store_user_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Store user profile in SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profile_json = json.dumps(asdict(profile), default=str)
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, profile_data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, profile_json, profile.created_at, profile.last_updated))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user profile for {user_id}: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT profile_data FROM user_profiles WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                profile_data = json.loads(result[0])
                # Convert datetime strings back to datetime objects
                profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
                return UserProfile(**profile_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            return None
    
    async def store_learning_pattern(self, pattern_id: str, pattern: LearningPattern) -> bool:
        """Store learning pattern in SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            pattern_json = json.dumps(asdict(pattern), default=str)
            user_id = pattern_id.split('_')[0] if '_' in pattern_id else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO learning_patterns
                (pattern_id, user_id, pattern_data, pattern_type, confidence, support, created_at, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pattern_id, user_id, pattern_json, pattern.pattern_type, 
                  pattern.confidence, pattern.support, pattern.created_at, pattern.last_seen))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store learning pattern {pattern_id}: {e}")
            return False
    
    async def get_learning_patterns(self, user_id: Optional[str] = None) -> List[LearningPattern]:
        """Retrieve learning patterns from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT pattern_data FROM learning_patterns 
                    WHERE user_id = ? ORDER BY confidence DESC, last_seen DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT pattern_data FROM learning_patterns 
                    ORDER BY confidence DESC, last_seen DESC
                """)
            
            results = cursor.fetchall()
            conn.close()
            
            patterns = []
            for result in results:
                pattern_data = json.loads(result[0])
                # Convert datetime strings back to datetime objects
                pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
                pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
                patterns.append(LearningPattern(**pattern_data))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get learning patterns for user {user_id}: {e}")
            return []
    
    async def store_interaction(self, interaction: UserInteraction) -> bool:
        """Store user interaction in SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            interaction_id = str(uuid.uuid4())
            interaction_json = json.dumps(asdict(interaction), default=str)
            
            cursor.execute("""
                INSERT INTO user_interactions
                (interaction_id, user_id, session_id, interaction_data, outcome_type, timestamp, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (interaction_id, interaction.user_id, interaction.session_id,
                  interaction_json, interaction.outcome_type.value, 
                  interaction.timestamp, interaction.selected_model))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store interaction for user {interaction.user_id}: {e}")
            return False
    
    async def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Retrieve user interactions from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT interaction_data FROM user_interactions 
                WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
            """, (user_id, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            interactions = []
            for result in results:
                interaction_data = json.loads(result[0])
                # Convert datetime strings and enums back to proper objects
                interaction_data['timestamp'] = datetime.fromisoformat(interaction_data['timestamp'])
                
                # Convert enum strings back to enums
                from autom8.core.routing.preference_learning import OutcomeType, FeedbackType
                interaction_data['outcome_type'] = OutcomeType(interaction_data['outcome_type'])
                if interaction_data.get('feedback_type'):
                    interaction_data['feedback_type'] = FeedbackType(interaction_data['feedback_type'])
                
                interactions.append(UserInteraction(**interaction_data))
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to get interactions for user {user_id}: {e}")
            return []
    
    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """Clean up old data from SQLite"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean up old interactions
            cursor.execute("""
                DELETE FROM user_interactions WHERE timestamp < ?
            """, (cutoff_date,))
            interactions_deleted = cursor.rowcount
            
            # Clean up old patterns (but keep if recently seen)
            cursor.execute("""
                DELETE FROM learning_patterns WHERE created_at < ? AND last_seen < ?
            """, (cutoff_date, cutoff_date))
            patterns_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            total_deleted = interactions_deleted + patterns_deleted
            logger.info(f"Cleaned up {total_deleted} old records ({interactions_deleted} interactions, {patterns_deleted} patterns)")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0


class FileStorageBackend(PreferenceStorageBackend):
    """File-based storage backend for preferences"""
    
    def __init__(self, storage_dir: str = "autom8_preferences"):
        self.storage_dir = Path(storage_dir)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize file storage directories"""
        if self._initialized:
            return True
        
        try:
            # Create directories
            self.storage_dir.mkdir(exist_ok=True)
            (self.storage_dir / "profiles").mkdir(exist_ok=True)
            (self.storage_dir / "patterns").mkdir(exist_ok=True)
            (self.storage_dir / "interactions").mkdir(exist_ok=True)
            
            self._initialized = True
            logger.info(f"File preference storage initialized at {self.storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize file storage: {e}")
            return False
    
    async def store_user_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Store user profile as JSON file"""
        try:
            profile_path = self.storage_dir / "profiles" / f"{user_id}.json"
            profile_data = asdict(profile)
            
            async with aiofiles.open(profile_path, 'w') as f:
                await f.write(json.dumps(profile_data, default=str, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user profile for {user_id}: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from JSON file"""
        try:
            profile_path = self.storage_dir / "profiles" / f"{user_id}.json"
            
            if not profile_path.exists():
                return None
            
            async with aiofiles.open(profile_path, 'r') as f:
                profile_data = json.loads(await f.read())
            
            # Convert datetime strings back to datetime objects
            profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
            profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
            
            return UserProfile(**profile_data)
            
        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            return None
    
    async def store_learning_pattern(self, pattern_id: str, pattern: LearningPattern) -> bool:
        """Store learning pattern as JSON file"""
        try:
            pattern_path = self.storage_dir / "patterns" / f"{pattern_id}.json"
            pattern_data = asdict(pattern)
            
            async with aiofiles.open(pattern_path, 'w') as f:
                await f.write(json.dumps(pattern_data, default=str, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store learning pattern {pattern_id}: {e}")
            return False
    
    async def get_learning_patterns(self, user_id: Optional[str] = None) -> List[LearningPattern]:
        """Retrieve learning patterns from JSON files"""
        try:
            patterns_dir = self.storage_dir / "patterns"
            patterns = []
            
            if not patterns_dir.exists():
                return patterns
            
            for pattern_file in patterns_dir.glob("*.json"):
                # Filter by user_id if specified
                if user_id and not pattern_file.stem.startswith(f"{user_id}_"):
                    continue
                
                try:
                    async with aiofiles.open(pattern_file, 'r') as f:
                        pattern_data = json.loads(await f.read())
                    
                    # Convert datetime strings back to datetime objects
                    pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
                    pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
                    
                    patterns.append(LearningPattern(**pattern_data))
                    
                except Exception as e:
                    logger.warning(f"Failed to load pattern file {pattern_file}: {e}")
                    continue
            
            # Sort by confidence and recency
            patterns.sort(key=lambda p: (p.confidence, p.last_seen), reverse=True)
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get learning patterns for user {user_id}: {e}")
            return []
    
    async def store_interaction(self, interaction: UserInteraction) -> bool:
        """Store user interaction as JSON file"""
        try:
            # Create user-specific interaction directory
            user_dir = self.storage_dir / "interactions" / interaction.user_id
            user_dir.mkdir(exist_ok=True)
            
            # Generate interaction filename with timestamp
            timestamp_str = interaction.timestamp.strftime("%Y%m%d_%H%M%S")
            interaction_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp_str}_{interaction_id}.json"
            
            interaction_path = user_dir / filename
            interaction_data = asdict(interaction)
            
            async with aiofiles.open(interaction_path, 'w') as f:
                await f.write(json.dumps(interaction_data, default=str, indent=2))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store interaction for user {interaction.user_id}: {e}")
            return False
    
    async def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Retrieve user interactions from JSON files"""
        try:
            user_dir = self.storage_dir / "interactions" / user_id
            interactions = []
            
            if not user_dir.exists():
                return interactions
            
            # Get all interaction files sorted by timestamp (newest first)
            interaction_files = sorted(
                user_dir.glob("*.json"), 
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            for interaction_file in interaction_files[:limit]:
                try:
                    async with aiofiles.open(interaction_file, 'r') as f:
                        interaction_data = json.loads(await f.read())
                    
                    # Convert datetime strings and enums back to proper objects
                    interaction_data['timestamp'] = datetime.fromisoformat(interaction_data['timestamp'])
                    
                    from autom8.core.routing.preference_learning import OutcomeType, FeedbackType
                    interaction_data['outcome_type'] = OutcomeType(interaction_data['outcome_type'])
                    if interaction_data.get('feedback_type'):
                        interaction_data['feedback_type'] = FeedbackType(interaction_data['feedback_type'])
                    
                    interactions.append(UserInteraction(**interaction_data))
                    
                except Exception as e:
                    logger.warning(f"Failed to load interaction file {interaction_file}: {e}")
                    continue
            
            return interactions
            
        except Exception as e:
            logger.error(f"Failed to get interactions for user {user_id}: {e}")
            return []
    
    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """Clean up old data files"""
        try:
            cutoff_time = (datetime.utcnow() - timedelta(days=retention_days)).timestamp()
            deleted_count = 0
            
            # Clean up old interaction files
            interactions_dir = self.storage_dir / "interactions"
            if interactions_dir.exists():
                for user_dir in interactions_dir.iterdir():
                    if user_dir.is_dir():
                        for interaction_file in user_dir.glob("*.json"):
                            if interaction_file.stat().st_mtime < cutoff_time:
                                interaction_file.unlink()
                                deleted_count += 1
            
            # Clean up old pattern files
            patterns_dir = self.storage_dir / "patterns"
            if patterns_dir.exists():
                for pattern_file in patterns_dir.glob("*.json"):
                    if pattern_file.stat().st_mtime < cutoff_time:
                        # Check if pattern was recently seen before deleting
                        try:
                            async with aiofiles.open(pattern_file, 'r') as f:
                                pattern_data = json.loads(await f.read())
                            
                            last_seen = datetime.fromisoformat(pattern_data['last_seen'])
                            if last_seen < datetime.utcnow() - timedelta(days=retention_days):
                                pattern_file.unlink()
                                deleted_count += 1
                                
                        except Exception:
                            # If we can't read it, delete it
                            pattern_file.unlink()
                            deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0


class PreferenceStorageManager:
    """Manager for preference storage operations"""
    
    def __init__(self, backend: Optional[PreferenceStorageBackend] = None):
        self.backend = backend or SQLiteStorageBackend()
        self._initialized = False
        
        # Cache for frequently accessed data
        self._profile_cache = {}
        self._pattern_cache = {}
        self._cache_ttl = timedelta(minutes=30)
        self._last_cache_clear = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """Initialize the storage manager"""
        if self._initialized:
            return True
        
        success = await self.backend.initialize()
        if success:
            self._initialized = True
            logger.info("Preference storage manager initialized")
        
        return success
    
    async def store_user_profile(self, user_id: str, profile: UserProfile) -> bool:
        """Store user profile with caching"""
        success = await self.backend.store_user_profile(user_id, profile)
        if success:
            # Update cache
            self._profile_cache[user_id] = (profile, datetime.utcnow())
        return success
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with caching"""
        # Check cache first
        if user_id in self._profile_cache:
            profile, cached_at = self._profile_cache[user_id]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return profile
        
        # Load from backend
        profile = await self.backend.get_user_profile(user_id)
        if profile:
            self._profile_cache[user_id] = (profile, datetime.utcnow())
        
        return profile
    
    async def store_learning_pattern(self, pattern_id: str, pattern: LearningPattern) -> bool:
        """Store learning pattern with caching"""
        success = await self.backend.store_learning_pattern(pattern_id, pattern)
        if success:
            # Update cache
            user_id = pattern_id.split('_')[0] if '_' in pattern_id else 'global'
            if user_id not in self._pattern_cache:
                self._pattern_cache[user_id] = {}
            self._pattern_cache[user_id][pattern_id] = (pattern, datetime.utcnow())
        
        return success
    
    async def get_learning_patterns(self, user_id: Optional[str] = None) -> List[LearningPattern]:
        """Get learning patterns with caching"""
        cache_key = user_id or 'global'
        
        # Check cache
        if cache_key in self._pattern_cache:
            cached_patterns = []
            for pattern_id, (pattern, cached_at) in self._pattern_cache[cache_key].items():
                if datetime.utcnow() - cached_at < self._cache_ttl:
                    cached_patterns.append(pattern)
            
            if cached_patterns:
                return cached_patterns
        
        # Load from backend
        patterns = await self.backend.get_learning_patterns(user_id)
        
        # Update cache
        if cache_key not in self._pattern_cache:
            self._pattern_cache[cache_key] = {}
        
        now = datetime.utcnow()
        for pattern in patterns:
            pattern_id = pattern.pattern_id
            self._pattern_cache[cache_key][pattern_id] = (pattern, now)
        
        return patterns
    
    async def store_interaction(self, interaction: UserInteraction) -> bool:
        """Store user interaction"""
        return await self.backend.store_interaction(interaction)
    
    async def get_interactions(self, user_id: str, limit: int = 100) -> List[UserInteraction]:
        """Get user interactions (no caching due to size)"""
        return await self.backend.get_interactions(user_id, limit)
    
    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user"""
        profile = await self.get_user_profile(user_id)
        patterns = await self.get_learning_patterns(user_id)
        interactions = await self.get_interactions(user_id, limit=1000)  # Get more for export
        
        return {
            "user_id": user_id,
            "profile": asdict(profile) if profile else None,
            "patterns": [asdict(p) for p in patterns],
            "interactions": [asdict(i) for i in interactions],
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
    
    async def import_user_data(self, user_data: Dict[str, Any]) -> bool:
        """Import user data"""
        try:
            user_id = user_data["user_id"]
            
            # Import profile
            if user_data.get("profile"):
                profile_data = user_data["profile"]
                profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                profile_data['last_updated'] = datetime.fromisoformat(profile_data['last_updated'])
                profile = UserProfile(**profile_data)
                await self.store_user_profile(user_id, profile)
            
            # Import patterns
            if user_data.get("patterns"):
                for pattern_data in user_data["patterns"]:
                    pattern_data['created_at'] = datetime.fromisoformat(pattern_data['created_at'])
                    pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
                    pattern = LearningPattern(**pattern_data)
                    await self.store_learning_pattern(pattern.pattern_id, pattern)
            
            # Import interactions
            if user_data.get("interactions"):
                from autom8.core.routing.preference_learning import OutcomeType, FeedbackType
                for interaction_data in user_data["interactions"]:
                    interaction_data['timestamp'] = datetime.fromisoformat(interaction_data['timestamp'])
                    interaction_data['outcome_type'] = OutcomeType(interaction_data['outcome_type'])
                    if interaction_data.get('feedback_type'):
                        interaction_data['feedback_type'] = FeedbackType(interaction_data['feedback_type'])
                    
                    interaction = UserInteraction(**interaction_data)
                    await self.store_interaction(interaction)
            
            logger.info(f"Successfully imported data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import user data: {e}")
            return False
    
    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """Clean up old data and clear cache"""
        deleted_count = await self.backend.cleanup_old_data(retention_days)
        
        # Clear cache
        self._profile_cache.clear()
        self._pattern_cache.clear()
        self._last_cache_clear = datetime.utcnow()
        
        return deleted_count
    
    def clear_cache(self):
        """Manually clear cache"""
        self._profile_cache.clear()
        self._pattern_cache.clear()
        self._last_cache_clear = datetime.utcnow()
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        try:
            # This would be backend-specific implementation
            stats = {
                "cache_size": len(self._profile_cache) + sum(len(patterns) for patterns in self._pattern_cache.values()),
                "last_cache_clear": self._last_cache_clear.isoformat(),
                "backend_type": type(self.backend).__name__
            }
            
            if isinstance(self.backend, SQLiteStorageBackend):
                # Get SQLite stats
                conn = sqlite3.connect(self.backend.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM user_profiles")
                stats["user_profiles_count"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM learning_patterns")
                stats["patterns_count"] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM user_interactions")
                stats["interactions_count"] = cursor.fetchone()[0]
                
                conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {"error": str(e)}


# Global instance
_preference_storage = None


async def get_preference_storage(backend: Optional[PreferenceStorageBackend] = None) -> PreferenceStorageManager:
    """Get global preference storage manager instance"""
    global _preference_storage
    
    if _preference_storage is None:
        _preference_storage = PreferenceStorageManager(backend)
        await _preference_storage.initialize()
    
    return _preference_storage