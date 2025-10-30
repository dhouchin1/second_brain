from typing import List, Dict, Optional
import sqlite3
from datetime import datetime
import uuid
import logging
from services.embeddings import Embeddings

logger = logging.getLogger(__name__)

class MemoryService:
    """Manages episodic and semantic memories"""

    def __init__(self, db_connection, embeddings_service: Optional[Embeddings] = None):
        self.db = db_connection
        self.embeddings = embeddings_service
        logger.info("MemoryService initialized")

    # ========== Episodic Memory ==========

    def add_episodic_memory(
        self,
        user_id: int,
        content: str,
        summary: str = None,
        importance: float = 0.5,
        context: str = None
    ) -> str:
        """Store conversation episode"""
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO episodic_memories (user_id, episode_id, content, summary, importance, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, episode_id, content, summary, importance, context))
        self.db.commit()

        logger.debug(f"Added episodic memory {episode_id} for user {user_id}")

        # Generate and store embedding if available
        if self.embeddings and summary:
            try:
                embedding = self.embeddings.generate_embedding(summary)
                # Store embedding
                cursor.execute("""
                    INSERT INTO episodic_vectors (episode_id, embedding)
                    VALUES (?, ?)
                """, (episode_id, embedding))
                self.db.commit()
                logger.debug(f"Stored vector embedding for episode {episode_id}")
            except Exception as e:
                logger.warning(f"Could not store episode vector: {e}")

        return episode_id

    def search_episodic(
        self,
        user_id: int,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0
    ) -> List[Dict]:
        """Search episodic memories using FTS + optional vector"""
        cursor = self.db.cursor()

        try:
            # FTS search
            cursor.execute("""
                SELECT
                    e.episode_id,
                    e.content,
                    e.summary,
                    e.importance,
                    e.context,
                    e.created_at,
                    episodic_fts.rank
                FROM episodic_memories e
                JOIN episodic_fts ON e.id = episodic_fts.rowid
                WHERE episodic_fts MATCH ?
                    AND e.user_id = ?
                    AND e.importance >= ?
                ORDER BY episodic_fts.rank
                LIMIT ?
            """, (query, user_id, min_importance, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'episode_id': row[0],
                    'content': row[1],
                    'summary': row[2],
                    'importance': row[3],
                    'context': row[4],
                    'created_at': row[5],
                    'rank': row[6]
                })

            logger.debug(f"Found {len(results)} episodic memories for query")
            return results

        except Exception as e:
            logger.error(f"Episodic search failed: {e}")
            return []

    def get_recent_episodes(
        self,
        user_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """Get most recent conversation episodes"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT episode_id, content, summary, importance, context, created_at
            FROM episodic_memories
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'episode_id': row[0],
                'content': row[1],
                'summary': row[2],
                'importance': row[3],
                'context': row[4],
                'created_at': row[5]
            })

        return results

    # ========== Semantic Memory ==========

    def add_semantic_memory(
        self,
        user_id: int,
        fact: str,
        category: str = 'general',
        confidence: float = 1.0,
        source: str = None
    ) -> str:
        """Store semantic fact about user"""
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO semantic_memories (user_id, fact_id, fact, category, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, fact_id, fact, category, confidence, source))
        self.db.commit()

        logger.debug(f"Added semantic memory {fact_id} for user {user_id}: {fact[:50]}")

        # Generate and store embedding if available
        if self.embeddings:
            try:
                embedding = self.embeddings.generate_embedding(fact)
                cursor.execute("""
                    INSERT INTO semantic_vectors (fact_id, embedding)
                    VALUES (?, ?)
                """, (fact_id, embedding))
                self.db.commit()
                logger.debug(f"Stored vector embedding for fact {fact_id}")
            except Exception as e:
                logger.warning(f"Could not store fact vector: {e}")

        return fact_id

    def search_semantic(
        self,
        user_id: int,
        query: str,
        limit: int = 10,
        category: str = None
    ) -> List[Dict]:
        """Search user facts"""
        cursor = self.db.cursor()

        try:
            if category:
                cursor.execute("""
                    SELECT
                        s.fact_id,
                        s.fact,
                        s.category,
                        s.confidence,
                        s.created_at,
                        semantic_fts.rank
                    FROM semantic_memories s
                    JOIN semantic_fts ON s.id = semantic_fts.rowid
                    WHERE semantic_fts MATCH ?
                        AND s.user_id = ?
                        AND s.category = ?
                    ORDER BY semantic_fts.rank
                    LIMIT ?
                """, (query, user_id, category, limit))
            else:
                cursor.execute("""
                    SELECT
                        s.fact_id,
                        s.fact,
                        s.category,
                        s.confidence,
                        s.created_at,
                        semantic_fts.rank
                    FROM semantic_memories s
                    JOIN semantic_fts ON s.id = semantic_fts.rowid
                    WHERE semantic_fts MATCH ? AND s.user_id = ?
                    ORDER BY semantic_fts.rank
                    LIMIT ?
                """, (query, user_id, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'fact_id': row[0],
                    'fact': row[1],
                    'category': row[2],
                    'confidence': row[3],
                    'created_at': row[4],
                    'rank': row[5]
                })

            logger.debug(f"Found {len(results)} semantic memories for query")
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def get_all_user_facts(self, user_id: int, category: str = None) -> List[Dict]:
        """Get all facts about a user"""
        cursor = self.db.cursor()

        if category:
            cursor.execute("""
                SELECT fact_id, fact, category, confidence, created_at
                FROM semantic_memories
                WHERE user_id = ? AND category = ?
                ORDER BY confidence DESC, created_at DESC
            """, (user_id, category))
        else:
            cursor.execute("""
                SELECT fact_id, fact, category, confidence, created_at
                FROM semantic_memories
                WHERE user_id = ?
                ORDER BY confidence DESC, created_at DESC
            """, (user_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'fact_id': row[0],
                'fact': row[1],
                'category': row[2],
                'confidence': row[3],
                'created_at': row[4]
            })

        return results

    def update_semantic_memory(
        self,
        fact_id: str,
        fact: str = None,
        confidence: float = None
    ):
        """Update existing semantic memory"""
        cursor = self.db.cursor()

        updates = []
        params = []

        if fact is not None:
            updates.append("fact = ?")
            params.append(fact)

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)

        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(fact_id)

            query = f"UPDATE semantic_memories SET {', '.join(updates)} WHERE fact_id = ?"
            cursor.execute(query, params)
            self.db.commit()
            logger.debug(f"Updated semantic memory {fact_id}")

    def delete_semantic_memory(self, fact_id: str):
        """Delete a semantic memory"""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM semantic_memories WHERE fact_id = ?", (fact_id,))
        self.db.commit()
        logger.debug(f"Deleted semantic memory {fact_id}")

    # ========== Conversation Sessions ==========

    def start_conversation(self, user_id: int) -> str:
        """Start new conversation session"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"

        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO conversation_sessions (user_id, session_id)
            VALUES (?, ?)
        """, (user_id, session_id))
        self.db.commit()

        logger.info(f"Started conversation session {session_id} for user {user_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ):
        """Add message to conversation"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT INTO conversation_messages (session_id, role, content)
            VALUES (?, ?, ?)
        """, (session_id, role, content))

        # Update session metadata
        cursor.execute("""
            UPDATE conversation_sessions
            SET message_count = message_count + 1,
                last_activity = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))

        self.db.commit()
        logger.debug(f"Added {role} message to session {session_id}")

    def get_conversation(self, session_id: str) -> List[Dict]:
        """Get all messages in a conversation"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT role, content, created_at
            FROM conversation_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'content': row[1],
                'created_at': row[2]
            })

        return messages

    def end_conversation(self, session_id: str):
        """Mark conversation as ended"""
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE conversation_sessions
            SET ended_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        self.db.commit()
        logger.info(f"Ended conversation session {session_id}")
