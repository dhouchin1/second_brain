# ──────────────────────────────────────────────────────────────────────────────
# File: services/demo_data_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Demo Data Service for Second Brain

Provides rich, varied demo/stock content to showcase search, AI features, and 
content relationships. Content can be filtered out via UI toggle.
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import random

from services.embeddings import Embeddings
from llm_utils import ollama_summarize, ollama_generate_title

logger = logging.getLogger(__name__)

@dataclass
class DemoContent:
    """Structure for demo content items."""
    title: str
    content: str
    content_type: str  # note, article, meeting, research, etc.
    tags: List[str]
    source_url: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    priority: int = 0  # 0=normal, 1=high priority content
    relationships: List[str] = None  # Related content IDs

class DemoDataService:
    """Service for managing demo/stock content."""
    
    def __init__(self, get_conn_func):
        """Initialize with database connection function."""
        self.get_conn = get_conn_func
        self.embedder = Embeddings()
    
    # Rich demo content sets
    DEMO_CONTENT_SETS = {
        
        "ai_ml_research": [
            DemoContent(
                title="The Future of Large Language Models",
                content="""
                Recent developments in large language models have shown remarkable progress in 
                natural language understanding and generation. Key breakthroughs include:
                
                - Emergent capabilities at scale
                - In-context learning without parameter updates
                - Multi-modal integration (text, images, code)
                - Reasoning and planning capabilities
                
                Challenges remain in areas like hallucination, factual accuracy, and computational 
                efficiency. The field is rapidly evolving with new architectures, training methods, 
                and applications emerging regularly.
                
                Important considerations for deployment include safety, alignment, and responsible 
                AI practices. Organizations must balance innovation with ethical considerations.
                """,
                content_type="research",
                tags=["ai", "machine-learning", "nlp", "llm", "research", "future-tech"],
                category="Technology",
                priority=1
            ),
            
            DemoContent(
                title="Vector Databases and Semantic Search",
                content="""
                Vector databases have become crucial infrastructure for AI applications, enabling 
                efficient similarity search over high-dimensional embeddings.
                
                Key capabilities:
                - Approximate Nearest Neighbor (ANN) search
                - Real-time embedding storage and retrieval
                - Hybrid search combining traditional and vector search
                - Scalability for billions of vectors
                
                Popular solutions include Pinecone, Weaviate, Chroma, and now sqlite-vec for 
                lightweight applications. The choice depends on scale, performance requirements, 
                and integration needs.
                
                Use cases span from recommendation systems to retrieval-augmented generation (RAG) 
                for language models.
                """,
                content_type="technical",
                tags=["vector-database", "embeddings", "search", "ai", "rag", "similarity-search"],
                category="Technology"
            )
        ],
        
        "productivity_pkm": [
            DemoContent(
                title="Building a Second Brain - Key Principles",
                content="""
                The Building a Second Brain (BASB) methodology by Tiago Forte emphasizes:
                
                1. **Capture** - Keep what resonates
                2. **Organize** - PARA method (Projects, Areas, Resources, Archive)  
                3. **Distill** - Progressive summarization
                4. **Express** - Share your work
                
                Core principles:
                - Focus on actionability over completeness
                - Create for your future self
                - External scaffolding for thinking
                - Just-in-time organization
                
                The goal is to create a trusted system that amplifies your thinking and helps 
                you make connections between ideas over time.
                """,
                content_type="methodology",
                tags=["pkm", "second-brain", "basb", "productivity", "knowledge-management", "para"],
                category="Productivity",
                priority=1
            ),
            
            DemoContent(
                title="Obsidian vs Notion vs Roam - PKM Tool Comparison",
                content="""
                Comparison of popular Personal Knowledge Management tools:
                
                **Obsidian**
                - Local files, markdown-based
                - Powerful graph view and backlinking
                - Extensive plugin ecosystem
                - Fast performance, works offline
                
                **Notion**
                - Database-driven, block-based
                - Great for structured data and collaboration
                - Templates and formulas
                - All-in-one workspace approach
                
                **Roam Research**  
                - Block-based with bi-directional linking
                - Daily notes and temporal organization
                - Query system for dynamic content
                - Research-oriented features
                
                Choice depends on workflow, technical comfort, and specific needs.
                """,
                content_type="comparison",
                tags=["obsidian", "notion", "roam", "pkm", "tools", "comparison", "knowledge-management"],
                category="Tools"
            )
        ],
        
        "meeting_notes": [
            DemoContent(
                title="Q4 Product Planning Meeting - Feature Roadmap",
                content="""
                **Date**: December 15, 2024
                **Attendees**: Sarah (PM), Mike (Eng), Lisa (Design), John (Data)
                
                **Key Decisions:**
                - Priority on mobile experience improvements
                - AI-powered search enhancement for Q1
                - User authentication system by February
                
                **Action Items:**
                - [ ] Sarah: Create detailed mobile UX specs by Dec 22
                - [ ] Mike: Research vector database options by Jan 5  
                - [ ] Lisa: Design system for search results by Jan 10
                - [ ] John: Set up analytics tracking by Dec 30
                
                **Technical Discussions:**
                - Database migration timeline
                - API rate limiting strategies
                - Performance optimization priorities
                
                **Next Meeting**: January 8, 2025 - Sprint planning
                """,
                content_type="meeting",
                tags=["meeting", "product-planning", "roadmap", "q4", "action-items", "team"],
                category="Work"
            ),
            
            DemoContent(
                title="Weekly Team Sync - Engineering Blockers",
                content="""
                **Date**: December 10, 2024
                **Team**: Engineering
                
                **Progress Updates:**
                - Authentication service: 80% complete
                - Search indexing: Performance issues identified
                - Mobile app: UI components ready for testing
                
                **Current Blockers:**
                - Database connection pooling causing timeouts
                - Third-party API rate limits affecting batch operations
                - Test environment instability
                
                **Solutions Discussed:**
                - Implement connection retry logic
                - Add caching layer to reduce API calls
                - Provision dedicated test infrastructure
                
                **This Week Goals:**
                - Resolve database connection issues
                - Complete search performance optimization
                - Begin integration testing phase
                """,
                content_type="meeting",
                tags=["meeting", "engineering", "sync", "blockers", "progress", "weekly"],
                category="Work"
            )
        ],
        
        "research_articles": [
            DemoContent(
                title="The Psychology of Note-Taking and Memory",
                content="""
                Research shows that the act of note-taking significantly impacts learning and retention:
                
                **Generation Effect**: Information we generate ourselves is better remembered than 
                information we simply read. This explains why summarizing in your own words is 
                more effective than highlighting.
                
                **Encoding Specificity**: Context during learning affects recall. Notes that include 
                contextual information (when, where, why) are more easily retrieved later.
                
                **Elaborative Processing**: Connecting new information to existing knowledge creates 
                stronger memory traces. This is why linking notes together is so powerful.
                
                **Cognitive Load Theory**: External memory systems (like note-taking) free up working 
                memory for higher-order thinking.
                
                Studies by Mueller & Oppenheimer (2014) found that handwritten notes led to better 
                conceptual understanding than typed notes, likely due to more selective processing.
                
                Modern digital tools can combine the benefits of both approaches when designed well.
                """,
                content_type="research",
                tags=["psychology", "memory", "learning", "note-taking", "cognition", "research", "education"],
                category="Research",
                priority=1
            )
        ],
        
        "web_articles": [
            DemoContent(
                title="The Rise of Local-First Software",
                content="""
                Local-first software represents a new approach to application architecture that 
                prioritizes user agency and data ownership.
                
                **Core Principles:**
                - Data lives primarily on user's device
                - Works offline by default
                - Network is enhancement, not requirement
                - User owns and controls their data
                - Fast and responsive interactions
                
                **Benefits:**
                - Privacy and security through data locality
                - Performance through local computation
                - Resilience to network issues
                - No vendor lock-in
                
                **Challenges:**
                - Synchronization complexity
                - Collaboration features
                - Backup and recovery
                - Device storage limitations
                
                Examples include Obsidian, Linear, and Figma's offline capabilities. The trend 
                reflects growing concern about data privacy and platform dependency.
                """,
                content_type="article",
                tags=["local-first", "software", "privacy", "offline", "data-ownership", "architecture"],
                category="Technology",
                source_url="https://example.com/local-first-software",
                priority=1
            )
        ],
        
        "project_notes": [
            DemoContent(
                title="Second Brain MVP - Technical Architecture",
                content="""
                **System Overview:**
                - FastAPI backend with SQLite database
                - React/vanilla JS frontend  
                - Local-first AI processing (sentence-transformers + Ollama)
                - Multi-modal input (text, audio, web content)
                
                **Key Components:**
                - Search: FTS5 + sqlite-vec hybrid search
                - Processing: Background queues for CPU-intensive tasks
                - Integrations: Obsidian sync, Discord bot, Apple Shortcuts
                - AI: Local embeddings and LLM processing
                
                **Data Flow:**
                1. Content capture from multiple sources
                2. AI processing (transcription, summarization, tagging)
                3. Embedding generation and storage
                4. Search indexing (FTS5 + vector)
                5. Obsidian sync with YAML frontmatter
                
                **Next Phase Priorities:**
                - Browser extension deployment
                - Mobile experience optimization  
                - Advanced search features
                - Collaboration functionality
                """,
                content_type="project",
                tags=["second-brain", "architecture", "mvp", "technical", "ai", "search", "local-first"],
                category="Project"
            )
        ]
    }
    
    def seed_demo_data(self, force_reseed: bool = False) -> Dict[str, Any]:
        """Seed database with comprehensive demo data."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        # Check if demo data already exists
        cursor.execute("SELECT COUNT(*) FROM notes WHERE tags LIKE '%demo-data%'")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0 and not force_reseed:
            logger.info(f"Demo data already exists ({existing_count} items). Use force_reseed=True to refresh.")
            return {
                "status": "skipped",
                "existing_items": existing_count,
                "message": "Demo data already exists"
            }
        
        if force_reseed and existing_count > 0:
            # Remove existing demo data
            cursor.execute("DELETE FROM notes WHERE tags LIKE '%demo-data%'")
            logger.info(f"Removed {existing_count} existing demo items")
        
        total_added = 0
        results = {}
        
        # Process each content set
        for set_name, content_items in self.DEMO_CONTENT_SETS.items():
            added_count = 0
            
            for item in content_items:
                try:
                    # Add demo-data tag
                    tags = item.tags + ["demo-data", f"demo-{set_name}"]
                    tags_str = ", ".join(tags)
                    
                    # Create metadata
                    metadata = {
                        "content_type": item.content_type,
                        "category": item.category,
                        "priority": item.priority,
                        "is_demo": True,
                        "demo_set": set_name
                    }
                    
                    if item.source_url:
                        metadata["source_url"] = item.source_url
                    if item.author:
                        metadata["author"] = item.author
                    
                    # Insert note
                    cursor.execute("""
                        INSERT INTO notes (title, body, tags, metadata, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        item.title,
                        item.content.strip(),
                        tags_str,
                        json.dumps(metadata),
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))
                    
                    note_id = cursor.lastrowid
                    added_count += 1
                    total_added += 1
                    
                    # Generate embeddings for demo content
                    try:
                        embedding_text = f"{item.title}\n\n{item.content}"
                        embedding = self.embedder.embed(embedding_text)
                        
                        # Store embedding if vector table exists
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
                        logger.warning(f"Failed to generate embedding for '{item.title}': {e}")
                    
                except Exception as e:
                    logger.error(f"Failed to add demo item '{item.title}': {e}")
                    continue
            
            results[set_name] = added_count
            logger.info(f"Added {added_count} items from {set_name} set")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Demo data seeding complete: {total_added} total items added")
        
        return {
            "status": "success",
            "total_added": total_added,
            "sets": results,
            "message": f"Successfully seeded {total_added} demo items across {len(results)} content sets"
        }
    
    def remove_demo_data(self) -> Dict[str, Any]:
        """Remove all demo data from database."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        # Count existing demo data
        cursor.execute("SELECT COUNT(*) FROM notes WHERE tags LIKE '%demo-data%'")
        count = cursor.fetchone()[0]
        
        if count == 0:
            return {
                "status": "no_data",
                "message": "No demo data found to remove"
            }
        
        # Get demo note IDs for vector cleanup
        cursor.execute("SELECT id FROM notes WHERE tags LIKE '%demo-data%'")
        demo_note_ids = [row[0] for row in cursor.fetchall()]
        
        # Remove from vector table if it exists
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='note_vecs'")
            if cursor.fetchone():
                for note_id in demo_note_ids:
                    cursor.execute("DELETE FROM note_vecs WHERE note_id = ?", (note_id,))
        except Exception as e:
            logger.debug(f"Vector cleanup not needed: {e}")
        
        # Remove demo notes
        cursor.execute("DELETE FROM notes WHERE tags LIKE '%demo-data%'")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Removed {count} demo data items")
        
        return {
            "status": "success",
            "removed_count": count,
            "message": f"Successfully removed {count} demo items"
        }
    
    def get_demo_data_stats(self) -> Dict[str, Any]:
        """Get statistics about current demo data."""
        conn = self.get_conn()
        cursor = conn.cursor()
        
        # Overall counts
        cursor.execute("SELECT COUNT(*) FROM notes WHERE tags LIKE '%demo-data%'")
        total_demo = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM notes WHERE tags NOT LIKE '%demo-data%'")
        total_user = cursor.fetchone()[0]
        
        # Demo data by set
        sets_stats = {}
        for set_name in self.DEMO_CONTENT_SETS.keys():
            cursor.execute("SELECT COUNT(*) FROM notes WHERE tags LIKE ?", (f'%demo-{set_name}%',))
            sets_stats[set_name] = cursor.fetchone()[0]
        
        # Content types
        cursor.execute("""
            SELECT json_extract(metadata, '$.content_type') as type, COUNT(*) as count
            FROM notes 
            WHERE tags LIKE '%demo-data%' AND metadata IS NOT NULL
            GROUP BY type
        """)
        content_types = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_demo_items": total_demo,
            "total_user_items": total_user,
            "demo_sets": sets_stats,
            "content_types": content_types,
            "demo_percentage": round((total_demo / (total_demo + total_user)) * 100, 1) if (total_demo + total_user) > 0 else 0
        }
    
    def toggle_demo_data_visibility(self, visible: bool) -> Dict[str, Any]:
        """Toggle demo data visibility in search results."""
        # This would be implemented at the search/UI level
        # For now, return configuration instruction
        return {
            "status": "info",
            "message": f"Demo data visibility set to: {visible}",
            "implementation": "Implement UI toggle to filter demo-data tagged content"
        }


def get_demo_data_service(get_conn_func):
    """Factory function to get demo data service."""
    return DemoDataService(get_conn_func)