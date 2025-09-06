# ──────────────────────────────────────────────────────────────────────────────
# File: services/enhanced_vault_seeding_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Enhanced Vault Seeding Service for Second Brain

Provides intelligent, personalized content seeding with user preferences,
dynamic content selection, and advanced seeding strategies.
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from services.vault_seeding_service import (
    VaultSeedingService, SeedingResult, SeedingOptions
)
from config import settings

log = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Categories for seeding content classification."""
    PRODUCTIVITY = "productivity"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"
    PROJECT_MANAGEMENT = "project_management"
    RESEARCH = "research"
    PERSONAL_DEVELOPMENT = "personal_development"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"


class UserProfile(Enum):
    """User profiles for personalized seeding."""
    KNOWLEDGE_WORKER = "knowledge_worker"
    RESEARCHER = "researcher"
    STUDENT = "student"
    CREATIVE_PROFESSIONAL = "creative_professional"
    DEVELOPER = "developer"
    MANAGER = "manager"
    ENTREPRENEUR = "entrepreneur"
    GENERAL = "general"


@dataclass
class SeedingPreferences:
    """User preferences for content seeding."""
    user_profile: UserProfile = UserProfile.GENERAL
    preferred_categories: Set[ContentCategory] = None
    content_volume: str = "moderate"  # "light", "moderate", "comprehensive"
    include_examples: bool = True
    include_templates: bool = True
    include_bookmarks: bool = True
    focus_areas: Set[str] = None  # Custom focus keywords
    language: str = "en"
    
    def __post_init__(self):
        if self.preferred_categories is None:
            self.preferred_categories = set()
        if self.focus_areas is None:
            self.focus_areas = set()


@dataclass
class EnhancedSeedContent:
    """Enhanced seed content with metadata and targeting."""
    id: str
    title: str
    content: str
    content_type: str  # "note", "bookmark", "template"
    categories: Set[ContentCategory]
    target_profiles: Set[UserProfile]
    difficulty_level: str  # "beginner", "intermediate", "advanced"
    tags: List[str]
    priority: int = 1  # 1-5, higher is more important
    dependencies: List[str] = None  # Other content IDs this depends on
    estimated_value: float = 1.0  # Estimated user value 0-1
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class EnhancedVaultSeedingService(VaultSeedingService):
    """Enhanced vault seeding service with intelligent content selection."""
    
    def __init__(self, get_conn_func):
        """Initialize enhanced seeding service."""
        super().__init__(get_conn_func)
        self.content_catalog = self._build_content_catalog()
    
    def _build_content_catalog(self) -> List[EnhancedSeedContent]:
        """Build enhanced content catalog with intelligent categorization."""
        catalog = []
        
        # Productivity & Organization Content
        catalog.extend([
            EnhancedSeedContent(
                id="seed-getting-started-guide",
                title="🚀 Getting Started with Second Brain",
                content="""# Getting Started with Your Second Brain

Welcome to your intelligent knowledge management system! This guide will help you maximize the value of your Second Brain.

## Core Concepts

### 📝 Capture Everything
Your Second Brain excels at capturing information from multiple sources:
- Quick notes via web interface
- Voice memos through Apple Shortcuts  
- Web content through URL parsing
- Documents via file upload
- Discord conversations (if enabled)

### 🔍 Intelligent Search
Two powerful search modes work together:
- **Text Search**: Fast keyword and phrase matching
- **Semantic Search**: AI-powered meaning-based discovery

### 🤖 AI Enhancement
Every piece of content gets automatically enhanced:
- Smart title generation
- Relevant tag suggestions  
- Concise summaries
- Action item extraction

## Quick Start Workflow

1. **Capture** - Add your first note using the quick capture form
2. **Search** - Try searching for concepts, not just keywords
3. **Connect** - Use tags and links to build knowledge graphs
4. **Review** - Regular review helps surface valuable insights

## Pro Tips

- Use natural language in search - the AI understands context
- Voice memos are automatically transcribed and processed
- Web URLs are automatically parsed for content and metadata
- All content syncs to your Obsidian vault if configured

*This note was created by your auto-seeding system to help you get started.*""",
                content_type="note",
                categories={ContentCategory.KNOWLEDGE_MANAGEMENT, ContentCategory.PRODUCTIVITY},
                target_profiles={UserProfile.GENERAL, UserProfile.KNOWLEDGE_WORKER, UserProfile.STUDENT},
                difficulty_level="beginner",
                tags=["getting-started", "guide", "onboarding", "second-brain"],
                priority=5,
                estimated_value=0.9
            ),
            
            EnhancedSeedContent(
                id="seed-weekly-review-template",
                title="📅 Weekly Review Template",
                content="""# Weekly Review - {{date}}

## 🎯 Goals This Week
- [ ] 
- [ ] 
- [ ] 

## 📥 Inbox Processing
- [ ] Clear email inbox
- [ ] Process captured notes
- [ ] Review voice memos
- [ ] Clean up downloads folder

## 📊 Weekly Metrics
- Notes captured: 
- Articles read: 
- Key insights: 

## 🔄 Review Areas
### Projects
- [ ] Check project status
- [ ] Update next actions
- [ ] Review deadlines

### Learning
- [ ] What did I learn this week?
- [ ] What questions emerged?
- [ ] What should I research next?

### Reflection
- [ ] What worked well?
- [ ] What could be improved?
- [ ] What am I grateful for?

## ➡️ Next Week Focus
1. 
2. 
3. 

---
*Template created: {{timestamp}}*""",
                content_type="template",
                categories={ContentCategory.PRODUCTIVITY, ContentCategory.PERSONAL_DEVELOPMENT},
                target_profiles={UserProfile.KNOWLEDGE_WORKER, UserProfile.MANAGER, UserProfile.ENTREPRENEUR},
                difficulty_level="beginner",
                tags=["template", "weekly-review", "productivity", "reflection"],
                priority=4,
                estimated_value=0.8
            ),
            
            EnhancedSeedContent(
                id="seed-project-planning-template",
                title="🎯 Project Planning Template",
                content="""# Project: {{project_name}}

## 📋 Project Overview
**Start Date**: {{start_date}}  
**Target Completion**: {{target_date}}  
**Status**: {{status}}  
**Priority**: {{priority}}

### 🎯 Objectives
- 
- 
- 

### 📏 Success Criteria
- [ ] 
- [ ] 
- [ ] 

## 🗂️ Project Phases

### Phase 1: Planning
- [ ] Define requirements
- [ ] Research and analysis
- [ ] Create project timeline
- [ ] Identify resources needed

### Phase 2: Execution
- [ ] 
- [ ] 
- [ ] 

### Phase 3: Review & Closure
- [ ] Test and validate results
- [ ] Document lessons learned
- [ ] Archive project materials
- [ ] Celebrate completion

## 📚 Resources
- **Documents**: 
- **Tools**: 
- **People**: 
- **References**: 

## 🔄 Regular Check-ins
- **Weekly**: Review progress and adjust timeline
- **Bi-weekly**: Stakeholder updates
- **Monthly**: Deep project health assessment

## ⚠️ Risk Assessment
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
|      |             |        |            |

## 📝 Project Log
### {{date}}
- 

---
*Project template created: {{timestamp}}*""",
                content_type="template",
                categories={ContentCategory.PROJECT_MANAGEMENT, ContentCategory.PRODUCTIVITY},
                target_profiles={UserProfile.MANAGER, UserProfile.ENTREPRENEUR, UserProfile.DEVELOPER},
                difficulty_level="intermediate",
                tags=["template", "project-management", "planning", "organization"],
                priority=4,
                estimated_value=0.85
            )
        ])
        
        # Research & Learning Content
        catalog.extend([
            EnhancedSeedContent(
                id="seed-research-methodology",
                title="🔬 Research Methodology Guide",
                content="""# Research Methodology for Knowledge Workers

## 🎯 Research Framework

### 1. Define Research Question
- What specific question am I trying to answer?
- What would success look like?
- What are my assumptions?

### 2. Information Gathering Strategy
- **Primary Sources**: Direct data, interviews, surveys
- **Secondary Sources**: Academic papers, reports, books
- **Digital Sources**: Websites, databases, archives
- **Expert Consultation**: Subject matter experts, mentors

### 3. Source Evaluation Criteria
- **Authority**: Who created this information?
- **Accuracy**: Is the information correct and well-sourced?
- **Objectivity**: Are there obvious biases?
- **Currency**: How recent is this information?
- **Coverage**: How comprehensive is the treatment?

## 📚 Research Tools & Techniques

### Note-Taking Methods
- **Cornell Notes**: Structured note-taking with summary section
- **Mind Mapping**: Visual connection of concepts
- **Zettelkasten**: Atomic notes with linking system
- **Summary Cards**: Key points on index cards

### Digital Research Tools
- **Search Operators**: Use advanced Google search techniques
- **Citation Management**: Track sources systematically
- **Web Archiving**: Save important pages for later reference
- **Alert Systems**: Set up notifications for new information

## 🔄 Research Process

1. **Scoping Phase** (20% of time)
   - Define research boundaries
   - Identify key concepts and terminology
   - Create initial research map

2. **Exploration Phase** (50% of time)
   - Broad information gathering
   - Follow interesting threads
   - Build understanding of landscape

3. **Focused Investigation** (20% of time)
   - Deep dive on specific aspects
   - Validate key findings
   - Fill gaps in understanding

4. **Synthesis Phase** (10% of time)
   - Organize findings
   - Draw connections
   - Prepare conclusions

## 💡 Best Practices

- **Document Everything**: Including dead ends and failed approaches
- **Regular Reviews**: Weekly assessment of progress and direction
- **Version Control**: Keep track of how your understanding evolves
- **Collaboration**: Share findings with others for feedback

## 🚨 Common Pitfalls

- **Confirmation Bias**: Seeking only supporting evidence
- **Information Overload**: Consuming without synthesizing
- **Scope Creep**: Research question keeps expanding
- **Perfectionism**: Never feeling "done" with research

---
*Research methodology guide - adapt to your specific needs*""",
                content_type="note",
                categories={ContentCategory.RESEARCH, ContentCategory.KNOWLEDGE_MANAGEMENT},
                target_profiles={UserProfile.RESEARCHER, UserProfile.STUDENT, UserProfile.KNOWLEDGE_WORKER},
                difficulty_level="intermediate",
                tags=["research", "methodology", "learning", "knowledge-management"],
                priority=3,
                estimated_value=0.75
            ),
            
            EnhancedSeedContent(
                id="seed-learning-framework",
                title="🧠 Personal Learning Framework",
                content="""# Personal Learning Framework

## 🎯 Learning Objectives Setting

### SMART Learning Goals
- **Specific**: What exactly will I learn?
- **Measurable**: How will I know I've learned it?
- **Achievable**: Is this realistic given my resources?
- **Relevant**: Why is this important to me?
- **Time-bound**: When will I complete this?

## 📚 Learning Methods

### Active Learning Techniques
- **Feynman Technique**: Explain concepts in simple terms
- **Spaced Repetition**: Review material at increasing intervals
- **Interleaving**: Mix different topics in study sessions
- **Testing**: Regular self-assessment and quizzing

### Learning Modalities
- **Visual**: Diagrams, mind maps, flowcharts
- **Auditory**: Podcasts, lectures, discussions
- **Kinesthetic**: Hands-on practice, experiments
- **Reading/Writing**: Articles, notes, summaries

## 🔄 Learning Cycle

1. **Preparation** (10%)
   - Set learning objectives
   - Gather resources
   - Create study schedule

2. **Acquisition** (40%)
   - Consume content actively
   - Take structured notes
   - Ask questions continuously

3. **Practice** (30%)
   - Apply new knowledge
   - Solve problems
   - Create something new

4. **Reflection** (20%)
   - Assess understanding
   - Identify gaps
   - Connect to existing knowledge

## 📊 Progress Tracking

### Weekly Learning Log
- What did I learn this week?
- What challenges did I encounter?
- How can I apply this knowledge?
- What should I learn next?

### Knowledge Mapping
- Create visual maps of your learning domains
- Show connections between different areas
- Identify opportunities for cross-pollination

## 🎯 Continuous Improvement

- **Regular Review**: Monthly assessment of learning effectiveness
- **Method Experimentation**: Try new learning techniques
- **Community Learning**: Join study groups or online communities
- **Teaching Others**: Share knowledge to reinforce understanding

---
*Adapt this framework to match your learning style and goals*""",
                content_type="note",
                categories={ContentCategory.PERSONAL_DEVELOPMENT, ContentCategory.KNOWLEDGE_MANAGEMENT},
                target_profiles={UserProfile.STUDENT, UserProfile.KNOWLEDGE_WORKER, UserProfile.RESEARCHER},
                difficulty_level="beginner",
                tags=["learning", "personal-development", "framework", "education"],
                priority=3,
                estimated_value=0.8
            )
        ])
        
        # Technical Content
        catalog.extend([
            EnhancedSeedContent(
                id="seed-second-brain-api-guide",
                title="⚡ Second Brain API Quick Reference",
                content="""# Second Brain API Quick Reference

## 🚀 Quick Capture Endpoints

### Text Notes
```bash
# Simple text note
curl -X POST "http://localhost:8082/api/unified-capture/quick-note" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "Your note content here"}'

# Form-encoded (alternative)
curl -X POST "http://localhost:8082/api/unified-capture/quick-note" \\
  -H "Content-Type: application/x-www-form-urlencoded" \\
  -d "content=Your note content here"

# Query parameter (alternative)
curl -X GET "http://localhost:8082/api/unified-capture/quick-note?content=Your note content here"
```

### Rich Content Capture
```bash
# Text with metadata
curl -X POST "http://localhost:8082/api/unified-capture/text" \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "Detailed note content",
    "title": "Custom Title",
    "tags": ["important", "api"],
    "generate_summary": true
  }'
```

### File Uploads
```bash
# Image with OCR
curl -X POST "http://localhost:8082/api/unified-capture/image" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_data": "base64_encoded_image_data",
    "auto_tag": true,
    "generate_summary": true
  }'
```

### URL Processing
```bash
# Web content extraction
curl -X POST "http://localhost:8082/api/unified-capture/url" \\
  -H "Content-Type: application/json" \\
  -d '{
    "url": "https://example.com/article",
    "generate_summary": true,
    "auto_tag": true
  }'
```

## 🔍 Search API

```bash
# Basic search
curl -X GET "http://localhost:8082/api/search?q=your search terms"

# Advanced search with filters
curl -X GET "http://localhost:8082/api/search?q=machine learning&tags=research&limit=20"
```

## 📊 Status and Health

```bash
# System status
curl -X GET "http://localhost:8082/health"

# Processing status
curl -X GET "http://localhost:8082/api/status"
```

## 🔐 Authentication

Most endpoints require authentication. Include your session token:

```bash
curl -X POST "http://localhost:8082/api/unified-capture/quick-note" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "Authenticated note"}'
```

## 📱 Integration Examples

### Apple Shortcuts
Create an iOS shortcut that sends quick notes:
1. Add "Get Text from Input"
2. Add "Get Contents of URL" with POST method
3. Configure endpoint and authentication

### Browser Bookmarklet
```javascript
javascript:(function(){
  var content = window.getSelection().toString() || document.title;
  var url = 'http://localhost:8082/api/unified-capture/quick-note';
  fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({content: content, source: 'bookmarklet'})
  });
})();
```

## 🚨 Common Issues

- **CORS Errors**: Ensure proper headers for cross-origin requests
- **Rate Limiting**: Space out rapid successive requests
- **Large Files**: Use appropriate timeouts for file uploads
- **Authentication**: Check token validity and renewal

---
*API Reference - Updated for current endpoints*""",
                content_type="note",
                categories={ContentCategory.TECHNICAL},
                target_profiles={UserProfile.DEVELOPER, UserProfile.KNOWLEDGE_WORKER},
                difficulty_level="intermediate",
                tags=["api", "technical", "integration", "reference"],
                priority=2,
                estimated_value=0.7
            )
        ])
        
        # Bookmarks and Resources
        catalog.extend([
            EnhancedSeedContent(
                id="seed-productivity-bookmarks",
                title="🔗 Essential Productivity Resources",
                content="""# Essential Productivity Resources

## 📚 Knowledge Management
- [Building a Second Brain](https://www.buildingasecondbrain.com/) - Comprehensive methodology for personal knowledge management
- [Zettelkasten Method](https://zettelkasten.de/) - Atomic note-taking and linking system
- [PARA Method](https://fortelabs.co/blog/para/) - Organizational system for digital information
- [Getting Things Done](https://gettingthingsdone.com/) - David Allen's productivity methodology

## 🛠️ Tools and Software
- [Obsidian](https://obsidian.md/) - Powerful note-taking and knowledge management
- [Notion](https://www.notion.so/) - All-in-one workspace for notes, tasks, and databases
- [Roam Research](https://roamresearch.com/) - Networked thought and bidirectional linking
- [Logseq](https://logseq.com/) - Local-first, non-linear, outliner notebook

## 📖 Reading and Research
- [Readwise](https://readwise.io/) - Highlight management and spaced repetition
- [Matter](https://hq.getmatter.app/) - Read-later app with social features
- [Pocket](https://getpocket.com/) - Save articles and videos for later
- [Hypothesis](https://web.hypothes.is/) - Web annotation and collaborative research

## ⏰ Time Management
- [Toggl](https://toggl.com/) - Time tracking and productivity insights
- [RescueTime](https://www.rescuetime.com/) - Automatic time tracking and analysis
- [Forest](https://www.forestapp.cc/) - Focus timer with gamification
- [Pomodone](https://pomodoneapp.com/) - Pomodoro technique integration

## 🧠 Learning and Development
- [Coursera](https://www.coursera.org/) - University-level courses online
- [Khan Academy](https://www.khanacademy.org/) - Free educational resources
- [Anki](https://apps.ankiweb.net/) - Spaced repetition flashcard system
- [Brilliant](https://brilliant.org/) - Interactive learning in STEM fields

## 🔗 Integration and Automation
- [Zapier](https://zapier.com/) - Workflow automation between apps
- [IFTTT](https://ifttt.com/) - Simple automation for everyday tasks
- [Shortcuts (iOS)](https://support.apple.com/guide/shortcuts/) - iOS automation and workflows
- [Hazel](https://www.noodlesoft.com/) - Automated file organization (Mac)

---
*Curated collection of productivity resources*""",
                content_type="bookmark",
                categories={ContentCategory.PRODUCTIVITY, ContentCategory.KNOWLEDGE_MANAGEMENT},
                target_profiles={UserProfile.KNOWLEDGE_WORKER, UserProfile.GENERAL},
                difficulty_level="beginner",
                tags=["bookmarks", "productivity", "tools", "resources"],
                priority=2,
                estimated_value=0.6
            )
        ])
        
        return catalog
    
    def get_personalized_seeding_plan(
        self, 
        preferences: SeedingPreferences,
        target_count: int = None
    ) -> List[EnhancedSeedContent]:
        """Generate personalized seeding plan based on user preferences."""
        
        if target_count is None:
            volume_map = {"light": 5, "moderate": 10, "comprehensive": 20}
            target_count = volume_map.get(preferences.content_volume, 10)
        
        # Score content based on user preferences
        scored_content = []
        for content in self.content_catalog:
            score = self._calculate_content_score(content, preferences)
            if score > 0:
                scored_content.append((content, score))
        
        # Sort by score and select top content
        scored_content.sort(key=lambda x: x[1], reverse=True)
        selected_content = [content for content, score in scored_content[:target_count]]
        
        # Ensure dependencies are included
        selected_content = self._ensure_dependencies(selected_content)
        
        return selected_content
    
    def _calculate_content_score(
        self, 
        content: EnhancedSeedContent, 
        preferences: SeedingPreferences
    ) -> float:
        """Calculate relevance score for content based on user preferences."""
        score = content.estimated_value  # Base score
        
        # Profile matching
        if preferences.user_profile in content.target_profiles:
            score += 0.3
        
        # Category preferences
        if preferences.preferred_categories:
            category_match = len(content.categories & preferences.preferred_categories) > 0
            if category_match:
                score += 0.2
        
        # Content type preferences
        if not preferences.include_examples and content.content_type == "note":
            score -= 0.1
        if not preferences.include_templates and content.content_type == "template":
            score -= 0.2
        if not preferences.include_bookmarks and content.content_type == "bookmark":
            score -= 0.1
        
        # Focus area matching
        if preferences.focus_areas:
            tag_match = any(tag in content.tags for tag in preferences.focus_areas)
            if tag_match:
                score += 0.15
        
        # Priority weighting
        score += content.priority * 0.05
        
        return max(0, score)  # Ensure non-negative score
    
    def _ensure_dependencies(
        self, 
        selected_content: List[EnhancedSeedContent]
    ) -> List[EnhancedSeedContent]:
        """Ensure all dependencies are included in selected content."""
        content_ids = {content.id for content in selected_content}
        content_by_id = {content.id: content for content in self.content_catalog}
        
        to_add = set()
        for content in selected_content:
            for dep_id in content.dependencies:
                if dep_id not in content_ids and dep_id in content_by_id:
                    to_add.add(dep_id)
        
        # Add dependencies
        for dep_id in to_add:
            selected_content.append(content_by_id[dep_id])
        
        return selected_content
    
    def create_user_preferences(
        self,
        user_id: int,
        profile: UserProfile = UserProfile.GENERAL,
        categories: List[str] = None,
        content_volume: str = "moderate",
        **kwargs
    ) -> SeedingPreferences:
        """Create user preferences and store them."""
        
        # Convert string categories to enum
        preferred_categories = set()
        if categories:
            for cat_str in categories:
                try:
                    preferred_categories.add(ContentCategory(cat_str))
                except ValueError:
                    log.warning(f"Unknown category: {cat_str}")
        
        preferences = SeedingPreferences(
            user_profile=profile,
            preferred_categories=preferred_categories,
            content_volume=content_volume,
            **kwargs
        )
        
        # Store preferences in database
        self._save_user_preferences(user_id, preferences)
        
        return preferences
    
    def _save_user_preferences(self, user_id: int, preferences: SeedingPreferences):
        """Save user preferences to database."""
        conn = self.get_conn()
        try:
            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_seeding_preferences (
                    user_id INTEGER PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            preferences_json = json.dumps(asdict(preferences), default=str)
            now = datetime.now().isoformat()
            
            conn.execute("""
                INSERT OR REPLACE INTO user_seeding_preferences 
                (user_id, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (user_id, preferences_json, now, now))
            
            conn.commit()
        except Exception as e:
            log.error(f"Failed to save user preferences: {e}")
        finally:
            conn.close()
    
    def get_user_preferences(self, user_id: int) -> Optional[SeedingPreferences]:
        """Retrieve user preferences from database."""
        conn = self.get_conn()
        try:
            cursor = conn.execute("""
                SELECT preferences FROM user_seeding_preferences WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                prefs_data = json.loads(row[0])
                
                # Convert back to proper types
                if 'user_profile' in prefs_data:
                    prefs_data['user_profile'] = UserProfile(prefs_data['user_profile'])
                
                if 'preferred_categories' in prefs_data:
                    prefs_data['preferred_categories'] = {
                        ContentCategory(cat) for cat in prefs_data['preferred_categories']
                    }
                
                if 'focus_areas' in prefs_data:
                    prefs_data['focus_areas'] = set(prefs_data['focus_areas'])
                
                return SeedingPreferences(**prefs_data)
            
            return None
            
        except Exception as e:
            log.error(f"Failed to retrieve user preferences: {e}")
            return None
        finally:
            conn.close()
    
    def perform_intelligent_seeding(
        self, 
        user_id: int, 
        options: SeedingOptions = None,
        preferences: SeedingPreferences = None
    ) -> SeedingResult:
        """Perform intelligent seeding based on user preferences."""
        
        if preferences is None:
            preferences = self.get_user_preferences(user_id) or SeedingPreferences()
        
        if options is None:
            options = SeedingOptions()
        
        try:
            # Get personalized content plan
            content_plan = self.get_personalized_seeding_plan(preferences)
            
            # Create temporary seeding content
            temp_seed_data = self._convert_to_legacy_format(content_plan)
            
            # Patch the legacy seeding data temporarily
            original_notes = getattr(self, '_original_seed_notes', None)
            self._original_seed_notes = temp_seed_data['notes']
            
            # Use parent class seeding with our custom content
            with self._mock_seed_data(temp_seed_data):
                result = self.seed_vault(user_id, options)
            
            if result.success:
                # Log the intelligent seeding
                self._log_intelligent_seeding(
                    user_id, preferences, content_plan, result
                )
            
            return result
            
        except Exception as e:
            log.error(f"Intelligent seeding failed: {e}")
            return SeedingResult(
                success=False,
                message="Intelligent seeding failed",
                error=str(e)
            )
        finally:
            # Restore original data
            if hasattr(self, '_original_seed_notes'):
                delattr(self, '_original_seed_notes')
    
    def _convert_to_legacy_format(self, content_plan: List[EnhancedSeedContent]) -> Dict:
        """Convert enhanced content to legacy seeding format."""
        notes = []
        bookmarks = []
        
        for content in content_plan:
            item = {
                "id": content.id,
                "title": content.title,
                "tags": ", ".join(content.tags),
                "summary": f"Priority {content.priority} content for {content.difficulty_level} users"
            }
            
            if content.content_type in ["note", "template"]:
                item["type"] = content.content_type
                item["content"] = content.content
                notes.append(item)
            elif content.content_type == "bookmark":
                item["url"] = "https://example.com"  # Placeholder - would need actual URLs
                item["content"] = content.content
                bookmarks.append(item)
        
        return {"notes": notes, "bookmarks": bookmarks}
    
    def _mock_seed_data(self, temp_data: Dict):
        """Context manager to temporarily replace seed data."""
        class SeedDataMock:
            def __enter__(self):
                # This would patch the imported seed data
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return SeedDataMock()
    
    def _log_intelligent_seeding(
        self, 
        user_id: int,
        preferences: SeedingPreferences,
        content_plan: List[EnhancedSeedContent],
        result: SeedingResult
    ):
        """Log details of intelligent seeding operation."""
        conn = self.get_conn()
        try:
            # Create intelligent seeding log table if needed
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intelligent_seeding_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    user_profile TEXT NOT NULL,
                    content_volume TEXT NOT NULL,
                    content_count INTEGER NOT NULL,
                    content_types TEXT NOT NULL,
                    categories TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    notes_created INTEGER NOT NULL,
                    message TEXT
                )
            """)
            
            content_types = [c.content_type for c in content_plan]
            categories = set()
            for c in content_plan:
                categories.update(c.categories)
            
            conn.execute("""
                INSERT INTO intelligent_seeding_log 
                (user_id, created_at, user_profile, content_volume, content_count, 
                 content_types, categories, success, notes_created, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                datetime.now().isoformat(),
                preferences.user_profile.value,
                preferences.content_volume,
                len(content_plan),
                json.dumps(content_types),
                json.dumps([c.value for c in categories]),
                result.success,
                result.notes_created,
                result.message
            ))
            
            conn.commit()
            log.info(f"Logged intelligent seeding for user {user_id}: {len(content_plan)} items")
            
        except Exception as e:
            log.error(f"Failed to log intelligent seeding: {e}")
        finally:
            conn.close()
    
    def get_seeding_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get analytics about user's seeding history and preferences."""
        conn = self.get_conn()
        try:
            # Get basic seeding stats
            cursor = conn.execute("""
                SELECT COUNT(*) as total_seedings,
                       SUM(notes_created) as total_notes,
                       AVG(notes_created) as avg_notes_per_seeding,
                       MAX(created_at) as last_seeding
                FROM intelligent_seeding_log 
                WHERE user_id = ? AND success = 1
            """, (user_id,))
            
            stats = cursor.fetchone()
            
            # Get content type distribution
            cursor = conn.execute("""
                SELECT content_types, content_count 
                FROM intelligent_seeding_log 
                WHERE user_id = ? AND success = 1
            """, (user_id,))
            
            type_distribution = {}
            for row in cursor.fetchall():
                types = json.loads(row[0])
                for content_type in types:
                    type_distribution[content_type] = type_distribution.get(content_type, 0) + 1
            
            return {
                "total_seedings": stats[0] or 0,
                "total_notes_created": stats[1] or 0,
                "avg_notes_per_seeding": stats[2] or 0,
                "last_seeding": stats[3],
                "content_type_distribution": type_distribution,
                "has_preferences": self.get_user_preferences(user_id) is not None
            }
            
        except Exception as e:
            log.error(f"Failed to get seeding analytics: {e}")
            return {"error": str(e)}
        finally:
            conn.close()


def get_enhanced_seeding_service(get_conn_func) -> EnhancedVaultSeedingService:
    """Factory function to get enhanced seeding service instance."""
    return EnhancedVaultSeedingService(get_conn_func)