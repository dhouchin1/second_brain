# ──────────────────────────────────────────────────────────────────────────────
# File: tests/test_enhanced_vault_seeding_service.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Test suite for Enhanced Vault Seeding Service.

Tests intelligent content selection, user preferences, and personalized seeding.
"""

import pytest
import sqlite3
import json
from unittest.mock import Mock, patch
from datetime import datetime

from services.enhanced_vault_seeding_service import (
    EnhancedVaultSeedingService,
    SeedingPreferences,
    EnhancedSeedContent,
    ContentCategory,
    UserProfile,
    get_enhanced_seeding_service
)


class TestEnhancedSeedContent:
    """Test the EnhancedSeedContent data class."""
    
    def test_content_creation_with_defaults(self):
        """Test creating enhanced seed content with default values."""
        content = EnhancedSeedContent(
            id="test-id",
            title="Test Content",
            content="Test content body",
            content_type="note",
            categories={ContentCategory.PRODUCTIVITY},
            target_profiles={UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["test", "example"]
        )
        
        assert content.id == "test-id"
        assert content.title == "Test Content"
        assert content.priority == 1  # default
        assert content.dependencies == []  # default
        assert content.estimated_value == 1.0  # default
    
    def test_content_creation_with_custom_values(self):
        """Test creating content with custom values."""
        content = EnhancedSeedContent(
            id="custom-id",
            title="Custom Content",
            content="Custom body",
            content_type="template",
            categories={ContentCategory.RESEARCH, ContentCategory.TECHNICAL},
            target_profiles={UserProfile.DEVELOPER, UserProfile.RESEARCHER},
            difficulty_level="advanced",
            tags=["custom", "advanced"],
            priority=5,
            dependencies=["dep-1", "dep-2"],
            estimated_value=0.8
        )
        
        assert content.priority == 5
        assert content.dependencies == ["dep-1", "dep-2"]
        assert content.estimated_value == 0.8
        assert len(content.categories) == 2
        assert len(content.target_profiles) == 2


class TestSeedingPreferences:
    """Test the SeedingPreferences data class."""
    
    def test_preferences_with_defaults(self):
        """Test preferences with default values."""
        prefs = SeedingPreferences()
        
        assert prefs.user_profile == UserProfile.GENERAL
        assert prefs.preferred_categories == set()
        assert prefs.content_volume == "moderate"
        assert prefs.include_examples == True
        assert prefs.include_templates == True
        assert prefs.include_bookmarks == True
        assert prefs.focus_areas == set()
        assert prefs.language == "en"
    
    def test_preferences_with_custom_values(self):
        """Test preferences with custom values."""
        categories = {ContentCategory.PRODUCTIVITY, ContentCategory.TECHNICAL}
        focus_areas = {"ai", "productivity", "automation"}
        
        prefs = SeedingPreferences(
            user_profile=UserProfile.DEVELOPER,
            preferred_categories=categories,
            content_volume="comprehensive",
            include_examples=False,
            focus_areas=focus_areas
        )
        
        assert prefs.user_profile == UserProfile.DEVELOPER
        assert prefs.preferred_categories == categories
        assert prefs.content_volume == "comprehensive"
        assert prefs.include_examples == False
        assert prefs.focus_areas == focus_areas


class TestEnhancedVaultSeedingService:
    """Test the enhanced vault seeding service."""
    
    @pytest.fixture
    def mock_db_conn(self):
        """Mock database connection."""
        conn = Mock(spec=sqlite3.Connection)
        cursor = Mock(spec=sqlite3.Cursor)
        conn.cursor.return_value = cursor
        conn.execute.return_value = cursor
        cursor.fetchone.return_value = None
        cursor.fetchall.return_value = []
        return conn
    
    @pytest.fixture
    def mock_get_conn(self, mock_db_conn):
        """Mock database connection function."""
        return lambda: mock_db_conn
    
    @pytest.fixture
    def service(self, mock_get_conn):
        """Create enhanced seeding service instance."""
        return EnhancedVaultSeedingService(mock_get_conn)
    
    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert hasattr(service, 'content_catalog')
        assert len(service.content_catalog) > 0
    
    def test_content_catalog_structure(self, service):
        """Test that content catalog has proper structure."""
        catalog = service.content_catalog
        
        # Check that we have different content types
        content_types = {content.content_type for content in catalog}
        assert "note" in content_types
        assert "template" in content_types
        
        # Check that all content has required fields
        for content in catalog:
            assert content.id.startswith("seed-")
            assert len(content.title) > 0
            assert len(content.content) > 0
            assert len(content.categories) > 0
            assert len(content.target_profiles) > 0
            assert content.difficulty_level in ["beginner", "intermediate", "advanced"]
    
    def test_content_score_calculation_profile_match(self, service):
        """Test content scoring with profile matching."""
        content = EnhancedSeedContent(
            id="test-content",
            title="Test",
            content="Test content",
            content_type="note",
            categories={ContentCategory.PRODUCTIVITY},
            target_profiles={UserProfile.DEVELOPER, UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["test"],
            estimated_value=0.5
        )
        
        prefs = SeedingPreferences(user_profile=UserProfile.DEVELOPER)
        score = service._calculate_content_score(content, prefs)
        
        # Base score (0.5) + profile match (0.3) = 0.8
        assert score >= 0.8
    
    def test_content_score_calculation_category_match(self, service):
        """Test content scoring with category preferences."""
        content = EnhancedSeedContent(
            id="test-content",
            title="Test",
            content="Test content",
            content_type="note",
            categories={ContentCategory.PRODUCTIVITY, ContentCategory.TECHNICAL},
            target_profiles={UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["test"],
            estimated_value=0.5
        )
        
        prefs = SeedingPreferences(
            preferred_categories={ContentCategory.PRODUCTIVITY}
        )
        score = service._calculate_content_score(content, prefs)
        
        # Base score (0.5) + category match (0.2) = 0.7
        assert score >= 0.7
    
    def test_content_score_calculation_focus_areas(self, service):
        """Test content scoring with focus area matching."""
        content = EnhancedSeedContent(
            id="test-content",
            title="Test",
            content="Test content",
            content_type="note",
            categories={ContentCategory.PRODUCTIVITY},
            target_profiles={UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["productivity", "automation"],
            estimated_value=0.5
        )
        
        prefs = SeedingPreferences(focus_areas={"productivity", "ai"})
        score = service._calculate_content_score(content, prefs)
        
        # Base score (0.5) + focus area match (0.15) = 0.65
        assert score >= 0.65
    
    def test_content_score_penalty_for_disabled_types(self, service):
        """Test content scoring penalties for disabled content types."""
        template_content = EnhancedSeedContent(
            id="test-template",
            title="Test Template",
            content="Template content",
            content_type="template",
            categories={ContentCategory.PRODUCTIVITY},
            target_profiles={UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["template"],
            estimated_value=0.8
        )
        
        prefs = SeedingPreferences(include_templates=False)
        score = service._calculate_content_score(template_content, prefs)
        
        # Base score (0.8) - template penalty (0.2) = 0.6
        assert score <= 0.6
    
    def test_personalized_seeding_plan_generation(self, service):
        """Test generating personalized seeding plan."""
        prefs = SeedingPreferences(
            user_profile=UserProfile.KNOWLEDGE_WORKER,
            content_volume="light"  # Should return ~5 items
        )
        
        plan = service.get_personalized_seeding_plan(prefs)
        
        assert len(plan) <= 6  # Should be around 5 items for "light"
        assert all(isinstance(content, EnhancedSeedContent) for content in plan)
        
        # Check that content is relevant to knowledge worker
        relevant_profiles = {UserProfile.KNOWLEDGE_WORKER, UserProfile.GENERAL}
        for content in plan:
            assert len(content.target_profiles & relevant_profiles) > 0
    
    def test_personalized_seeding_plan_volume_scaling(self, service):
        """Test that content volume affects plan size."""
        prefs_light = SeedingPreferences(content_volume="light")
        prefs_comprehensive = SeedingPreferences(content_volume="comprehensive")
        
        plan_light = service.get_personalized_seeding_plan(prefs_light)
        plan_comprehensive = service.get_personalized_seeding_plan(prefs_comprehensive)
        
        assert len(plan_light) < len(plan_comprehensive)
        assert len(plan_light) <= 6  # ~5 for light
        assert len(plan_comprehensive) >= 15  # ~20 for comprehensive
    
    def test_ensure_dependencies_inclusion(self, service):
        """Test that dependencies are included in seeding plan."""
        # Create content with dependencies
        dependent_content = EnhancedSeedContent(
            id="dependent-content",
            title="Dependent Content",
            content="Content that depends on others",
            content_type="note",
            categories={ContentCategory.PRODUCTIVITY},
            target_profiles={UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["dependent"],
            dependencies=["seed-getting-started-guide"]  # Should exist in catalog
        )
        
        # Add to catalog temporarily
        service.content_catalog.append(dependent_content)
        
        selected = [dependent_content]
        result = service._ensure_dependencies(selected)
        
        # Should include both the dependent content and its dependency
        content_ids = {content.id for content in result}
        assert "dependent-content" in content_ids
        assert "seed-getting-started-guide" in content_ids
    
    def test_create_user_preferences(self, service, mock_db_conn):
        """Test creating and storing user preferences."""
        user_id = 123
        
        prefs = service.create_user_preferences(
            user_id=user_id,
            profile=UserProfile.DEVELOPER,
            categories=["technical", "productivity"],
            content_volume="comprehensive",
            include_templates=False
        )
        
        assert prefs.user_profile == UserProfile.DEVELOPER
        assert ContentCategory.TECHNICAL in prefs.preferred_categories
        assert ContentCategory.PRODUCTIVITY in prefs.preferred_categories
        assert prefs.content_volume == "comprehensive"
        assert prefs.include_templates == False
        
        # Verify database interaction
        mock_db_conn.execute.assert_called()
        mock_db_conn.commit.assert_called()
    
    def test_get_user_preferences_existing(self, service, mock_db_conn):
        """Test retrieving existing user preferences."""
        user_id = 123
        
        # Mock database return
        prefs_data = {
            "user_profile": "developer",
            "preferred_categories": ["technical", "productivity"],
            "content_volume": "comprehensive",
            "include_examples": True,
            "include_templates": False,
            "include_bookmarks": True,
            "focus_areas": ["ai", "automation"],
            "language": "en"
        }
        
        cursor = mock_db_conn.execute.return_value
        cursor.fetchone.return_value = (json.dumps(prefs_data),)
        
        prefs = service.get_user_preferences(user_id)
        
        assert prefs is not None
        assert prefs.user_profile == UserProfile.DEVELOPER
        assert ContentCategory.TECHNICAL in prefs.preferred_categories
        assert prefs.content_volume == "comprehensive"
        assert prefs.include_templates == False
        assert "ai" in prefs.focus_areas
    
    def test_get_user_preferences_nonexistent(self, service, mock_db_conn):
        """Test retrieving preferences for user who has none."""
        user_id = 999
        
        cursor = mock_db_conn.execute.return_value
        cursor.fetchone.return_value = None
        
        prefs = service.get_user_preferences(user_id)
        
        assert prefs is None
    
    def test_convert_to_legacy_format(self, service):
        """Test converting enhanced content to legacy format."""
        content_plan = [
            EnhancedSeedContent(
                id="seed-note-1",
                title="Test Note",
                content="Note content",
                content_type="note",
                categories={ContentCategory.PRODUCTIVITY},
                target_profiles={UserProfile.GENERAL},
                difficulty_level="beginner",
                tags=["test", "note"],
                priority=3
            ),
            EnhancedSeedContent(
                id="seed-bookmark-1",
                title="Test Bookmark",
                content="Bookmark description",
                content_type="bookmark",
                categories={ContentCategory.TECHNICAL},
                target_profiles={UserProfile.DEVELOPER},
                difficulty_level="intermediate",
                tags=["test", "bookmark"],
                priority=2
            )
        ]
        
        legacy_format = service._convert_to_legacy_format(content_plan)
        
        assert "notes" in legacy_format
        assert "bookmarks" in legacy_format
        assert len(legacy_format["notes"]) == 1
        assert len(legacy_format["bookmarks"]) == 1
        
        note = legacy_format["notes"][0]
        assert note["id"] == "seed-note-1"
        assert note["title"] == "Test Note"
        assert note["type"] == "note"
        assert "test, note" in note["tags"]
    
    def test_seeding_analytics_generation(self, service, mock_db_conn):
        """Test generating seeding analytics for a user."""
        user_id = 123
        
        # Mock database queries
        cursor = mock_db_conn.execute.return_value
        cursor.fetchone.side_effect = [
            (3, 25, 8.33, "2024-01-15T10:00:00"),  # Basic stats
        ]
        cursor.fetchall.return_value = [
            (['["note", "template"]'], 10),
            (['["bookmark"]'], 5)
        ]
        
        # Mock preferences exist
        with patch.object(service, 'get_user_preferences', return_value=SeedingPreferences()):
            analytics = service.get_seeding_analytics(user_id)
        
        assert analytics["total_seedings"] == 3
        assert analytics["total_notes_created"] == 25
        assert abs(analytics["avg_notes_per_seeding"] - 8.33) < 0.01
        assert analytics["last_seeding"] == "2024-01-15T10:00:00"
        assert analytics["has_preferences"] == True
    
    @pytest.mark.asyncio
    async def test_perform_intelligent_seeding(self, service, mock_db_conn):
        """Test performing intelligent seeding operation."""
        user_id = 123
        
        preferences = SeedingPreferences(
            user_profile=UserProfile.KNOWLEDGE_WORKER,
            content_volume="light"
        )
        
        # Mock parent class seeding method
        with patch.object(service, 'seed_vault') as mock_seed_vault, \
             patch.object(service, '_log_intelligent_seeding'):
            
            mock_result = Mock()
            mock_result.success = True
            mock_result.notes_created = 5
            mock_result.message = "Seeding successful"
            mock_seed_vault.return_value = mock_result
            
            result = service.perform_intelligent_seeding(user_id, preferences=preferences)
            
            assert result.success == True
            assert result.notes_created == 5
            mock_seed_vault.assert_called_once()


class TestEnhancedSeedingIntegration:
    """Integration tests for enhanced seeding service."""
    
    def test_factory_function(self):
        """Test the factory function."""
        mock_get_conn = Mock()
        service = get_enhanced_seeding_service(mock_get_conn)
        
        assert isinstance(service, EnhancedVaultSeedingService)
        assert service.get_conn == mock_get_conn
    
    def test_content_categories_and_profiles_coverage(self):
        """Test that content catalog covers various categories and profiles."""
        mock_get_conn = Mock()
        service = get_enhanced_seeding_service(mock_get_conn)
        
        catalog = service.content_catalog
        
        # Check category coverage
        all_categories = set()
        for content in catalog:
            all_categories.update(content.categories)
        
        # Should have good coverage of main categories
        assert ContentCategory.PRODUCTIVITY in all_categories
        assert ContentCategory.KNOWLEDGE_MANAGEMENT in all_categories
        assert ContentCategory.TECHNICAL in all_categories
        
        # Check profile coverage
        all_profiles = set()
        for content in catalog:
            all_profiles.update(content.target_profiles)
        
        # Should target multiple user types
        assert UserProfile.GENERAL in all_profiles
        assert UserProfile.KNOWLEDGE_WORKER in all_profiles
        assert UserProfile.DEVELOPER in all_profiles


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def service(self):
        """Service with mocked dependencies."""
        mock_get_conn = Mock()
        return EnhancedVaultSeedingService(mock_get_conn)
    
    def test_content_scoring_with_zero_estimated_value(self, service):
        """Test content scoring when estimated value is zero."""
        content = EnhancedSeedContent(
            id="zero-value-content",
            title="Zero Value",
            content="Content",
            content_type="note",
            categories={ContentCategory.PRODUCTIVITY},
            target_profiles={UserProfile.GENERAL},
            difficulty_level="beginner",
            tags=["zero"],
            estimated_value=0.0
        )
        
        prefs = SeedingPreferences()
        score = service._calculate_content_score(content, prefs)
        
        # Score should be non-negative even with zero base value
        assert score >= 0
    
    def test_preferences_with_unknown_categories(self, service):
        """Test handling of unknown category strings."""
        user_id = 123
        
        # Should handle unknown categories gracefully
        prefs = service.create_user_preferences(
            user_id=user_id,
            categories=["unknown_category", "productivity", "invalid"]
        )
        
        # Should only include valid categories
        assert ContentCategory.PRODUCTIVITY in prefs.preferred_categories
        assert len(prefs.preferred_categories) == 1
    
    def test_seeding_plan_with_empty_catalog(self, service):
        """Test seeding plan generation with empty content catalog."""
        service.content_catalog = []  # Empty catalog
        
        prefs = SeedingPreferences()
        plan = service.get_personalized_seeding_plan(prefs)
        
        assert plan == []
    
    def test_database_error_handling_in_preferences(self, service):
        """Test database error handling in preferences operations."""
        service.get_conn.side_effect = sqlite3.Error("Database error")
        
        # Should handle database errors gracefully
        prefs = service.get_user_preferences(123)
        assert prefs is None
        
        # Should not raise exception when saving preferences
        try:
            service._save_user_preferences(123, SeedingPreferences())
        except Exception as e:
            pytest.fail(f"Should handle database errors gracefully: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])