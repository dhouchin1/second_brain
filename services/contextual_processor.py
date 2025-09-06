# ──────────────────────────────────────────────────────────────────────────────
# File: services/contextual_processor.py
# ──────────────────────────────────────────────────────────────────────────────
"""
Contextual Processor Service

Provides location-aware, time-based, and source-specific processing capabilities
for the Second Brain capture system. This service enables smart automation
based on contextual information such as:

- Location-based content enrichment and processing rules
- Time-based processing strategies (time of day, day of week, seasonal)
- Source-specific processing profiles and customization
- Dynamic processing rule engine with user-configurable rules
- Context-aware metadata enhancement and tag suggestions
"""

import logging
import json
import asyncio
import hashlib
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from pathlib import Path

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

try:
    # For reverse geocoding (optional)
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Processing strategy types."""
    MINIMAL = "minimal"           # Basic processing, fast
    STANDARD = "standard"         # Default processing
    ENHANCED = "enhanced"         # Full processing with all features
    CUSTOM = "custom"             # User-defined processing rules


class TimeContext(Enum):
    """Time-based context categories."""
    EARLY_MORNING = "early_morning"  # 5-8 AM
    MORNING = "morning"              # 8-12 PM
    AFTERNOON = "afternoon"          # 12-5 PM
    EVENING = "evening"              # 5-9 PM
    NIGHT = "night"                  # 9 PM-12 AM
    LATE_NIGHT = "late_night"        # 12-5 AM


class LocationContext(Enum):
    """Location-based context categories."""
    HOME = "home"
    WORK = "work"
    COMMUTE = "commute"
    TRAVEL = "travel"
    MEETING_ROOM = "meeting_room"
    CAFE = "cafe"
    UNKNOWN = "unknown"


class DayContext(Enum):
    """Day-based context categories."""
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    VACATION = "vacation"


@dataclass
class LocationInfo:
    """Location information structure."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    place_type: Optional[str] = None  # home, work, restaurant, etc.
    accuracy: Optional[float] = None  # Location accuracy in meters
    timezone: Optional[str] = None


@dataclass
class TemporalInfo:
    """Temporal information structure."""
    timestamp: datetime
    timezone: Optional[str] = None
    time_context: Optional[TimeContext] = None
    day_context: Optional[DayContext] = None
    is_business_hours: bool = False
    season: Optional[str] = None  # spring, summer, fall, winter


@dataclass
class SourceProfile:
    """Source-specific processing profile."""
    source_type: str
    processing_strategy: ProcessingStrategy = ProcessingStrategy.STANDARD
    priority_boost: float = 1.0  # Multiplier for processing priority
    
    # Processing preferences
    enable_ai_enhancement: bool = True
    enable_location_enrichment: bool = True
    enable_time_enrichment: bool = True
    
    # Content-specific settings
    auto_tag_confidence_threshold: float = 0.5
    action_extraction_enabled: bool = True
    summary_generation_enabled: bool = True
    
    # Custom tags and metadata
    default_tags: List[str] = field(default_factory=list)
    metadata_enrichments: Dict[str, Any] = field(default_factory=dict)
    
    # Processing rules
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcessingRule:
    """A contextual processing rule."""
    rule_id: str
    name: str
    description: str
    
    # Conditions
    time_conditions: Dict[str, Any] = field(default_factory=dict)
    location_conditions: Dict[str, Any] = field(default_factory=dict)
    source_conditions: Dict[str, Any] = field(default_factory=dict)
    content_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Actions
    processing_modifications: Dict[str, Any] = field(default_factory=dict)
    tag_additions: List[str] = field(default_factory=list)
    metadata_additions: Dict[str, Any] = field(default_factory=dict)
    
    # Rule metadata
    priority: int = 100  # Lower = higher priority
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0


@dataclass
class ContextualProcessingResult:
    """Result of contextual processing."""
    # Enhanced context information
    location_info: Optional[LocationInfo] = None
    temporal_info: Optional[TemporalInfo] = None
    
    # Applied processing modifications
    applied_rules: List[str] = field(default_factory=list)
    processing_strategy: ProcessingStrategy = ProcessingStrategy.STANDARD
    priority_adjustment: float = 1.0
    
    # Content enhancements
    contextual_tags: List[str] = field(default_factory=list)
    location_tags: List[str] = field(default_factory=list)
    time_tags: List[str] = field(default_factory=list)
    
    # Metadata enhancements
    enriched_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    rules_evaluated: int = 0
    location_accuracy: Optional[float] = None


class LocationEnricher:
    """Handles location-based content enrichment."""
    
    def __init__(self):
        """Initialize location enricher."""
        self.location_cache: Dict[str, LocationInfo] = {}
        self.cache_ttl = timedelta(hours=24)  # Cache locations for 24 hours
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Known location patterns
        self.location_patterns = {
            LocationContext.HOME: ['home', 'house', 'apartment', 'residence'],
            LocationContext.WORK: ['office', 'workplace', 'work', 'company', 'building'],
            LocationContext.MEETING_ROOM: ['meeting room', 'conference room', 'boardroom'],
            LocationContext.CAFE: ['cafe', 'coffee shop', 'restaurant', 'bar'],
            LocationContext.COMMUTE: ['train', 'bus', 'car', 'subway', 'transit'],
            LocationContext.TRAVEL: ['airport', 'hotel', 'station', 'terminal']
        }
    
    async def enrich_location_info(
        self,
        location_data: Optional[Dict[str, Any]]
    ) -> Optional[LocationInfo]:
        """Enrich location information with additional context."""
        if not location_data:
            return None
        
        # Create location info from provided data
        location_info = LocationInfo(
            latitude=location_data.get('latitude'),
            longitude=location_data.get('longitude'),
            name=location_data.get('name'),
            address=location_data.get('address'),
            city=location_data.get('city'),
            country=location_data.get('country'),
            accuracy=location_data.get('accuracy'),
            timezone=location_data.get('timezone')
        )
        
        # Generate cache key
        cache_key = self._generate_location_cache_key(location_info)
        
        # Check cache first
        if self._is_location_cached(cache_key):
            return self.location_cache[cache_key]
        
        # Enhance location info
        enhanced_info = await self._enhance_location_info(location_info)
        
        # Cache the result
        self.location_cache[cache_key] = enhanced_info
        self.cache_timestamps[cache_key] = datetime.now()
        
        return enhanced_info
    
    async def _enhance_location_info(self, location_info: LocationInfo) -> LocationInfo:
        """Enhance location information with reverse geocoding and classification."""
        # Classify location type based on name/address
        if location_info.name:
            location_info.place_type = self._classify_location_type(location_info.name)
        
        # Attempt reverse geocoding if coordinates available
        if (location_info.latitude and location_info.longitude and 
            not location_info.address and REQUESTS_AVAILABLE):
            try:
                enhanced_data = await self._reverse_geocode(
                    location_info.latitude, 
                    location_info.longitude
                )
                
                if enhanced_data:
                    location_info.address = enhanced_data.get('address')
                    location_info.city = enhanced_data.get('city')
                    location_info.country = enhanced_data.get('country')
                    
                    if not location_info.place_type:
                        location_info.place_type = enhanced_data.get('place_type')
                        
            except Exception as e:
                logger.debug(f"Reverse geocoding failed: {e}")
        
        # Determine timezone if not provided
        if not location_info.timezone and PYTZ_AVAILABLE:
            location_info.timezone = self._determine_timezone(location_info)
        
        return location_info
    
    def _classify_location_type(self, location_name: str) -> Optional[str]:
        """Classify location type based on name patterns."""
        name_lower = location_name.lower()
        
        for context, keywords in self.location_patterns.items():
            if any(keyword in name_lower for keyword in keywords):
                return context.value
        
        return None
    
    async def _reverse_geocode(
        self, 
        latitude: float, 
        longitude: float
    ) -> Optional[Dict[str, Any]]:
        """Perform reverse geocoding using a free service."""
        # Using OpenStreetMap Nominatim service (free, rate-limited)
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': latitude,
            'lon': longitude,
            'format': 'json',
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'SecondBrain/1.0'
        }
        
        try:
            # Use asyncio to avoid blocking
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'address': data.get('display_name'),
                            'city': data.get('address', {}).get('city') or data.get('address', {}).get('town'),
                            'country': data.get('address', {}).get('country'),
                            'place_type': data.get('type')
                        }
        except Exception as e:
            logger.debug(f"Reverse geocoding API call failed: {e}")
        
        return None
    
    def _determine_timezone(self, location_info: LocationInfo) -> Optional[str]:
        """Determine timezone from location information."""
        # Simple timezone mapping (could be enhanced with proper timezone lookup)
        country_timezones = {
            'united states': 'America/New_York',
            'canada': 'America/Toronto',
            'united kingdom': 'Europe/London',
            'germany': 'Europe/Berlin',
            'japan': 'Asia/Tokyo',
            'australia': 'Australia/Sydney'
        }
        
        if location_info.country:
            country_lower = location_info.country.lower()
            return country_timezones.get(country_lower)
        
        return None
    
    def _generate_location_cache_key(self, location_info: LocationInfo) -> str:
        """Generate cache key for location information."""
        key_data = {
            'lat': location_info.latitude,
            'lon': location_info.longitude,
            'name': location_info.name,
            'address': location_info.address
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _is_location_cached(self, cache_key: str) -> bool:
        """Check if location is cached and not expired."""
        if cache_key not in self.location_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        return datetime.now() - self.cache_timestamps[cache_key] < self.cache_ttl
    
    def generate_location_tags(self, location_info: LocationInfo) -> List[str]:
        """Generate location-based tags."""
        tags = []
        
        if location_info.place_type:
            tags.append(f"location_{location_info.place_type}")
        
        if location_info.city:
            city_tag = re.sub(r'[^\w\s-]', '', location_info.city.lower())
            city_tag = re.sub(r'\s+', '_', city_tag.strip())
            tags.append(f"city_{city_tag}")
        
        if location_info.country:
            country_tag = re.sub(r'[^\w\s-]', '', location_info.country.lower())
            country_tag = re.sub(r'\s+', '_', country_tag.strip())
            tags.append(f"country_{country_tag}")
        
        # Add contextual location tags
        if location_info.place_type:
            context = LocationContext.UNKNOWN
            for ctx, keywords in self.location_patterns.items():
                if location_info.place_type in keywords:
                    context = ctx
                    break
            
            if context != LocationContext.UNKNOWN:
                tags.append(f"context_{context.value}")
        
        return tags


class TemporalProcessor:
    """Handles time-based processing logic."""
    
    def __init__(self):
        """Initialize temporal processor."""
        self.business_hours_start = time(9, 0)  # 9 AM
        self.business_hours_end = time(17, 0)   # 5 PM
        
        # Holiday detection (simplified)
        self.holidays = {
            (1, 1): "New Year's Day",
            (7, 4): "Independence Day",
            (12, 25): "Christmas Day"
        }
    
    def analyze_temporal_context(
        self,
        timestamp: datetime,
        timezone_name: Optional[str] = None
    ) -> TemporalInfo:
        """Analyze temporal context from timestamp."""
        # Convert to specified timezone if available
        if timezone_name and PYTZ_AVAILABLE:
            try:
                tz = pytz.timezone(timezone_name)
                if timestamp.tzinfo is None:
                    timestamp = tz.localize(timestamp)
                else:
                    timestamp = timestamp.astimezone(tz)
            except Exception as e:
                logger.debug(f"Timezone conversion failed: {e}")
        
        # Determine time context
        time_context = self._determine_time_context(timestamp.time())
        
        # Determine day context
        day_context = self._determine_day_context(timestamp)
        
        # Check if business hours
        is_business_hours = self._is_business_hours(timestamp)
        
        # Determine season (Northern Hemisphere)
        season = self._determine_season(timestamp)
        
        return TemporalInfo(
            timestamp=timestamp,
            timezone=timezone_name,
            time_context=time_context,
            day_context=day_context,
            is_business_hours=is_business_hours,
            season=season
        )
    
    def _determine_time_context(self, time_obj: time) -> TimeContext:
        """Determine time context from time object."""
        hour = time_obj.hour
        
        if 5 <= hour < 8:
            return TimeContext.EARLY_MORNING
        elif 8 <= hour < 12:
            return TimeContext.MORNING
        elif 12 <= hour < 17:
            return TimeContext.AFTERNOON
        elif 17 <= hour < 21:
            return TimeContext.EVENING
        elif 21 <= hour < 24:
            return TimeContext.NIGHT
        else:  # 0 <= hour < 5
            return TimeContext.LATE_NIGHT
    
    def _determine_day_context(self, timestamp: datetime) -> DayContext:
        """Determine day context from timestamp."""
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        
        # Check for holidays
        month_day = (timestamp.month, timestamp.day)
        if month_day in self.holidays:
            return DayContext.HOLIDAY
        
        # Weekend check
        if weekday >= 5:  # Saturday or Sunday
            return DayContext.WEEKEND
        
        return DayContext.WEEKDAY
    
    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during business hours."""
        time_obj = timestamp.time()
        weekday = timestamp.weekday()
        
        # Only weekdays
        if weekday >= 5:
            return False
        
        # Within business hours
        return self.business_hours_start <= time_obj <= self.business_hours_end
    
    def _determine_season(self, timestamp: datetime) -> str:
        """Determine season from timestamp (Northern Hemisphere)."""
        month = timestamp.month
        
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:  # [9, 10, 11]
            return "fall"
    
    def generate_temporal_tags(self, temporal_info: TemporalInfo) -> List[str]:
        """Generate time-based tags."""
        tags = []
        
        if temporal_info.time_context:
            tags.append(f"time_{temporal_info.time_context.value}")
        
        if temporal_info.day_context:
            tags.append(f"day_{temporal_info.day_context.value}")
        
        if temporal_info.is_business_hours:
            tags.append("business_hours")
        else:
            tags.append("after_hours")
        
        if temporal_info.season:
            tags.append(f"season_{temporal_info.season}")
        
        # Add specific date tags for special occasions
        date_tags = self._generate_date_specific_tags(temporal_info.timestamp)
        tags.extend(date_tags)
        
        return tags
    
    def _generate_date_specific_tags(self, timestamp: datetime) -> List[str]:
        """Generate date-specific tags."""
        tags = []
        
        # Year tag
        tags.append(f"year_{timestamp.year}")
        
        # Month tag
        month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']
        tags.append(f"month_{month_names[timestamp.month - 1]}")
        
        # Day of week tag
        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        tags.append(f"day_{day_names[timestamp.weekday()]}")
        
        return tags


class RuleEngine:
    """Contextual processing rule engine."""
    
    def __init__(self, rules_storage_path: Optional[str] = None):
        """Initialize rule engine."""
        self.storage_path = Path(rules_storage_path or settings.base_dir / "config" / "processing_rules")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.rules: Dict[str, ProcessingRule] = {}
        self.load_rules()
        
        # Initialize default rules
        if not self.rules:
            self._create_default_rules()
    
    def load_rules(self):
        """Load processing rules from storage."""
        rules_file = self.storage_path / "rules.json"
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data.get('rules', []):
                    rule = ProcessingRule(**rule_data)
                    self.rules[rule.rule_id] = rule
                
                logger.info(f"Loaded {len(self.rules)} processing rules")
            except Exception as e:
                logger.error(f"Failed to load processing rules: {e}")
    
    def save_rules(self):
        """Save processing rules to storage."""
        rules_file = self.storage_path / "rules.json"
        
        try:
            rules_data = {
                'rules': [asdict(rule) for rule in self.rules.values()],
                'updated_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            with open(rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.rules)} processing rules")
        except Exception as e:
            logger.error(f"Failed to save processing rules: {e}")
    
    def _create_default_rules(self):
        """Create default processing rules."""
        # Work hours rule
        work_hours_rule = ProcessingRule(
            rule_id="work_hours_priority",
            name="Work Hours Priority",
            description="Boost priority for work-related content during business hours",
            time_conditions={'is_business_hours': True},
            source_conditions={'source_type': ['discord', 'email', 'api']},
            processing_modifications={'priority_boost': 1.5},
            tag_additions=['work_hours'],
            priority=10
        )
        self.rules[work_hours_rule.rule_id] = work_hours_rule
        
        # Weekend personal rule
        weekend_rule = ProcessingRule(
            rule_id="weekend_personal",
            name="Weekend Personal Mode",
            description="Enhanced personal processing on weekends",
            time_conditions={'day_context': 'weekend'},
            processing_modifications={
                'processing_strategy': 'enhanced',
                'summary_generation_enabled': True
            },
            tag_additions=['weekend', 'personal_time'],
            priority=20
        )
        self.rules[weekend_rule.rule_id] = weekend_rule
        
        # Location-based work rule
        work_location_rule = ProcessingRule(
            rule_id="work_location_processing",
            name="Work Location Processing",
            description="Work-focused processing when at work location",
            location_conditions={'place_type': 'work'},
            processing_modifications={'action_extraction_enabled': True},
            tag_additions=['at_work', 'professional'],
            priority=15
        )
        self.rules[work_location_rule.rule_id] = work_location_rule
        
        # Mobile source rule
        mobile_rule = ProcessingRule(
            rule_id="mobile_quick_processing",
            name="Mobile Quick Processing",
            description="Fast processing for mobile sources",
            source_conditions={'source_type': ['apple_shortcuts', 'mobile']},
            processing_modifications={
                'processing_strategy': 'standard',
                'priority_boost': 1.2
            },
            tag_additions=['mobile_capture'],
            priority=25
        )
        self.rules[mobile_rule.rule_id] = mobile_rule
        
        self.save_rules()
    
    def evaluate_rules(
        self,
        context: Dict[str, Any],
        location_info: Optional[LocationInfo] = None,
        temporal_info: Optional[TemporalInfo] = None
    ) -> List[ProcessingRule]:
        """Evaluate rules against current context and return matching rules."""
        matching_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if self._evaluate_rule_conditions(rule, context, location_info, temporal_info):
                matching_rules.append(rule)
                rule.usage_count += 1
        
        # Sort by priority (lower number = higher priority)
        matching_rules.sort(key=lambda r: r.priority)
        
        return matching_rules
    
    def _evaluate_rule_conditions(
        self,
        rule: ProcessingRule,
        context: Dict[str, Any],
        location_info: Optional[LocationInfo],
        temporal_info: Optional[TemporalInfo]
    ) -> bool:
        """Evaluate if rule conditions are met."""
        # Time conditions
        if rule.time_conditions and temporal_info:
            if not self._check_time_conditions(rule.time_conditions, temporal_info):
                return False
        
        # Location conditions
        if rule.location_conditions and location_info:
            if not self._check_location_conditions(rule.location_conditions, location_info):
                return False
        
        # Source conditions
        if rule.source_conditions:
            if not self._check_source_conditions(rule.source_conditions, context):
                return False
        
        # Content conditions
        if rule.content_conditions:
            if not self._check_content_conditions(rule.content_conditions, context):
                return False
        
        return True
    
    def _check_time_conditions(
        self,
        conditions: Dict[str, Any],
        temporal_info: TemporalInfo
    ) -> bool:
        """Check if time conditions are met."""
        for key, value in conditions.items():
            if key == 'is_business_hours':
                if temporal_info.is_business_hours != value:
                    return False
            elif key == 'time_context':
                if temporal_info.time_context and temporal_info.time_context.value != value:
                    return False
            elif key == 'day_context':
                if temporal_info.day_context and temporal_info.day_context.value != value:
                    return False
            elif key == 'season':
                if temporal_info.season != value:
                    return False
        
        return True
    
    def _check_location_conditions(
        self,
        conditions: Dict[str, Any],
        location_info: LocationInfo
    ) -> bool:
        """Check if location conditions are met."""
        for key, value in conditions.items():
            if key == 'place_type':
                if location_info.place_type != value:
                    return False
            elif key == 'city':
                if not location_info.city or location_info.city.lower() != value.lower():
                    return False
            elif key == 'country':
                if not location_info.country or location_info.country.lower() != value.lower():
                    return False
        
        return True
    
    def _check_source_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check if source conditions are met."""
        for key, value in conditions.items():
            if key == 'source_type':
                context_value = context.get('source_type')
                if isinstance(value, list):
                    if context_value not in value:
                        return False
                elif context_value != value:
                    return False
        
        return True
    
    def _check_content_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check if content conditions are met."""
        content = context.get('content', '')
        
        for key, value in conditions.items():
            if key == 'min_word_count':
                if len(content.split()) < value:
                    return False
            elif key == 'contains_keywords':
                content_lower = content.lower()
                if isinstance(value, list):
                    if not any(keyword.lower() in content_lower for keyword in value):
                        return False
                elif value.lower() not in content_lower:
                    return False
        
        return True
    
    def add_rule(self, rule: ProcessingRule) -> bool:
        """Add a new processing rule."""
        try:
            self.rules[rule.rule_id] = rule
            self.save_rules()
            logger.info(f"Added processing rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add processing rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a processing rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.save_rules()
            logger.info(f"Removed processing rule: {rule_id}")
            return True
        return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule usage statistics."""
        total_rules = len(self.rules)
        enabled_rules = sum(1 for rule in self.rules.values() if rule.enabled)
        total_usage = sum(rule.usage_count for rule in self.rules.values())
        
        most_used = None
        if self.rules:
            most_used = max(self.rules.values(), key=lambda r: r.usage_count)
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'total_usage': total_usage,
            'most_used_rule': most_used.name if most_used else None,
            'rules': [
                {
                    'id': rule.rule_id,
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'usage_count': rule.usage_count,
                    'priority': rule.priority
                }
                for rule in self.rules.values()
            ]
        }


class SourceProfileManager:
    """Manages source-specific processing profiles."""
    
    def __init__(self, profiles_storage_path: Optional[str] = None):
        """Initialize source profile manager."""
        self.storage_path = Path(profiles_storage_path or settings.base_dir / "config" / "source_profiles")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.profiles: Dict[str, SourceProfile] = {}
        self.load_profiles()
        
        # Initialize default profiles if none exist
        if not self.profiles:
            self._create_default_profiles()
    
    def load_profiles(self):
        """Load source profiles from storage."""
        profiles_file = self.storage_path / "profiles.json"
        
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for profile_data in profiles_data.get('profiles', []):
                    # Handle enum conversion
                    if 'processing_strategy' in profile_data:
                        profile_data['processing_strategy'] = ProcessingStrategy(
                            profile_data['processing_strategy']
                        )
                    
                    profile = SourceProfile(**profile_data)
                    self.profiles[profile.source_type] = profile
                
                logger.info(f"Loaded {len(self.profiles)} source profiles")
            except Exception as e:
                logger.error(f"Failed to load source profiles: {e}")
    
    def save_profiles(self):
        """Save source profiles to storage."""
        profiles_file = self.storage_path / "profiles.json"
        
        try:
            profiles_data = {
                'profiles': [asdict(profile) for profile in self.profiles.values()],
                'updated_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.profiles)} source profiles")
        except Exception as e:
            logger.error(f"Failed to save source profiles: {e}")
    
    def _create_default_profiles(self):
        """Create default source profiles."""
        # Discord profile
        discord_profile = SourceProfile(
            source_type='discord',
            processing_strategy=ProcessingStrategy.ENHANCED,
            priority_boost=1.3,
            enable_ai_enhancement=True,
            action_extraction_enabled=True,
            default_tags=['discord', 'conversation'],
            metadata_enrichments={'platform': 'discord'}
        )
        self.profiles['discord'] = discord_profile
        
        # Apple Shortcuts profile
        ios_profile = SourceProfile(
            source_type='apple_shortcuts',
            processing_strategy=ProcessingStrategy.STANDARD,
            priority_boost=1.5,  # Higher priority for mobile
            enable_location_enrichment=True,
            enable_time_enrichment=True,
            default_tags=['mobile', 'ios'],
            metadata_enrichments={'platform': 'ios'}
        )
        self.profiles['apple_shortcuts'] = ios_profile
        
        # Web UI profile
        web_profile = SourceProfile(
            source_type='web_ui',
            processing_strategy=ProcessingStrategy.STANDARD,
            priority_boost=1.0,
            enable_ai_enhancement=True,
            summary_generation_enabled=True,
            default_tags=['web_ui'],
            metadata_enrichments={'platform': 'web'}
        )
        self.profiles['web_ui'] = web_profile
        
        # API profile
        api_profile = SourceProfile(
            source_type='api',
            processing_strategy=ProcessingStrategy.ENHANCED,
            priority_boost=1.1,
            enable_ai_enhancement=True,
            auto_tag_confidence_threshold=0.6,
            default_tags=['api'],
            metadata_enrichments={'platform': 'api'}
        )
        self.profiles['api'] = api_profile
        
        self.save_profiles()
    
    def get_profile(self, source_type: str) -> SourceProfile:
        """Get source profile, creating default if not found."""
        if source_type in self.profiles:
            return self.profiles[source_type]
        
        # Create default profile for unknown source
        default_profile = SourceProfile(
            source_type=source_type,
            processing_strategy=ProcessingStrategy.STANDARD,
            default_tags=[source_type],
            metadata_enrichments={'platform': source_type}
        )
        
        self.profiles[source_type] = default_profile
        self.save_profiles()
        
        return default_profile
    
    def update_profile(self, source_type: str, updates: Dict[str, Any]) -> bool:
        """Update source profile with new settings."""
        try:
            if source_type not in self.profiles:
                self.profiles[source_type] = SourceProfile(source_type=source_type)
            
            profile = self.profiles[source_type]
            
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            self.save_profiles()
            logger.info(f"Updated source profile: {source_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to update source profile: {e}")
            return False
    
    def get_all_profiles(self) -> Dict[str, SourceProfile]:
        """Get all source profiles."""
        return self.profiles.copy()


class ContextualProcessor:
    """Main contextual processor coordinating all contextual processing."""
    
    def __init__(self):
        """Initialize contextual processor."""
        self.location_enricher = LocationEnricher()
        self.temporal_processor = TemporalProcessor()
        self.rule_engine = RuleEngine()
        self.profile_manager = SourceProfileManager()
        
        logger.info("Contextual Processor initialized")
    
    async def process_context(
        self,
        content: str,
        source_type: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ContextualProcessingResult:
        """
        Process content with contextual awareness.
        
        Args:
            content: Content to process
            source_type: Source type of the content
            context: Optional context information
            user_id: Optional user ID for personalized processing
            
        Returns:
            ContextualProcessingResult: Enhanced processing results
        """
        start_time = datetime.now()
        context = context or {}
        
        logger.info(f"Processing contextual information for {source_type} source")
        
        # Get source profile
        source_profile = self.profile_manager.get_profile(source_type)
        
        # Process location information
        location_info = None
        location_tags = []
        if source_profile.enable_location_enrichment and context.get('location_data'):
            location_info = await self.location_enricher.enrich_location_info(
                context['location_data']
            )
            if location_info:
                location_tags = self.location_enricher.generate_location_tags(location_info)
        
        # Process temporal information
        temporal_info = None
        time_tags = []
        if source_profile.enable_time_enrichment:
            timestamp = context.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            
            timezone_name = None
            if location_info and location_info.timezone:
                timezone_name = location_info.timezone
            
            temporal_info = self.temporal_processor.analyze_temporal_context(
                timestamp, timezone_name
            )
            time_tags = self.temporal_processor.generate_temporal_tags(temporal_info)
        
        # Evaluate processing rules
        evaluation_context = {
            'content': content,
            'source_type': source_type,
            'user_id': user_id,
            **context
        }
        
        matching_rules = self.rule_engine.evaluate_rules(
            evaluation_context, location_info, temporal_info
        )
        
        # Apply rule modifications
        processing_strategy = source_profile.processing_strategy
        priority_adjustment = source_profile.priority_boost
        contextual_tags = list(source_profile.default_tags)
        enriched_metadata = dict(source_profile.metadata_enrichments)
        processing_notes = []
        
        for rule in matching_rules:
            # Apply processing modifications
            if rule.processing_modifications:
                mods = rule.processing_modifications
                
                if 'processing_strategy' in mods:
                    try:
                        processing_strategy = ProcessingStrategy(mods['processing_strategy'])
                    except ValueError:
                        pass
                
                if 'priority_boost' in mods:
                    priority_adjustment *= mods['priority_boost']
            
            # Add rule tags and metadata
            contextual_tags.extend(rule.tag_additions)
            enriched_metadata.update(rule.metadata_additions)
            processing_notes.append(f"Applied rule: {rule.name}")
        
        # Combine all tags
        all_contextual_tags = list(set(contextual_tags + location_tags + time_tags))
        
        # Add context-specific metadata
        if location_info:
            enriched_metadata['location_info'] = {
                'name': location_info.name,
                'city': location_info.city,
                'country': location_info.country,
                'place_type': location_info.place_type,
                'accuracy': location_info.accuracy
            }
        
        if temporal_info:
            enriched_metadata['temporal_info'] = {
                'time_context': temporal_info.time_context.value if temporal_info.time_context else None,
                'day_context': temporal_info.day_context.value if temporal_info.day_context else None,
                'is_business_hours': temporal_info.is_business_hours,
                'season': temporal_info.season,
                'timezone': temporal_info.timezone
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ContextualProcessingResult(
            location_info=location_info,
            temporal_info=temporal_info,
            applied_rules=[rule.rule_id for rule in matching_rules],
            processing_strategy=processing_strategy,
            priority_adjustment=priority_adjustment,
            contextual_tags=all_contextual_tags,
            location_tags=location_tags,
            time_tags=time_tags,
            enriched_metadata=enriched_metadata,
            processing_notes=processing_notes,
            processing_time=processing_time,
            rules_evaluated=len(self.rule_engine.rules),
            location_accuracy=location_info.accuracy if location_info else None
        )
        
        logger.info(f"Contextual processing completed in {processing_time:.2f}s, {len(matching_rules)} rules applied")
        return result
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get contextual processing statistics."""
        rule_stats = self.rule_engine.get_rule_statistics()
        
        return {
            'location_cache_size': len(self.location_enricher.location_cache),
            'rule_statistics': rule_stats,
            'source_profiles_count': len(self.profile_manager.profiles),
            'features': {
                'location_enrichment': True,
                'temporal_processing': True,
                'rule_engine': True,
                'source_profiles': True,
                'reverse_geocoding': REQUESTS_AVAILABLE,
                'timezone_support': PYTZ_AVAILABLE
            }
        }
    
    def get_source_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all source profiles for configuration."""
        profiles = {}
        for source_type, profile in self.profile_manager.get_all_profiles().items():
            profiles[source_type] = asdict(profile)
        return profiles
    
    def update_source_profile(self, source_type: str, updates: Dict[str, Any]) -> bool:
        """Update source profile configuration."""
        return self.profile_manager.update_profile(source_type, updates)
    
    def add_processing_rule(self, rule_data: Dict[str, Any]) -> bool:
        """Add a new processing rule."""
        try:
            rule = ProcessingRule(**rule_data)
            return self.rule_engine.add_rule(rule)
        except Exception as e:
            logger.error(f"Failed to add processing rule: {e}")
            return False
    
    def remove_processing_rule(self, rule_id: str) -> bool:
        """Remove a processing rule."""
        return self.rule_engine.remove_rule(rule_id)
    
    def get_processing_rules(self) -> List[Dict[str, Any]]:
        """Get all processing rules."""
        return [asdict(rule) for rule in self.rule_engine.rules.values()]


# Global service instance
_contextual_processor: Optional[ContextualProcessor] = None


def get_contextual_processor() -> ContextualProcessor:
    """Get the global contextual processor instance."""
    global _contextual_processor
    if _contextual_processor is None:
        _contextual_processor = ContextualProcessor()
    return _contextual_processor