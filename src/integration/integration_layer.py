"""
Integration Layer: Unified preprocessing for CV → LLM pipeline
Handles: Aggregation + Memory + Routing + Cache Management
"""

from collections import deque
from typing import Optional, Dict, List, Set
from statistics import mean
import time

try:
    import diskcache
except ImportError:
    diskcache = None

class Config:
    """Integration layer configuration"""
    
    # Temporal Filtering (Aggregation)
    SOURCE_FPS = 15
    WINDOW_SIZE_SECONDS = 10
    WINDOW_SIZE_FRAMES = SOURCE_FPS * WINDOW_SIZE_SECONDS  # 150 frames
    MIN_FRAMES = SOURCE_FPS * 5  # Need 5 seconds of history (75 frames)
    
    MIN_PERSISTENCE_RATE = 0.30  # Appear in 30%+ of frames
    MIN_CONFIDENCE = 0.35        # Average confidence threshold
    MIN_DURATION_SECONDS = 3.0   # Must persist for 3+ seconds
    
    # Deduplication
    MIN_COACHING_INTERVAL = 10   # Seconds between any coaching feedbacks
    RE_COACHING_THRESHOLD = 20   # Re-coach if persists 20+ seconds after first coaching
    
    # Everity Classification
    CRITICAL_KEYWORDS = [
        'not moving',
        'knee valgus',
        'lumbar',
        'pain',
        'dangerous'
    ]
    
    FORM_KEYWORDS = [
        'twisting',
        'fast',
        'incomplete',
        'range',
        'arm raise',
        'lean'
    ]
    
    # Cache Settings
    CACHE_DIR = "./cache/tier1_responses"
    CACHE_TTL = 86400 * 7  # 7 days

class ResponseCache:
    """
    Manages Tier 1 cached responses with optional disk persistence.
    Falls back to in-memory cache when diskcache is unavailable.
    """

    def __init__(self, cache_dir: str, ttl: int = Config.CACHE_TTL):
        self.ttl = ttl
        self._memory_cache = {}
        self._use_diskcache = diskcache is not None
        if self._use_diskcache:
            self.cache = diskcache.Cache(cache_dir)
        else:
            self.cache = self._memory_cache
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """
        Lookup cached response
        Returns: {"response": str, "timing": str} or None
        """
        return self.cache.get(cache_key)
    
    def set(self, cache_key: str, response: str, timing: str = "immediate"):
        """
        Add/update cached response

        Usage:
            cache.set("heel_lift_not_moving_up",
                     "Pause - are you experiencing pain?",
                     timing="immediate")
        """
        value = {"response": response, "timing": timing}
        if self._use_diskcache:
            self.cache.set(cache_key, value, expire=self.ttl)
        else:
            self.cache[cache_key] = value
    
    def has(self, cache_key: str) -> bool:
        """Check if key exists in cache"""
        return cache_key in self.cache
    
    def delete(self, cache_key: str):
        """Remove cached response"""
        if self._use_diskcache:
            self.cache.delete(cache_key)
        else:
            self.cache.pop(cache_key, None)
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
    
    def list_all(self) -> List[str]:
        """List all cached keys"""
        if self._use_diskcache:
            return list(self.cache.iterkeys())
        return list(self.cache.keys())
    
    def populate_defaults(self):
        """
        Populate cache with common patterns
        Call this once during initialization
        
        TODO: Add your top 20 common mistake patterns here
        """
        defaults = {
            "heel_lift_not_moving_up": {
                "response": "Pause - are you experiencing any pain? Stop if uncomfortable.",
                "timing": "immediate"
            },
            "squat_knee_valgus": {
                "response": "Push your knees outward over your toes.",
                "timing": "immediate"
            },
            "jumping_jacks_incomplete_arm_raise": {
                "response": "Raise your arms fully overhead on each jump.",
                "timing": "rep_end"
            },
            # TODO: Add more patterns as you discover them
        }
        
        for key, data in defaults.items():
            self.set(key, data["response"], data["timing"])


class IntegrationLayer:
    """
    Unified integration layer: Aggregation + Memory + Routing
    
    Usage:
        layer = IntegrationLayer(session_id="session_123")
        layer.cache.populate_defaults()
        
        for cv_frame in cv_stream:
            coaching_event = layer.process_frame(cv_frame)
            if coaching_event:
                # Send to LangGraph...
    """
    
    def __init__(self, session_id: str, config: Config = None):
        """Initialize integration layer for a session"""
        self.session_id = session_id
        self.config = config or Config()
        
        # Aggregation State
        self.frame_buffer = deque(maxlen=self.config.WINDOW_SIZE_FRAMES)
        self.event_counter = 0
        
        # Memory State
        self.coached_mistakes = set()  # Mistake types we've addressed
        self.coaching_history = []     # Full history with timestamps
        self.last_coaching_time = -1000  # Initialize to large negative value (never coached before)
        
        # Cache Memory
        self.cache = ResponseCache(self.config.CACHE_DIR)
    
    
    def process_frame(self, cv_frame: Dict) -> Optional[Dict]:
        """
        Called for each CV frame
        
        Process:
        1. Add frame to sliding window buffer
        2. Find persistent mistakes (temporal filtering)
        3. Check if any meet coaching threshold
        4. Check deduplication (already coached?)
        5. Route to tier
        6. Return coaching event OR None
        
        Args:
            cv_frame: CV output JSON frame
            
        Returns:
            Coaching event dict OR None if no coaching needed
        """
        
        # Add to sliding window
        self.frame_buffer.append(cv_frame)
        
        # Need minimum history to analyze
        if len(self.frame_buffer) < self.config.MIN_FRAMES:
            return None
        
        # Step 1: Aggregation - Find persistent mistakes
        persistent_mistakes = self._find_persistent_mistakes()
        
        if not persistent_mistakes:
            return None
        
        # Step 2: Deduplication - Check if we should coach
        coachable_mistakes = [
            m for m in persistent_mistakes 
            if self._should_coach(m, cv_frame)
        ]
        
        if not coachable_mistakes:
            return None
        
        # Take highest priority mistake
        top_mistake = self._select_top_priority(coachable_mistakes)
        
        # Step 3: Create Event - Build structured coaching event
        coaching_event = self._create_coaching_event(top_mistake, cv_frame)
        
        # Step 4: Routing - Decide which tier
        routing_info = self._route_to_tier(coaching_event)
        coaching_event.update(routing_info)
        
        # Step 5: Record - Mark as coached (prevent duplicate)
        self._record_coaching_intent(coaching_event)
        
        return coaching_event
    
    def _find_persistent_mistakes(self) -> List[Dict]:
        """
        Temporal Filtering Logic
        
        Scan sliding window to find mistakes that:
        - Appear in 30%+ of frames (persistent)
        - Have average confidence > 0.35
        - Duration > 3 seconds
        
        Returns list of persistent mistake dicts
        """
        mistake_tracker = {}
        
        for frame in self.frame_buffer:
            for mistake in frame.get('mistakes', []):
                name = mistake['name']
                confidence = mistake['p']
                
                if name not in mistake_tracker:
                    mistake_tracker[name] = {
                        'name': name,
                        'occurrences': 0,
                        'confidences': [],
                        'first_seen': frame['timestamp_s'],
                        'last_seen': frame['timestamp_s']
                    }
                
                mistake_tracker[name]['occurrences'] += 1
                mistake_tracker[name]['confidences'].append(confidence)
                mistake_tracker[name]['last_seen'] = frame['timestamp_s']
        
        # Filter for persistent patterns
        persistent = []
        window_size = len(self.frame_buffer)
        
        for name, data in mistake_tracker.items():
            persistence_rate = data['occurrences'] / window_size
            avg_confidence = mean(data['confidences'])
            duration = data['last_seen'] - data['first_seen']
            
            # Check thresholds
            if (persistence_rate >= self.config.MIN_PERSISTENCE_RATE and
                avg_confidence >= self.config.MIN_CONFIDENCE and
                duration >= self.config.MIN_DURATION_SECONDS):
                
                persistent.append({
                    'name': name,
                    'persistence_rate': persistence_rate,
                    'avg_confidence': avg_confidence,
                    'duration': duration,
                    'occurrences': data['occurrences']
                })
        
        return persistent
    
    def _should_coach(self, mistake: Dict, current_frame: Dict) -> bool:
        """
        Deduplication Check
        
        Decide if we should coach on this mistake:
        - NOT already coached in this session?
        - Enough time since last coaching (cooldown)?
        - OR is this a re-coaching situation (persists 20s+)?
        
        Returns True if should coach, False if skip
        """
        
        mistake_type = mistake['name']
        current_time = current_frame['timestamp_s']
        
        # Check cooldown period
        time_since_last = current_time - self.last_coaching_time
        if time_since_last < self.config.MIN_COACHING_INTERVAL:
            return False  # Too soon after last feedback
        
        # First time seeing this mistake?
        if mistake_type not in self.coached_mistakes:
            return True  # Definitely coach
        
        # Already coached - check if we should re-coach
        return self._should_re_coach(mistake_type, current_time)
    
    def _should_re_coach(self, mistake_type: str, current_time: float) -> bool:
        """
        Re-coaching Logic
        
        If same mistake persists 20+ seconds after first coaching:
        → Re-coach with escalation
        """
        first_coached = next(
            (entry for entry in self.coaching_history 
             if entry['mistake_type'] == mistake_type),
            None
        )
        
        if not first_coached:
            return False
        
        time_since_coaching = current_time - first_coached['timestamp']
        
        if time_since_coaching >= self.config.RE_COACHING_THRESHOLD:
            return True
        
        return False
    
    def _select_top_priority(self, mistakes: List[Dict]) -> Dict:
        """
        If multiple mistakes are coachable, pick highest priority:
        1. Highest average confidence
        2. Longest duration
        3. Highest persistence rate
        """
        return max(
            mistakes,
            key=lambda m: (
                m['avg_confidence'],
                m['duration'],
                m['persistence_rate']
            )
        )
    
    
    # ==========================================
    # EVENT CREATION
    # ==========================================
    
    def _create_coaching_event(self, mistake: Dict, current_frame: Dict) -> Dict:
        """
        Building Coaching Event
        
        Transform aggregated mistake data + CV frame 
        → Structured coaching event for LangGraph
        """
        
        # Calculate severity
        severity = self._calculate_severity(mistake, current_frame)
        
        # Check if this is re-coaching
        is_recoaching = mistake['name'] in self.coached_mistakes
        
        # Build structured event
        coaching_event = {
            "event_id": f"{self.session_id}_event_{self.event_counter}",
            "timestamp": current_frame['timestamp_s'],
            "frame_index": current_frame['frame_index'],
            
            # Exercise context
            "exercise": {
                "name": current_frame['exercise']['name'],
                "confidence": current_frame['exercise']['p']
            },
            
            # Mistake details
            "mistake": {
                "type": mistake['name'],
                "confidence": mistake['avg_confidence'],
                "duration_seconds": mistake['duration'],
                "persistence_rate": mistake['persistence_rate'],
                "occurrences": mistake['occurrences']
            },
            
            # CV metrics
            "metrics": current_frame['metrics'].copy(),
            "quality_score": current_frame.get('quality_score', 0.5),
            
            # Severity assessment
            "severity": severity,
            
            # Context flags
            "is_recoaching": is_recoaching,
            "session_time_minutes": current_frame['timestamp_s'] / 60
        }
        
        self.event_counter += 1
        return coaching_event
    
    def _calculate_severity(self, mistake: Dict, frame: Dict) -> str:
        """
        Severity Classification Logic
        
        Rules:
        - Safety-critical patterns → HIGH
        - Form errors + low quality → MEDIUM
        - Optimization issues → LOW
        
        Returns: "high", "medium", or "low"
        """
        
        mistake_name = mistake['name'].lower()
        quality_score = frame.get('quality_score', 0.5)
        confidence = mistake['avg_confidence']
        
        is_critical = any(
            keyword in mistake_name 
            for keyword in self.config.CRITICAL_KEYWORDS
        )
        
        is_form_issue = any(
            keyword in mistake_name 
            for keyword in self.config.FORM_KEYWORDS
        )
        
        # SEVERITY DECISION TREE
        if is_critical or (quality_score < 0.3 and confidence > 0.4):
            return "high"
        elif is_form_issue or quality_score < 0.4:
            return "medium"
        else:
            return "low"
    
    
    # ==========================================
    # ROUTING LOGIC
    # ==========================================
    
    def _route_to_tier(self, coaching_event: Dict) -> Dict:
        """
        Tier Routing Decision
        
        Decision tree:
        1. IF in cache AND not re-coaching → TIER 1 (50ms)
        2. ELIF high/medium severity → TIER 2 (1-2s)
        3. ELIF complex pattern OR re-coaching → TIER 3 (3-5s)
        4. ELSE → TIER 2 (default)
        
        Returns dict with: {"tier", "cache_key", "routing_reason"}
        """
        
        exercise = coaching_event['exercise']['name']
        mistake = coaching_event['mistake']['type']
        severity = coaching_event['severity']
        is_recoaching = coaching_event['is_recoaching']
        
        # Generate cache key
        cache_key = self._make_cache_key(exercise, mistake)
        
        # === TIER 1: Cached responses ===
        if self.cache.has(cache_key) and not is_recoaching:
            return {
                "tier": "tier_1",
                "cache_key": cache_key,
                "routing_reason": "Common mistake with cached response"
            }
        
        # === TIER 3: Complex situations ===
        if is_recoaching or self._is_complex_pattern(coaching_event):
            return {
                "tier": "tier_3",
                "cache_key": None,
                "routing_reason": (
                    "Re-coaching situation" if is_recoaching 
                    else "Complex pattern requiring full reasoning"
                )
            }
        
        # === TIER 2: Standard RAG + LLM ===
        return {
            "tier": "tier_2",
            "cache_key": None,
            "routing_reason": f"{severity} severity mistake needs RAG context"
        }
    
    def _is_complex_pattern(self, coaching_event: Dict) -> bool:
        """
        Complex Pattern Detection
        
        Indicators:
        - Very low quality score (<0.3)
        - High persistence (>60%) but low confidence (<0.5)
        """
        quality = coaching_event['quality_score']
        persistence = coaching_event['mistake']['persistence_rate']
        confidence = coaching_event['mistake']['confidence']
        
        return (
            quality < 0.3 or
            (persistence > 0.6 and confidence < 0.5)
        )
    
    def _make_cache_key(self, exercise: str, mistake: str) -> str:
        """
        Generate cache key: "exercise_mistake_type"
        Example: "heel_lift_not_moving_up"
        """
        return f"{exercise}_{mistake}".replace(" ", "_").replace("-", "_").lower()
    
    def _record_coaching_intent(self, coaching_event: Dict):
        """
        RECORD that we're about to coach on this mistake
        
        Update:
        - coached_mistakes set (for deduplication)
        - last_coaching_time (for cooldown)
        """
        mistake_type = coaching_event['mistake']['type']
        timestamp = coaching_event['timestamp']
        
        self.coached_mistakes.add(mistake_type)
        self.last_coaching_time = timestamp
    
    def record_coaching_complete(self, coaching_event: Dict, 
                                  response: str, tier: str):
        """
        PUBLIC METHOD - Called after LangGraph finishes
        
        Record full coaching history entry
        """
        self.coaching_history.append({
            'timestamp': coaching_event['timestamp'],
            'mistake_type': coaching_event['mistake']['type'],
            'response': response,
            'tier_used': tier,
            'severity': coaching_event['severity'],
            'event_id': coaching_event['event_id']
        })
    
    def add_cached_response(self, exercise: str, mistake: str, 
                           response: str, timing: str = "immediate"):
        """
        Dynamically add new cached response
        
        Usage:
            layer.add_cached_response(
                "squat", 
                "knee valgus",
                "Push your knees outward",
                timing="immediate"
            )
        """
        cache_key = self._make_cache_key(exercise, mistake)
        self.cache.set(cache_key, response, timing)
    
    def update_cached_response(self, exercise: str, mistake: str, response: str):
        """Update existing cached response"""
        self.add_cached_response(exercise, mistake, response)
    
    def remove_cached_response(self, exercise: str, mistake: str):
        """Remove cached response"""
        cache_key = self._make_cache_key(exercise, mistake)
        self.cache.delete(cache_key)
    
    def list_cached_patterns(self) -> List[str]:
        """List all cached patterns"""
        return self.cache.list_all()
    
    def get_session_summary(self) -> Dict:
        """Get session summary for Progress Agent"""
        duration_seconds = self.last_coaching_time if self.event_counter > 0 else 0.0
        return {
            'session_id': self.session_id,
            'total_events': self.event_counter,
            'coached_mistakes': list(self.coached_mistakes),
            'coaching_history': self.coaching_history,
            'session_duration_seconds': duration_seconds
        }
    
    def reset_session(self):
        """Reset session state (call between sessions)"""
        self.coached_mistakes.clear()
        self.coaching_history.clear()
        self.last_coaching_time = 0
        self.event_counter = 0