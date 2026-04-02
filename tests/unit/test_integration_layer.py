"""
Unit tests for src/integration/integration_layer.py

Basic test coverage for:
  - Config defaults
  - ResponseCache CRUD operations
  - IntegrationLayer core logic (severity, routing, deduplication)
"""
import json
import pytest
from integration.integration_layer import Config, ResponseCache, IntegrationLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(timestamp=5.0, exercise="squat", mistakes=None, quality_score=0.5):
    """Build a minimal CV frame."""
    return {
        "timestamp_s": timestamp,
        "frame_index": 0,
        "exercise": {"name": exercise, "p": 0.9},
        "mistakes": mistakes or [],
        "metrics": {"rep_count": 3},
        "quality_score": quality_score,
    }


def make_mistake(name="knee_valgus", confidence=0.7):
    """Build a mistake entry."""
    return {"name": name, "p": confidence}


def make_coaching_event(exercise="squat", mistake_type="knee_valgus", severity="medium", quality_score=0.5):
    """Build a coaching event."""
    return {
        "event_id": "test_event_0",
        "timestamp": 10.0,
        "frame_index": 150,
        "exercise": {"name": exercise, "confidence": 0.9},
        "mistake": {
            "type": mistake_type,
            "confidence": 0.6,
            "duration_seconds": 4.0,
            "persistence_rate": 0.4,
            "occurrences": 45,
        },
        "metrics": {},
        "quality_score": quality_score,
        "severity": severity,
        "is_recoaching": False,
        "session_time_minutes": 0.17,
    }


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Test Config defaults."""

    def test_min_persistence_rate(self):
        assert Config.MIN_PERSISTENCE_RATE == 0.30

    def test_min_confidence(self):
        assert Config.MIN_CONFIDENCE == 0.35

    def test_min_duration(self):
        assert Config.MIN_DURATION_SECONDS == 3.0

    def test_re_coaching_threshold(self):
        assert Config.RE_COACHING_THRESHOLD == 45

    def test_coaching_interval(self):
        assert Config.MIN_COACHING_INTERVAL == 10


# ---------------------------------------------------------------------------
# ResponseCache Tests
# ---------------------------------------------------------------------------

class TestResponseCache:
    """Test cache CRUD operations."""

    def test_set_and_get(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        cache.set("key1", "response1", "immediate")
        assert cache.get("key1") == {"response": "response1", "timing": "immediate"}

    def test_get_missing_returns_none(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        assert cache.get("nonexistent") is None

    def test_has_key(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        cache.set("key1", "resp", "rep_end")
        assert cache.has("key1") is True
        assert cache.has("key2") is False

    def test_delete(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        cache.set("key1", "resp", "immediate")
        cache.delete("key1")
        assert cache.has("key1") is False

    def test_clear(self, tmp_path):
        cache = ResponseCache(str(tmp_path))
        cache.set("a", "r1", "immediate")
        cache.set("b", "r2", "immediate")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_populate_defaults_fallback(self, tmp_path):
        """When file missing, fallback defaults are loaded."""
        cache = ResponseCache(str(tmp_path))
        cache.populate_defaults(cache_file="nonexistent.json")
        assert cache.has("squat__knee_valgus")  # hardcoded fallback

    def test_populate_defaults_from_file(self, tmp_path):
        """Load defaults from JSON file."""
        cache_data = {"squat__test": {"response": "Test response", "timing": "immediate"}}
        cache_file = tmp_path / "defaults.json"
        cache_file.write_text(json.dumps(cache_data))

        cache = ResponseCache(str(tmp_path))
        cache.populate_defaults(str(cache_file))
        assert cache.has("squat__test")


# ---------------------------------------------------------------------------
# IntegrationLayer Tests
# ---------------------------------------------------------------------------

class TestIntegrationLayer:
    """Test core IntegrationLayer logic."""

    @pytest.fixture
    def layer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path))
        return IntegrationLayer(session_id="test", config=Config())

    def test_make_cache_key(self, layer):
        key = layer._make_cache_key("squat", "knee valgus")
        assert key == "squat__knee_valgus"

    def test_make_cache_key_normalizes_case(self, layer):
        key = layer._make_cache_key("SQUAT", "KNEE VALGUS")
        assert key == "squat__knee_valgus"

    def test_calculate_severity_high_critical_keyword(self, layer):
        """Critical keywords → high severity."""
        severity = layer._calculate_severity(
            {"name": "knee valgus", "avg_confidence": 0.7},
            make_frame(quality_score=0.5)
        )
        assert severity == "high"

    def test_calculate_severity_medium_form_keyword(self, layer):
        """Form keywords → medium severity."""
        severity = layer._calculate_severity(
            {"name": "incomplete range", "avg_confidence": 0.6},
            make_frame(quality_score=0.6)
        )
        assert severity == "medium"

    def test_calculate_severity_low_generic(self, layer):
        """No keywords, good quality → low severity."""
        severity = layer._calculate_severity(
            {"name": "minor issue", "avg_confidence": 0.5},
            make_frame(quality_score=0.7)
        )
        assert severity == "low"

    def test_is_complex_pattern_low_quality(self, layer):
        """Quality < 0.15 → complex."""
        event = make_coaching_event(quality_score=0.1)
        assert layer._is_complex_pattern(event) is True

    def test_is_complex_pattern_normal_quality(self, layer):
        """Normal quality → not complex."""
        event = make_coaching_event(quality_score=0.5)
        assert layer._is_complex_pattern(event) is False

    def test_select_top_priority_highest_confidence(self, layer):
        """Highest confidence wins."""
        mistakes = [
            {"name": "m1", "avg_confidence": 0.5, "duration": 5, "persistence_rate": 0.4},
            {"name": "m2", "avg_confidence": 0.9, "duration": 3, "persistence_rate": 0.2},
        ]
        top = layer._select_top_priority(mistakes)
        assert top["name"] == "m2"

    def test_route_to_tier_cache_hit(self, layer):
        """Cache hit → tier_1."""
        layer.cache.set("squat__knee_valgus", "Push knees out", "immediate")
        event = make_coaching_event(exercise="squat", mistake_type="knee_valgus", severity="low")
        result = layer._route_to_tier(event)
        assert result["tier"] == "tier_1"

    def test_route_to_tier_default_to_tier2(self, layer):
        """No cache, not recoaching → tier_2."""
        event = make_coaching_event()
        result = layer._route_to_tier(event)
        assert result["tier"] == "tier_2"

    def test_route_to_tier_complex_high_severity_to_tier3(self, layer):
        """Complex + high severity → tier_3."""
        event = make_coaching_event(quality_score=0.1, severity="high")
        result = layer._route_to_tier(event)
        assert result["tier"] == "tier_3"

    def test_should_coach_first_time(self, layer):
        """First coaching of a mistake."""
        mistake = {"name": "knee_valgus"}
        assert layer._should_coach(mistake, make_frame(timestamp=10.0)) is True

    def test_should_coach_within_cooldown(self, layer):
        """Within MIN_COACHING_INTERVAL → False."""
        layer.last_coaching_time = 5.0
        mistake = {"name": "knee_valgus"}
        # 10.0 - 5.0 = 5s < 10s interval
        assert layer._should_coach(mistake, make_frame(timestamp=10.0)) is False

    def test_should_coach_past_cooldown(self, layer):
        """Past MIN_COACHING_INTERVAL → True."""
        layer.last_coaching_time = 0.0
        mistake = {"name": "knee_valgus"}
        # 20.0 - 0.0 = 20s >= 10s interval
        assert layer._should_coach(mistake, make_frame(timestamp=20.0)) is True
