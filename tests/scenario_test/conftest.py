"""
Shared setup for scenario tests.

Provides:
  - sys.path wiring so 'integration' module is importable
  - SmallConfig  (10-frame warm-up, 20-frame window) for fast test runs
  - make_scenario_frame()  builds a realistic CV frame for a given exercise/mistake
  - run_scenario()         feeds a frame list through IntegrationLayer and returns first event
  - layer()                pytest fixture — fresh IntegrationLayer per test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pytest
from integration.integration_layer import Config, IntegrationLayer


class SmallConfig(Config):
    """Reduced window size so tests don't need to feed 75 frames."""
    SOURCE_FPS = 5
    WINDOW_SIZE_SECONDS = 4
    WINDOW_SIZE_FRAMES = 20
    MIN_FRAMES = 10


def make_scenario_frame(
    exercise: str,
    mistake_name: str,
    frame_index: int,
    confidence: float = 0.75,
    quality_score: float = 0.6,
) -> dict:
    """
    Build a single CV output frame for a scenario.

    timestamp advances by 0.5 s per frame, so 15 frames spans 7 s
    (well above MIN_DURATION_SECONDS = 3.0).
    """
    return {
        "timestamp_s": frame_index * 0.5,
        "frame_index": frame_index,
        "exercise": {"name": exercise, "p": 0.9},
        "mistakes": [{"name": mistake_name, "p": confidence}],
        "metrics": {"rep_count": frame_index // 5 + 1},
        "quality_score": quality_score,
    }


def make_clean_frame(exercise: str, frame_index: int, quality_score: float = 0.8) -> dict:
    """Build a CV frame with no mistakes (good form)."""
    return {
        "timestamp_s": frame_index * 0.5,
        "frame_index": frame_index,
        "exercise": {"name": exercise, "p": 0.9},
        "mistakes": [],
        "metrics": {"rep_count": frame_index // 5 + 1},
        "quality_score": quality_score,
    }


def run_scenario(layer: IntegrationLayer, frames: list) -> dict | None:
    """Feed all frames through the layer and return the first coaching event, or None."""
    for frame in frames:
        event = layer.process_frame(frame)
        if event is not None:
            return event
    return None


@pytest.fixture
def layer(tmp_path, monkeypatch):
    """Fresh IntegrationLayer with isolated cache and small window config."""
    monkeypatch.setattr(Config, "CACHE_DIR", str(tmp_path / "cache"))
    return IntegrationLayer(session_id="scenario-test", config=SmallConfig())
