"""
Shared fixtures for unit tests.
Adds src/ to sys.path so tests can import from integration, agents, etc.
"""
import sys
from pathlib import Path

# Resolve src/ relative to this file so tests work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# coaching_agent uses internal imports like `from session_prompts import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "agents" / "coaching_agent"))

# upstream_adapter uses `from progress_tracker_agent.schemas import ...`
# so src/agents (the *parent* of the package dir) must be on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "agents"))
