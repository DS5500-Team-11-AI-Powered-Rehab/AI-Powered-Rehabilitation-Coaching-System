"""
Shared fixtures for unit tests.
Adds src/ to sys.path so tests can import from integration, agents, etc.
"""
import sys
from pathlib import Path

# Resolve src/ relative to this file so tests work from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
