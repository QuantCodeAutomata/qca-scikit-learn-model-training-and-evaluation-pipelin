"""
Pytest configuration and shared fixtures for the experiment test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is importable from tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fix numpy random seed for reproducibility across all tests."""
    np.random.seed(42)
