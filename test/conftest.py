"""
Pytest configuration for aided test suite.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import sys
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = Path(__file__).parent
UNIT_TESTS_DIR = TEST_DIR / "unit_tests"

# Define paths as constants
STATIC_VALIDATION_FILE = TEST_DIR / "validation" / "static_validation.txt"
TEST_DATA_DIR = TEST_DIR / "data"
WFN_FILES_DIR = TEST_DATA_DIR / "wfns"
MSDA_FILES_DIR = TEST_DATA_DIR / "msda"

# Add to path
for path in [PROJECT_ROOT, UNIT_TESTS_DIR]:
    PATH_STR = str(path)
    if PATH_STR not in sys.path:
        sys.path.insert(0, PATH_STR)
