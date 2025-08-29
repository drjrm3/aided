"""
io.logger test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import logging

from aided.core.ed.base import EDType
from aided import get_logger

from ut_helper import CxTestCase


class TestLogger(CxTestCase):

    def test_logger_initialization(self):
        """Tests logger with initialization."""

        logger = get_logger("warn", self.tmp_dir + "tmp.log", initialize=True)

        self.assertEqual(logger.level, logging.WARN)

    def test_logger_cached(self):
        """Tests logger with cached initialization."""

        logger1 = get_logger()
        self.assertEqual(logger1.level, logging.NOTSET)

        logger2 = get_logger("warn", self.tmp_dir + "tmp.log", initialize=True)
        self.assertEqual(logger2.level, logging.WARN)

        logger3 = get_logger()
        self.assertEqual(logger3.level, logging.WARN)
