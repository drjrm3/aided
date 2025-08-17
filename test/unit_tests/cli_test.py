"""
aided.cli

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import logging

from unittest.mock import patch
from io import StringIO

from aided import cli

from ut_helper import CxTestCase


class TestCli(CxTestCase):

    def test_no_args(self):
        """Tests the CLI with no arguments."""

        with (
            self.assertRaises(SystemExit),
            patch("sys.stderr", new_callable=StringIO) as mock_stderr,
        ):
            cli.parse_args([])
        output = mock_stderr.getvalue().strip()
        self.assertIn(
            "usage: aided",
            output,
            "Help output did not match expected pattern.",
        )

    def test_bad_app(self):
        """Tests the CLI with an invalid app command."""
        with (
            self.assertRaises(SystemExit),
            patch("sys.stderr", new_callable=StringIO) as mock_stderr,
        ):
            cli.parse_args(["bad_app"])
        output = mock_stderr.getvalue().strip()
        self.assertIn(
            "aided: error: argument command: invalid choice: 'bad_app'",
            output,
            "Help output did not match expected pattern.",
        )

    def test_parse_args(self):
        """Tests the CLI argument parsing using create_msda as an example."""

        argv = ["create-msda", "-l", "inp.log", "-T", "23", "-o", "out.msda"]

        args = cli.parse_args(argv)

        # FIXME: Having trouble testing this. Popping for now.
        args.__dict__.pop("func", None)

        self.assertDictEqual(
            args.__dict__,
            {
                "config": None,
                "log_file": "inp.log",
                "log_level": "info",
                "command": "create-msda",
                "output_file": "out.msda",
                "T": 23.0,
            },
        )

        # This ends up creating a logger, we must remove it to keep other tests
        # clean which test for caching of the logger.
        logging.Logger.manager.loggerDict.pop("aided", None)
