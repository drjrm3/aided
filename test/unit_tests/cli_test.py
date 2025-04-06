"""
cli test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import re
from aided import cli
from .helper import CxTestCase

from unittest.mock import patch
from io import StringIO


class TestCli(CxTestCase):

    def test_parse_args(self):
        """Tests basic command line argument parsing."""

        # Test with no arguments
        args = cli.parse_args([])
        self.assertEqual(args.config, None)

        # Test with config
        args = cli.parse_args(["-c", "foo.json"])
        self.assertEqual(args.config, "foo.json")

        # Test with help command
        with (
            self.assertRaises(SystemExit),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            args = cli.parse_args(["--help"])
        self.assertIn("usage: aided", mock_stdout.getvalue())

        # Test with version command
        with (
            self.assertRaises(SystemExit),
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
        ):
            args = cli.parse_args(["--version"])
        output = mock_stdout.getvalue().strip()
        self.assertRegex(
            output,
            r"^aided \d+\.\d+(\.\d+)?([^\s]*)?$",
            f"Version output did not match expected pattern: {output}",
        )

        # Test with invalid command
        with (
            self.assertRaises(SystemExit),
            patch("sys.stderr", new_callable=StringIO) as mock_stdout,
        ):
            cli.parse_args(["invalid_command"])
        self.assertIn(
            "aided: error: unrecognized arguments: invalid_command", mock_stdout.getvalue()
        )
