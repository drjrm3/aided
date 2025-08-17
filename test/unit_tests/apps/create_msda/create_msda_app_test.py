"""
aided.apps.create_msda test module
"""

import argparse
import os
from pathlib import Path

from aided.apps.create_msda.main import main
from aided.apps.create_msda.cli import add_arguments

from ut_helper import CxTestCase


class TestCreateMSDAMain(CxTestCase):
    """Tests the aided create_msda main function."""

    def test_simple_main(self):
        """Test simple invocation of main function."""
        log_file = os.path.realpath(f"{self.test_data_dir}/msda/g09/formamide.b3lyp.6311gss.log")
        out_file = f"{self.tmp_dir}/formamide.msda"

        if not Path(log_file).is_file():
            raise FileNotFoundError(f"Test log file not found: {log_file}")

        args = argparse.Namespace(log_file=log_file, T=0, output_file=out_file)

        rc = main(args)
        self.assertEqual(rc, 0)


class TestCreateMSDACLI(CxTestCase):
    """Tests the aided create_msda command line interface."""

    def test_simple_cli(self):
        """Test simple invocation of CLI."""
        parser = argparse.ArgumentParser()

        add_arguments(parser)
        args = parser.parse_args(["-l", "foo.log", "-T", "0", "-o", "foo.msda"])
        self.assertDictEqual(
            args.__dict__, {"log_file": "foo.log", "T": 0, "output_file": "foo.msda"}
        )
