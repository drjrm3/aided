"""
aided.cli

Commandline utilities and main method call.

Automically discovers top level application modules located in the "apps" directory to be called and
turns them into a sub-command:

    aided <command> [<options>]

Each command module must expose "register(subparsers)", which adds its own arg-parser and sets
"func" to dispatch to it.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import argparse
import importlib
import sys
import pkgutil
from typing import List
from pathlib import Path

from aided import get_logger

from .version import __version__

_APPS_DIR = str(Path(__file__).parent / "apps")


def _discover_apps(subparsers: argparse._SubParsersAction) -> None:
    """Import every module in aided.apps (except those start with "_").

    Args:
        subparsers (argparse._SubParsersAction): Subparsers to add commands to.
    """

    # Get the package name for dynamic imports (e.g., "aided.apps").
    package_name = __name__.rsplit(".", 1)[0] + ".apps"

    for app in pkgutil.iter_modules([str(_APPS_DIR)]):
        if app.name.startswith("_"):
            # Ignore private helpers like _foo.py and __init__.py.
            continue

        # Import the module dynamically.
        module = importlib.import_module(f"{package_name}.{app.name}")

        # Try to get the "register" function.
        register = getattr(module, "register", None)
        if callable(register):
            register(subparsers)


def root_parser():
    """Build root ArgumentParser to be shared by all commands.

    Args:
        argv (List[str]): List of commandline arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog="aided", description="AIDED - Assisted Analysis of the Dynamic Electron Density"
    )
    parser.add_argument("-c", "--config", type=str, help="JSON config file.")
    parser.add_argument("-v", "--version", action="version", version=f"aided {__version__}")
    parser.add_argument(
        "-l", "--log_file", type=str, default="aided.log", help="Log file to write to."
    )
    parser.add_argument(
        "-V",
        dest="log_level",
        type=str,
        help="Verbosity level (debug, info, warning, error).",
        choices=["debug", "info", "warning", "error"],
        default="info",
    )

    return parser


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse arguments for main entry point call to aided

    Args:
        argv (List[str] | None): List of commandline arguments. Defaults to None.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """

    root = root_parser()
    subparsers = root.add_subparsers(dest="command", required=True)

    # Adds one sub-parser per file in aided/apps
    _discover_apps(subparsers)

    args = root.parse_args(argv)

    # Get the global logger once to set up logging configuration.
    get_logger(args.log_level, args.log_file, initialize=True)

    return args


def main() -> int:  # pragma: no cover
    """Main entry point for aided commandline interface.

    Returns:
        return_code (int): Return code for the command executed.
    """

    args = parse_args(sys.argv[1:])

    return_code = args.func(args)

    return return_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
