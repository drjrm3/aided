"""
aided.apps._helper

Utility that builds a `register(subparsers)` function for each app.

Each `apps/<app_name>/__init__.py` should be of the form:

    from aided.apps._helper import make_register
    register = make_register(__name__)
"""

from argparse import _SubParsersAction
from importlib import import_module
from types import ModuleType
from typing import Callable


def _derive_command_name(app_module: str) -> str:
    """Derives the command name from the app package name.

    Args:
        app_module (str): The name of the app package, e.g. "aided.apps.foo".

    Returns:
        app_name (str): The command name derived from the app package name, e.g. "foo".
    """

    app_name = app_module.split(".", -1)[-1].replace("_", "-")

    return app_name


def _load_main(app_module: str) -> ModuleType:
    """Loads the main entrypoint module for an app package name.

    Args:
        app_module (str): The name of the app package, e.g. "aided.apps.foo".

    Returns:
        main (ModuleType): The loaded main module for the app package.
    """

    main = import_module(app_module + ".main")

    return main


def make_register(app_name: str) -> Callable[[_SubParsersAction], None]:
    """Returns a "register" function for package "app_name"

    Args:
        app_name (str): The name of the app package, e.g. "aided.apps.foo".

    Returns:
        register (Callable): A function that registers the app's command in the given subparsers.
    """

    # Load the main module for the app package.
    main = _load_main(app_name)

    # Derive the command name from the app package name.
    command = _derive_command_name(app_name)

    # Get the docstring for the main function, if available.
    doc = (main.__doc__ or "").strip()

    def register(subparsers: _SubParsersAction) -> None:
        """Registers the app's command in the given subparsers.

        Args:
            subparsers (_SubParsersAction): The subparsers to register the command in.
        """

        main_first_line = doc.splitlines()[0] if doc else f"No help available for {command}"

        # TODO: Add a description entry?
        parser = subparsers.add_parser(command, help=main_first_line)

        if hasattr(main, "add_arguments"):
            main.add_arguments(parser)

        parser.set_defaults(func=main.main)

    return register
