"""
Initialization file which provides the main entry point for the package.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from aided.apps._helper import make_register
from .cli import add_arguments

register = make_register(__name__)
