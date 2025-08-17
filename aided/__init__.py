"""
aided

Analysis and Investigation of the Dynamic Electron Density

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

# pylint: disable=wrong-import-position
# Suppress denormal warnings that occur on systems with FTZ enabled
import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

#import warnings
# Suppress denormal warnings that occur on systems with FTZ enabled
#warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

# Intentionally importing this here so that we can swap it out with cupy, cupynumeric, etc.
import numpy as np

# import cupynumeric as np
import numpy.typing as npt

from .core import EDWfn, EDWfns
from .io._logger import _get_logger as get_logger
