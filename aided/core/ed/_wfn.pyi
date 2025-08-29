"""Stub file for pybind11 interface.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import Tuple, Any
import numpy as np
import numpy.typing as npt

# Disable many pylint warnings for this stub file.
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=unused-argument
# pylint: disable=trailing-whitespace
def gen_gs(
    x: float,
    y: float, 
    z: float,
    ider: int,
    last_der: Any,
    last_point: Any,
    types: npt.NDArray[np.int32],
    centers: npt.NDArray[np.int32],
    expons: npt.NDArray[np.float64],
    atpos: npt.NDArray[np.float64],
    gs: npt.NDArray[np.float64],
    gs1: npt.NDArray[np.float64],
    gs2: npt.NDArray[np.float64]
) -> Tuple[bool, Any, Any]: ...
