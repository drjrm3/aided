"""
aided

Analysis and Investigation of the Dynamic Electron Density

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

# Intentionally importing this here so that we can swap it out with cupy, cupynumeric, etc.
import numpy as np

# import cupynumeric as np
import numpy.typing as npt

# fmt: off
# Integer exponents "l,m,n" for spherical harmonics.
LMNS = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
    (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1),
    (0, 2, 1), (1, 2, 0), (1, 0, 2), (0, 1, 2), (1, 1, 1),
    (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1),
    (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0),
    (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2)
]
# fmt: on
