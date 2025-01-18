"""
aided.core.units

Copyright (C) J. Robert Michael, PhD
"""

from enum import Enum

class Units(Enum):
    """
    Units to use for distance / properties.
        - BOHR = au : Used internally for all calculations.
        - ANGSTROM = 10e-10 : Optionally can be read in or printed out in this format.
    """
    # Atomic Units
    BOHR = 0
    # Angstrom
    ANG = 1

AU_TO_ANG = 0.52917721090380
ANG_TO_AU = 1.88972612545782
