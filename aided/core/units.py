"""
aided.core.units

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from enum import Enum
from numpy import pi as np_pi


# fmt: off
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

# Physical constants
HBAR = 1.054_571_817e-34     # Reduced Planck constant, J·s
KB   = 1.380_649e-23         # Boltzmann constant, J/K
C_CM_S = 2.997_924_58e10     # Speed of light, cm/s
AMU  = 1.660_539_066_60e-27  # Atomic mass unit, kg

# Conversion factors
AU_TO_ANG = 0.52917721090380     # Bohr to Angstrom conversion
ANG_TO_AU = 1.88972612545782     # Angstrom to Bohr conversion
ANG_TO_M  = 1e-10                # Angstrom to meter conversion
AMU_TO_KG = 1.660_539_066_60e-27 # Atomic mass unit to kg conversion
ANG2_TO_M2 = 1e-20               # Angstrom squared to meter squared conversion

################################################################################
# ************************************************************************** ###
# ***             Conversion factors used in PhD work.                   *** ###
# ************************************************************************** ###
################################################################################
# NOTE: This section was generated with extensive conversations with ChatGPT ###
# to assist in descriptively documenting the derivations of these constants. ###
# from my PhD work. The comments are verbose to ensure clarity.              ###
################################################################################
# ------------------------------------------------------------------------------
# "aa"  –  converts the exact coth argument  ħ ω / 2kₑT
#          into the format  AA · ν / T  used by Gaussian
#          ( ν in cm⁻¹, T in kelvin ).
# ------------------------------------------------------------------------------
# Step-by-step transformation:
#
#   ω  (rad·s⁻¹)        →  2π c ν              (insert 2π and speed of light)
#   ħω / 2kᴮT           replace ω              (introduce ν)
#   ↓
#   (ħ π c / kᴮ) · ν/T
#
# Everything in brackets is a **constant with units cm·K**.
THERM_FACTOR_CM_K = HBAR * np_pi * C_CM_S / KB   # 0.719385043 …  cm·K
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# "BB"  –  gathers all unit factors in the zero-point prefactor
#          [ħ / (2ω)] / μ  and expresses it in
#          Å² · cm · amu   (Gaussian’s favourite mix).
# ------------------------------------------------------------------------------
# Pieces to include:
#
#   1) ħ / (2ω)                – SI     (J·s / s⁻¹  =  J)
#          → divide by (2π c ν) to replace ω with ν  (cm⁻¹)
#
#   2) convert   J  →  Å²·kg·cm⁻²·s⁻²
#          – multiply by   1 / (1 Å² in m²)          (× 10²⁰)
#
#   3) divide by the reduced mass μ_s **in amu**
#          – so multiply by  1 / AMU_TO_KG
#
# The constant part emerging from (1)…(3) is:
ZPE_PREF_ANG2 = (
      HBAR                              # ħ
    / (4.0 * np_pi * C_CM_S)            # 1 / (2 ω) with ω = 2π c ν
    / AMU_TO_KG                         # kg  →  amu
    / ANG2_TO_M2                        # m²  →  Å²
)
# numeric value: 16.857629181311183  Å²·cm·amu
# ------------------------------------------------------------------------------
