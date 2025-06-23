#!/usr/bin/env python3
"""
Sympy notebook for dynamic ED calculations.

Scheringer defines:

    - A = alpha . I3
    - B = beta . I3
    - G = A + B
    - a = (ax, ay, az) as the center of GTO centered position a.
    - b = (bx, by, bz) as the center of GTO centered position b.
    - x as the position vector (x, y, z).
        - We use r = (x, y, z) as the position vector.

* Scheringer also defines a GTO as g(r) = exp(-0.5 alpha |r - R|**2).
    However, we use g(r) = exp(-alpha |r - R|**2) to avoid the 0.5 factor since this factor is
    contracted into the exponent arguments.

"""

import sympy as sp

from sympy import symbols, init_printing
from sympy import Matrix, Float

def main():

    # Define symbols for the position vector.
    x, y, z = symbols('x y z', real=True)
    r = Matrix([x, y, z])

    # Define the center of GTO centered position a and b.
    ax, ay, az = symbols('ax ay az', real=True)
    a = Matrix([ax, ay, az])
    bx, by, bz = symbols('bx by bz', real=True)
    b = Matrix([bx, by, bz])

    # Build the matrices
    al, be = symbols('al be', real=True, positive=True)
    ga = 1. / (al + be)
    I3 = Matrix.eye(3)
    A = al * I3
    B = be * I3
    G = A + B






if __name__ == "__main__":

    main()

    init_printing(use_latex=True, wrap_line=False)
