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

from itertools import product

from datetime import datetime, time
import sys
from typing import cast

import sympy as sp
from sympy import Derivative, simplify, symbols, init_printing
from sympy import Matrix, Float, Expr

from aided._sympy.dynamic.functions import dyn_prefactor, gss_dyn
from aided._sympy.sympy_utils import eval_in_mma


from .verifications import (
    __verify_product_of_gtos,
    __verify_product_of_gtos_scheringer,
    __verify_detG_inv,
    __verify_gss_dyn_term_by_term,
    __verify_Wdet,
    __verify_gdyn,
)


def main():

    ##################
    # Define symbols #
    ##################

    # Define position vectors and coordinates.
    x, y, z = symbols("x y z", real=True)
    r = Matrix([x, y, z])

    # Define the center of GTO centered position a and b.
    ax, ay, az = symbols("ax ay az", real=True)
    a = Matrix([ax, ay, az])
    bx, by, bz = symbols("bx by bz", real=True)
    b = Matrix([bx, by, bz])
    al, be = symbols("al be", real=True, positive=True)

    # Define the position c which is the center of the GTO centered position a and b.
    ga = al + be
    c = (al * a + be * b) / ga

    # Build Scheringer's matrices
    ga = al + be
    I3 = Matrix.eye(3)
    A = al * I3
    B = be * I3
    G = A + B

    # Build U matrix
    u11, u12, u13 = symbols("u11 u12 u13", real=True)
    u22, u23, u33 = symbols("u22 u23 u33", real=True)

    # fmt: off
    U = Matrix([
        [u11, u12, u13],
        [u12, u22, u23],
        [u13, u23, u33]
    ])
    # fmt: on


    # fmt: off
    # W = G.inv() + U
    W = Matrix([
        [u11 + 1/ga, u12,        u13       ],
        [u12       , u22 + 1/ga, u23       ],
        [u13       , u23,        u33 + 1/ga]
    ])
    # fmt: on

    """
    # Loop over La = [ax, ay, az], Lb = [bx, by, bz] where any of the values ranges from 0 to 2.
    for la, ma, na, lb, mb, nb in product(range(2), repeat=6):
        print(f"[*] La = [{la}, {ma}, {na}] Lb = [{lb}, {mb}, {nb}] ... ", end="", flush=True)
        ok = __verify_gdyn(r, a, b, c, al, be, ga, W, La=[la, ma, na], Lb=[lb, mb, nb], verbose=False)
        if not ok:
            print("FAILED", flush=True,)
        else:
            print(f"OK", flush=True)
    """

    for i in range(6):
        La = [0, 0, 0]
        Lb = [0, 0, 0]
        if i < 3:
            La[i] = 1
        else:
            Lb[i - 3] = 1
        print(f"[*] La = {La} Lb = {Lb} ... ", end="", flush=True)
        ok = __verify_gdyn(r, a, b, c, al, be, ga, W, La=La, Lb=Lb, verbose=False)
        if not ok:
            print("FAILED", flush=True,)
        else:
            print(f"OK", flush=True)



    sys.exit(0)
    ############################################################
    # Verify our definition of the product of GTOs is correct. #
    ############################################################
    __verify_product_of_gtos(r, a, b, c, al, be, ga)

    #################################
    # Verify W.det() simplification #
    #################################
    __verify_Wdet(W, ga, u11, u12, u13, u22, u23, u33)

    #######################################################################################
    # Verify Our definition of the product of GTOs is correct is the same as Scheringer's #
    #######################################################################################
    __verify_product_of_gtos_scheringer(r, a, b, c, al, be, ga, A, B, G)

    ##########################################
    # Verify that 1 / det(G) == det(G.inv()) #
    ##########################################
    __verify_detG_inv(G)

    ###############################
    # Verify gss_dyn term by term #
    ###############################
    __verify_gss_dyn_term_by_term(r, a, b, c, al, be, ga, A, B, G, U, W)


if __name__ == "__main__":

    main()

    init_printing(use_latex=True, wrap_line=False)
