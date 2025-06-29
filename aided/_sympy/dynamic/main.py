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

    Winv = cast(Matrix, W.inv())

    ##########################
    ## Exploratory analysis ##
    ##########################
    assumptions = "al > 0 && be > 0 && ga > 0"
    assumptions += " && u11 > 0 && u12 > 0 && u13 > 0 && u22 > 0 && u23 > 0 && u33 > 0"
    assumptions += " && ax in Reals && ay in Reals && az in Reals"
    assumptions += " && bx in Reals && by in Reals && bz in Reals"

    print(f"Calculating _Cax1 ... ", end="", flush=True)
    _Cax1 = dyn_prefactor(r, a, b, c, al, be, ga, F=Winv, La=[1, 0, 0], Lb=[0, 0, 0])
    print("done.", flush=True)

    print(f"Calculating gss_dyn ... ", end="", flush=True)
    g0 = gss_dyn(r, a, b, c, al, be, ga, W)
    print("done.", flush=True)

    print(f"Calculating g_ax1 ... ", end="", flush=True)
    g_ax1 = Derivative(g0, ax, 1).doit()
    print("done.", flush=True)

    print(f"Calculating Cax1 ... ", end="", flush=True)
    Cax1 = g_ax1 / g0
    print("done.", flush=True)

    _Cax1_str = eval_in_mma(_Cax1, simplify="Simplify", assumptions=assumptions)
    Cax1_str = eval_in_mma(Cax1, simplify="Simplify", assumptions=assumptions)
    print("Cax1_str = ", Cax1_str, flush=True)
    print("_Cax1_str = ", _Cax1_str, flush=True)

    print(
        "ratio = ",
        eval_in_mma(Cax1 / _Cax1, simplify="Simplify", assumptions=assumptions),
        flush=True,
    )

    sys.exit(0)

    g_ay1 = Derivative(g0, ay, 1).doit()
    print(
        "\ng_ay1 = ", eval_in_mma(g_ay1, simplify="Simplify", assumptions=assumptions), flush=True
    )
    Cay1 = eval_in_mma(g_ay1 / g0, simplify="Simplify")
    print("\nCay1 = ", Cay1, flush=True)

    g_az1 = Derivative(g0, az, 1).doit()
    print(
        "\ng_az1 = ", eval_in_mma(g_az1, simplify="Simplify", assumptions=assumptions), flush=True
    )
    Caz1 = eval_in_mma(g_az1 / g0, simplify="Simplify")
    print("\nCaz1 = ", Caz1, flush=True)

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
