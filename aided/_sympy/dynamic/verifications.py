"""Verifications for Dynamic ED math.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List, cast

import sympy as sp
from sympy import Matrix, Expr, Rational, simplify

from .functions import dyn_prefactor, gss, Eg, gss_dyn, gss_stat_scheringer
from ..sympy_utils import eval_in_mma

assumptions="al > 0 && be > 0 && ga > 0"
assumptions += " && u11 > 0 && u12 > 0 && u13 > 0 && u22 > 0 && u23 > 0 && u33 > 0"
assumptions += " && ax in Reals && ay in Reals && az in Reals"
assumptions += " && bx in Reals && by in Reals && bz in Reals"


# fmt: off
def __verify_product_of_gtos(
    r: Matrix, # Position vector (x, y, z)
    a: Matrix, # Center of the first GTO (ax, ay, az)
    b: Matrix, # Center of the second GTO (bx, by, bz)
    c: Matrix, # Center of the product GTO (cx, cy, cz)
    al: Expr,  # Exponent of the first GTO
    be: Expr,  # Exponent of the second GTO
    ga: Expr   # Exponent of the product GTO
) -> bool:
    """Verify the product of two GTOS centered at a and b."""

    print("[*] Verifying product of GTOS ... ", end="", flush=True)

    assert ga == al + be, f"ga must be equal to al + be, it is {ga} != {al + be}"
    assert c == (al * a + be * b) / ga, "c must be the center of the product GTO"

    lhs = simplify(gss(r, a, al) * gss(r, b, be))
    rhs = simplify(Eg(a, b, al, be) * gss(r, c, ga))

    assert eval_in_mma(lhs / rhs, simplify="FullSimplify") == "1"

    # print a checkmark
    print("[✓]")
    return True

def _make_multi_derivative(expr, a, b, La, Lb):
    """
    Derivative of *expr* of total order ΣLa+ΣLb with respect to
        (ax, ay, az)^La   and   (bx, by, bz)^Lb
    where  a = (ax, ay, az)  and  b = (bx, by, bz).

    If all angular momenta are zero  →  returns *expr* itself.
    """
    ax, ay, az = a
    bx, by, bz = b

    deriv_vars = (
          [ax] * La[0] + [ay] * La[1] + [az] * La[2]
        + [bx] * Lb[0] + [by] * Lb[1] + [bz] * Lb[2]
    )
    return sp.diff(expr, *deriv_vars) if deriv_vars else expr

def __verify_gdyn(
    r: Matrix,  # Position vector (x, y, z)
    a: Matrix,  # Center of the first GTO (ax, ay, az)
    b: Matrix,  # Center of the second GTO (bx, by, bz)
    c: Matrix,  # Center of the product GTO (cx, cy, cz)
    al: Expr,   # Exponent of the first GTO
    be: Expr,   # Exponent of the second GTO
    ga: Expr,   # Exponent of the product GTO
    W: Matrix,  # G.inv() + U
    La: List[int], # Angular momentum of the first GTO
    Lb: List[int], # Angular momentum of the second GTO
) -> bool:
    """Verify the dynamic GTO for arbitrary angular momenta by comparing to the derivative."""

    F = cast(Matrix, W.inv())  # Scheringer's F matrix

    # s-s orbital
    print("Calculating gss_dyn ... ", end="", flush=True)
    g0 = gss_dyn(r, a, b, c, al, be, ga, W)
    print("done.")

    # Calculate derivatives for La, Lb
    print("Calculating derivatives ... ", end="", flush=True)
    gdyn_from_derivs = _make_multi_derivative(g0, a, b, La, Lb)
    print("done.")

    print("Canceling common factors ... ", end="", flush=True)
    lhs = sp.cancel(gdyn_from_derivs / g0)
    print("done.")

    # Calculate from prefactor. 
    print("Calculating prefactor ... ", end="", flush=True)
    C_pref = dyn_prefactor(r, a, b, c, al, be, ga, F, La, Lb)
    rhs = C_pref
    print("done.")
    print("Expanding prefactor ... ", end="", flush=True)
    #gdyn_from_prefactors = sp.expand_mul(C_pref * g0)
    #rhs = gdyn_from_prefactors
    print("done.")

    print("Comparing ... ", end="", flush=True)
    out = eval_in_mma(lhs / rhs, simplify="Simplify", assumptions=assumptions)
    print("done.")
    print(out)

    return True


    ### Compare







def __verify_product_of_gtos_scheringer(
    r: Matrix,  # Position vector (x, y, z)
    a: Matrix,  # Center of the first GTO (ax, ay, az)
    b: Matrix,  # Center of the second GTO (bx, by, bz)
    c: Matrix,  # Center of the product GTO (cx, cy, cz)
    al: Expr,   # Exponent of the first GTO
    be: Expr,   # Exponent of the second GTO
    ga: Expr,   # Exponent of the product GTO
    A: Matrix,  # Matrix for the first GTO
    B: Matrix,  # Matrix for the second GTO
    G: Matrix,  # Matrix for the product GTO
) -> bool:
    """Verify that Scheringer's definition of gss_stat is the same as ours."""


    print("[*] Verifying Scheringer gss_stat ... ", end="", flush=True)
    lhs = gss_stat_scheringer(r, a, b, c, A, B, G)
    rhs = Eg(a, b, al, be) * gss(r, c, ga)

    assert eval_in_mma(lhs / rhs, simplify="FullSimplify").replace("{", "").replace("}", "") == "1"

    print("[✓]")
    return True

def __verify_detG_inv(G: Matrix):
    """Verify that the determinant of G is the inverse of G."""
    print("[*] Verifying det(G**-1) == det(G)**-1 ... ", end="", flush=True)

    detGinv1 = 1. / cast(Expr, G.det())
    detGinv2 = G.inv().det()


    lhs = detGinv1
    rhs = detGinv2

    assert eval_in_mma(lhs - rhs, simplify="FullSimplify") == "0"

    print("[✓]")
    return True

# fmt: off
def __verify_gss_dyn_term_by_term(
    r: Matrix, # Position vector (x, y, z)
    a: Matrix, # Center of the first GTO (ax, ay, az)
    b: Matrix, # Center of the second GTO (bx, by, bz)
    c: Matrix, # Center of the multiplied GTO (cx, cy, cz)j
    al: Expr,  # Exponent of the first GTO
    be: Expr,  # Exponent of the second GTO
    ga: Expr,  # Exponent of the product GTO
    A: Matrix, # Exponent matrix for the first GTO
    B: Matrix, # Exponent matrix for the second GTO
    G: Matrix, # Exponent matrix for the product of the two GTOs
    U: Matrix, # MSDA for the product of the two GTOs
    W: Matrix  # G.inv() + U
) -> bool:
    # fmt: on

    #####################################
    ### Verify first timer in gss_dyn ###
    #####################################
    print("[**] Verifying 1st term in gss_dyn ... ", end="", flush=True)
    my_term1 = (ga ** 3 * W.det()) ** Rational(-1, 2)
    G_inv_det = 1. / cast(Expr, G.det())
    G_inv_U__inv_det = (1 / cast(Expr, (G.inv() + U).det()))
    sc_term1 = (G_inv_det * G_inv_U__inv_det) ** Rational(1, 2)

    lhs1 = my_term1
    rhs1 = sc_term1

    assert eval_in_mma(
        lhs1 - rhs1,
        simplify=["PowerExpand", "FullSimplify"],
        assumptions=assumptions
    ).replace(".", "") == "0"
    print("[✓]")

    #####################################
    ### Verify second term in gss_dyn ###
    #####################################
    print("[**] Verifying 2nd term in gss_dyn ... ", end="", flush=True)
    my_arg1 = (r - c).T @ W.inv() @ (r - c)
    assert my_arg1.shape == (1, 1)
    my_arg1 = my_arg1[0, 0]
    sc_arg1 = (r - c).T @ (G.inv() + U).inv() @ (r - c)
    assert sc_arg1.shape == (1, 1)
    sc_arg1 = sc_arg1[0, 0]

    assert eval_in_mma(
        sp.exp(my_arg1) - sp.exp(sc_arg1),
        simplify=["FullSimplify"],
        assumptions=assumptions
    ) == "0"
    print("[✓]")

    ##################################
    ### Verify 3rd term in gss_dyn ###
    ##################################
    print("[**] Verifying 3rd term in gss_dyn ... ", end="", flush=True)
    my_term3 = Eg(a, b, al, be)
    sc_arg3 = simplify(a.T @ A @ a + b.T @ B @ b - c.T @ G @ c)
    assert sc_arg3.shape == (1, 1)
    sc_arg3 = sc_arg3[0, 0]
    sc_term3 = sp.exp(-sc_arg3)

    assert eval_in_mma(
        my_term3 - sc_term3,
        simplify=["Simplify"],
        assumptions=assumptions
    ).replace(".", "") == "0"
    print("[✓]")

    ###############################
    ### Verify the entire thing ###
    ###############################

    print("[**] Verifying gss_dyn ... ", end="", flush=True)
    lhs = my_term1 * sp.exp(my_arg1) * my_term3
    rhs = sc_term1 * sp.exp(sc_arg1) * sc_term3
    assert eval_in_mma(
        lhs - rhs,
        simplify=["PowerExpand", "FullSimplify"],
        assumptions=assumptions
    ).replace(".", "") == "0"
    print("[✓]")

    return True


# fmt: on
def __verify_Wdet(W: Matrix, ga, u11, u12, u13, u22, u23, u33) -> bool:
    """Verify the determinant of W."""
    Wdet = ga**-3 * (
        1
        + ga
        * (
            u11
            + u22
            + u33
            + ga
            * (
                u11 * (u22 - ga * u23**2 + u33 + ga * u22 * u33)
                - 1
                * (
                    u13**2 * (1 + ga * u22)
                    - 2 * ga * u12 * u13 * u23
                    + u23**2
                    - u22 * u33
                    + u12**2 * (1 + ga * u33)
                )
            )
        )
    )
    print("[*] Verifying W.det() ... ", end="", flush=True)
    result = eval_in_mma(W.det() - Wdet, simplify="FullSimplify", assumptions="al > 0 && be > 0")
    assert result == "0", f"W.det() verification failed: {result}"
    print("[✓]")

    return True
