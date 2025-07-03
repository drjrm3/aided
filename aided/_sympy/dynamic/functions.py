"""
Functions to be used in dynamic ED calculation.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List, cast
import sympy as sp
from sympy import Matrix, Expr, Rational, hermite


def gss(r: Matrix, R: Matrix, alpha: Expr) -> Expr:
    """Gaussian type orbital (GTO) function g(r) = exp(-alpha |r - R|**2).

    Args:
        r (Matrix): The position vector (x, y, z).
        R (Matrix): The center of the GTO (R_x, R_y, R_z).
        alpha (Float): The exponent of the GTO.

    Returns:
        Expr: The value of the GTO at position r.
    """

    arg = -alpha * (r - R).norm() ** 2
    _gss = cast(Expr, sp.exp(arg))

    return _gss


def Eg(a: Matrix, b: Matrix, al: Expr, be: Expr) -> Expr:
    """The constant factor of the product of two GTOS centered at a and b.

    Used in calculation of dynamic s-s orbital.

    In the fortran code from PhD work this is the Eg value.

    Args:
        a (Matrix): The center of the first GTO (ax, ay, az).
        b (Matrix): The center of the second GTO (bx, by, bz).
        al (Float): The exponent of the first GTO.
        be (Float): The exponent of the second GTO.

    Returns:
        Expr: The constant factor of the product of the two GTOS.
    """

    arg = -((al * be) / (al + be)) * (a - b).norm() ** 2
    _factor = cast(Expr, sp.exp(arg))

    return _factor


def hermite_vector(q: Matrix, L: List[int]) -> Expr:
    lx, ly, lz = L

    Hx = hermite(lx, q[0], evaluate=False)
    Hy = hermite(ly, q[1], evaluate=False)
    Hz = hermite(lz, q[2], evaluate=False)

    return Hx * Hy * Hz


def _scale(F, L):
    """∏_i F_ii^{L_i/2}"""
    return sp.prod(F[i, i] ** (L[i] / 2) for i in range(3))


def dyn_prefactor(
    r: Matrix,  # Position vector (x, y, z)
    a: Matrix,  # Center of the first GTO (ax, ay, az)
    b: Matrix,  # Center of the second GTO (bx, by, bz)
    c: Matrix,  # Center of the multiplied GTO (cx, cy, cz)
    al: Expr,  # Exponent of the first GTO
    be: Expr,  # Exponent of the second GTO
    ga: Expr,  # Exponent of the product of the two GTOs (al + be)
    F: Matrix,  # Inverse of exponent matrix for the product of the two GTOs (G.inv() + U).inv()
    La: List[int],  # Angular momentum order of the first GTO (la, ma, na)
    Lb: List[int],  # Angular momentum order of the second GTO (lb, mb, nb)
) -> Expr:
    """Calculates the dynamic prefactor term which must be used in the products to get derivatives
    of higher order GTO products for the dynamic ED.

    NOTE: Scheringer's F is our Winv.

    Giving order == 0 must return 1.0.
    """

    Frc = F @ (r - c)
    Pa = Frc + be * (b - a)
    Pb = Frc + be * (a - b)

    sqrt = [sp.sqrt(F[i, i] * ga) for i in range(3)]

    q_a = Matrix([Pa[i] / sqrt[i] for i in range(3)])
    q_b = Matrix([-Pb[i] / sqrt[i] for i in range(3)])  # minus sign for b

    if sum(La) > 0:
        Ca = hermite_vector(q_a, La) * (al / sp.sqrt(ga)) ** sum(La) * _scale(F, La)
    else:
        Ca = 1

    if sum(Lb) > 0:
        Cb = hermite_vector(q_b, Lb) * (be / sp.sqrt(ga)) ** sum(Lb) * _scale(F, Lb)
    else:
        Cb = 1

    prefactor = sp.powsimp(Ca * Cb)

    return prefactor


# fmt: off
def gss_stat_scheringer(
    r: Matrix, # Position vector (x, y, z)
    a: Matrix, # Center of the first GTO (ax, ay, az)
    b: Matrix, # Center of the second GTO (bx, by, bz)
    c: Matrix, # Center of the multiplied GTO (cx, cy, cz)
    A: Matrix, # Exponent matrix for the first GTO
    B: Matrix, # Exponent matrix for the second GTO
    G: Matrix  # Exponent matrix for the product of the two GTOs
) -> Expr:
    """This is the Scheringer definition for the Gaussian type orbital (GTO) function."""

    assert G == A + B, "Matrix G must equal A + B."
    assert c == G.inv() @ (A @ a + B @ b), "Matrix c must equal G inverse times (A*a + B*b)."

    arg1 = (r - c).T @ G @ (r - c)
    arg2 = a.T @ A @ a + b.T @ B @ b - c.T @ G @ c

    _gss = cast(Expr, sp.exp(-arg1) * sp.exp(-arg2))

    return _gss

def gss_dyn(
    r: Matrix, # Position vector (x, y, z)
    a: Matrix, # Center of the first GTO (ax, ay, az)
    b: Matrix, # Center of the second GTO (bx, by, bz)
    c: Matrix, # Center of the multiplied GTO (cx, cy, cz): 
    al: Expr, # Exponent of the first GTO
    be: Expr, # Exponent of the second GTO
    ga: Expr, # Exponent of the product of the two GTOs: al + be
    W: Matrix, # Exponent matrix for the product of the two GTOs: G.inv() + U
) -> Expr:
    """My definition for the dynamic ED."""

    # G.inv().det() == 1 / ga**3
    term1 = (ga ** 3 * W.det()) ** Rational(-1, 2)

    arg2 = (r - c).T @ W.inv() @ (r - c)
    assert arg2.shape == (1,1), f"args is {arg2.shape}, expected [1,1]."
    arg2 = arg2[0, 0]

    term3 = Eg(a, b, al, be)

    _gss_dyn = term1 * sp.exp(-arg2) * term3

    return _gss_dyn



def gss_dyn_scheringer(
    r: Matrix, # Position vector (x, y, z)
    a: Matrix, # Center of the first GTO (ax, ay, az)
    b: Matrix, # Center of the second GTO (bx, by, bz)
    c: Matrix, # Center of the multiplied GTO (cx, cy, cz)j
    A: Matrix, # Exponent matrix for the first GTO
    B: Matrix, # Exponent matrix for the second GTO
    G: Matrix, # Exponent matrix for the product of the two GTOs
    U: Matrix  # MSDA for the product of the two GTOs
) -> Expr:
    """Scheringer's definition of the dynamic ED."""

    term1 = (G.inv().det() * (G.inv() + U).inv().det()) ** (1/2)
    arg1 = (r - c).T @ (G.inv() + U).inv() @ (r - c)
    arg2 = a.T @ A @ a + b.T @ B @ b - c.T @ G @ c

    _gss_dyn = term1 * sp.exp(-arg1) * sp.exp(-arg2)

    return _gss_dyn
