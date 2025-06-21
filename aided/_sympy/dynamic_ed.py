"""
Validation and testing of dynamic electron density.

This work is based on:
    1. "Validation of convolution approximation to the thermal-average electron density",
        Michael, J. R, Koritsaznsky, T, J. Math Chem (2015) 53:250-259
    2. "Direct Calculation of Dynamic Electron Density",
        Scheringer, C., Reitz, H., Acta Crys (1976), A32, 271.

    See also:
    - J. A. Pople & J. W. McIver, Int. J. Quantum Chem. 3, 269 (1969)
    - E. R. Davidson & J. B. McMurchie, J. Comput. Phys. 26, 218 (1978)
    - T. Helgaker, P. Jørgensen & J. Olsen, Molecular Electronic-Structure Theory (2000), ch. 8

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import sys
import sympy as sp
from sympy import Matrix
from sympy.printing.mathematica import mathematica_code
from mathematica_utils import mathematica_eval


def validate_scheringer_c(
    G: Matrix, a: sp.Matrix, b: sp.Matrix, c: sp.Matrix, alpha: float, beta: float
):
    """Validate that Scheringer's expression in the general sense is equal to mine.

    G: Scheringer expression for the Gaussian product matrix which does not assume isotropic tensors
        Size 3x3 - Here it is assumed to be isotropic with G = A + B, A = alpha * I, B = beta * I
    a: Position of GTO 'a'
    b: Position of GTO 'b'
    alpha: Exponent of GTO 'a'
    beta: Exponent of GTO 'b'
    """

    c_scheringer = G.inv() * (alpha * a + beta * b)
    if not all([x == 0 for x in sp.simplify(c_scheringer - c)]):

        print("Scheringer center ... ", c_scheringer)
        print("My center ........... ", c)
        print("Simplify ............ ", sp.simplify(c_scheringer - c))

        raise ValueError("Scheringer center is not the same as GTO center.")


def validate_scheringer_ss(
    r: Matrix,
    a: Matrix,
    b: Matrix,
    c: Matrix,
    A: Matrix,
    B: Matrix,
    G: Matrix,
    alpha: float,
    beta: float,
):
    """Validate that Scheringer's expression for a ss orbital is equal to mine.

    r: Position vector to evaluate function at.
    a: Position of GTO 'a'
    b: Position of GTO 'b'
    c: Center of GTO 'a' and 'b'.
    A: GTO exponent matrix for 'a': alpha * I
    B: GTO exponent matrix for 'b': beta * I
    G: GTO exponent matrix for 'c': alpha * I + beta * I
    alpha: Exponent of GTO 'a'
    beta: Exponent of GTO 'b'
    """

    xcGxc = ((r - c).T * G * (r - c))[0]
    aAa = (a.T * A * a)[0]
    bBb = (b.T * B * b)[0]
    cGc = (c.T * G * c)[0]

    # NOTE: Scheringer uses -0.5 here. I do not.
    # gss_c = sp.exp(-0.5 * xcGxc) * sp.exp(-0.5 * (aAa + bBb - cGc))
    gss_c = sp.exp(-xcGxc) * sp.exp(-(aAa + bBb - cGc))

    ga_ss = gto(r, a, alpha, Matrix([0, 0, 0]))
    gb_ss = gto(r, b, beta, Matrix([0, 0, 0]))
    gc_ss = sp.simplify(ga_ss * gb_ss)
    diff = sp.simplify(sp.log(gss_c) / sp.log(gc_ss))

    if diff != 1:
        print("Scheringer ss ... ", gss_c)
        print("My ss ........... ", gc_ss)
        print("Diff ............ ", diff)

        raise ValueError("Scheringer ss is not the same as my ss.")


def gto(r: Matrix, R: sp.Matrix, alpha: float, lmn: sp.Matrix) -> float:
    """Definition for gaussian type function.

    r: Position vector to evaluate function at.
    R: Center of GTO.
    alpha: Exponent of GTO.
    lmn: Angular momentum quantum numbers.
    """

    x, y, z = map(sp.sympify, [r[i] for i in range(3)])
    X, Y, Z = map(sp.sympify, [R[i] for i in range(3)])
    l, m, n = map(sp.sympify, [lmn[i] for i in range(3)])
    # Gaussian function.
    gto = (
        (x - X) ** l
        * (y - Y) ** m
        * (z - Z) ** n
        * sp.exp(-alpha * ((x - X) ** 2 + (y - Y) ** 2 + (z - Z) ** 2))
    )

    return gto


def validate_W(U: Matrix, W: Matrix):
    """Verify that W is the inverse of U."""

    if sp.simplify(U * W) != sp.eye(3):
        print("U: ", U)
        print("W: ", W)
        raise ValueError("W is not the inverse of U.")


def dynamic_ed():
    """Workspace to go through equations of dynamic ED."""

    #### Symbols
    # Generic points, x, y, z to evaluate functions at.
    x, y, z = sp.symbols("x y z", real=True)
    # Centers of GTOs 'a' and 'b'.
    a_x, a_y, a_z = sp.symbols("a_x a_y a_z", real=True)
    b_x, b_y, b_z = sp.symbols("b_x b_y b_z", real=True)
    # GTO angular momentum quantum numbers.
    l, m, n = sp.symbols("l m n", integer=True, nonnegative=True)
    l_a, m_a, n_a = sp.symbols("l_a m_a n_a", integer=True, nonnegative=True)
    l_b, m_b, n_b = sp.symbols("l_b m_b n_b", integer=True, nonnegative=True)

    # GTO exponents and contraction coefficients.
    alpha, beta = sp.symbols("alpha beta", positive=True, real=True)

    ### Positional vectors.
    r = Matrix([x, y, z])
    a = Matrix([a_x, a_y, a_z])
    b = Matrix([b_x, b_y, b_z])

    # Center of GTO 'a' and 'b'.
    c = (alpha * a + beta * b) / (alpha + beta)

    # Gausian product matrix, as defined by Scheringer.
    A = alpha * sp.eye(3)
    B = beta * sp.eye(3)
    G = A + B

    # Vibrational symmetric tensor U.
    u11, u12, u13, u22, u23, u33 = sp.symbols("u11 u12 u13 u22 u23 u33", real=True)

    # fmt: off
    U = sp.Matrix([
        [u11, u12, u13],
        [u12, u22, u23],
        [u13, u23, u33]
    ])

    # Validate that Scheringer's expression in the general sense is equal to mine.
    validate_scheringer_c(G, a, b, c, alpha, beta)

    # Validate that Scheringer's expression for a ss orbital is equal to mine.
    validate_scheringer_ss(r, a, b, c, A, B, G, alpha, beta)

    #############################################
    ### Validate dynamic gss with Scheringer. ###
    #############################################

    # Terms for Scheringer's gss_dyn
    s_term1 = (G.inv().det() * (G.inv() + U).det() ** -1) ** (1 / 2)
    print("STerm1: ", s_term1, "\n")

    s_term2 = -0.5 * (((r - c).T * (G.inv() + U).det() * (r - c)))[0]
    print("STerm2: ", s_term2, "\n")

    s_term3 = -0.5 * ((a.T * A * a + b.T * B * b - c.T * G * c))[0]
    print("STerm3: ", s_term3, "\n")

    gss_dyn_scheringer = s_term1 * sp.exp(s_term2) * sp.exp(s_term3)
    print("Scheringer gss_dyn: ", gss_dyn_scheringer, "\n")


    ### TODO: Define W, etc. and generate the simplisitic form of gss_dyn ###
