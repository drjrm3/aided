"""

Helper functions for the dynamic electron density until it is moved into wfn_dynamic.py

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np
import numpy.typing as npt

from numpy.typing import NDArray


def gss(r: np.ndarray, R: np.ndarray, alpha: float) -> float:
    """Gaussian type orbital (GTO) function g(r) = exp(-alpha |r - R|**2).

    Args:
        r (np.ndarray): The position vector (x, y, z) of shape (3,).
        R (np.ndarray): The center of the GTO (R_x, R_y, R_z) of shape (3,).
        alpha (float): The exponent of the GTO.

    Returns:
        float: The value of the GTO at position r.
    """
    diff = r - R
    norm_squared = np.sum(diff**2)
    arg = -alpha * norm_squared
    return np.exp(arg)


def Eg(A: np.ndarray, B: np.ndarray, al: float, be: float) -> float:
    """The constant factor of the product of two GTOS centered at a and b.

    Used in calculation of dynamic s-s orbital.

    In the fortran code from PhD work this is the Eg value.

    Args:
        a (np.ndarray): The center of the first GTO (ax, ay, az) of shape (3,).
        b (np.ndarray): The center of the second GTO (bx, by, bz) of shape (3,).
        al (float): The exponent of the first GTO.
        be (float): The exponent of the second GTO.

    Returns:
        float: The constant factor of the product of the two GTOS.
    """
    diff = A - B
    norm_squared = np.sum(diff**2)
    arg = -((al * be) / (al + be)) * norm_squared
    return np.exp(arg)


def gss_dyn(
    r: np.ndarray,
    A: np.ndarray, 
    B: np.ndarray,
    C: np.ndarray,
    al: float,
    be: float,
    ga: float,
    W: np.ndarray
) -> float:
    """Dynamic s-s orbital of two primitives.

    Args:
        r (np.ndarray): Position vector (x, y, z) of shape (3,).
        A (np.ndarray): Center of the first GTO (ax, ay, az) of shape (3,).
        B (np.ndarray): Center of the second GTO (bx, by, bz) of shape (3,).
        C (np.ndarray): Center of the multiplied GTO (cx, cy, cz) of shape (3,).
        al (float): Exponent of the first GTO.
        be (float): Exponent of the second GTO.
        ga (float): Exponent of the product of the two GTOs (al + be).
        W (np.ndarray): Exponent matrix for the product of the two GTOs (G^-1 + U) of shape (3, 3).

    Returns:
        float: The dynamic ED value.
    """
    # Term 1: (ga^3 * det(W))^(-1/2)
    term1 = (ga**3 * np.linalg.det(W))**(-0.5)
    
    # Term 2: exp(-(r-c)^T * W^-1 * (r-c))
    diff = r - C
    W_inv = np.linalg.inv(W)
    arg2 = np.dot(diff, np.dot(W_inv, diff))
    term2 = np.exp(-arg2)
    
    # Term 3: Eg factor
    term3 = Eg(A, B, al, be)
    
    return term1 * term2 * term3
