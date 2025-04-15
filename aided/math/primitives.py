"""
Simple mathematical primitives.
"""

from typing import Union
from .. import np, npt


def gpow(
    x: Union[float, npt.NDArray], expon: Union[int, npt.NDArray]
) -> Union[float, npt.NDArray]:
    """Custom power operator for integer exponents, supporting both scalars and vectors.

    This computes np.power on most elements but handles special cases where x or expon is zero.
        - If expon is zero, it returns 1.0. This is handled by initialization and exclusion.
        - If x is zero, it returns 0.0. This is handled by a special case.
        - For everything else, it computes x ** expon.

    Args:
        x: Base value.
        expon: Exponent value.

    Returns:
        result: The resulting value of x raised to the power of expon.
    """

    # Convert inputs to NumPy arrays for vectorized operations
    x = np.asarray(x)  # Ensure x is a NumPy array
    expon = np.asarray(expon)  # Ensure expon is a NumPy array

    # Broadcast to a common shape so that x[mask] and expon[mask] work
    x, expon = np.broadcast_arrays(x, expon)

    # Initialize result with ones
    result = np.ones_like(x, dtype=float)

    # If x == 0, return 0.0
    result[(x == 0) & (expon != 0)] = 0.0

    # Everything else
    mask = (expon != 0) & (x != 0)
    result[mask] = np.power(x[mask], expon[mask])

    # Scalar output if both inputs are scalars
    if np.isscalar(x) and np.isscalar(expon):
        return result.item()  # Convert NumPy scalar to Python scalar
    return result  # Return NumPy array for vector inputs
