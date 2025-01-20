"""
Simple mathematical primitives.
"""

def gpow(x: float, expon: int) -> float:
    """Custom power operator for integer exponents."""

    if expon == 0:
        return 1.0

    if x == 0:
        return 0.0

    if expon < 0:
        return 1.0 / gpowr(x, -expon)

    return gpowr(x, expon)

def gpowr(x: float, expon: int) -> float:
    """Custom power operator for integer exponents."""

    return x if expon == 1 else x * gpowr(x, expon - 1)
