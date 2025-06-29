"""
Utilities to use in sympy
"""

import subprocess
from typing import Dict, Mapping, cast, Any, Iterable

from sympy import Expr
from sympy.printing.mathematica import mathematica_code
from sympy.parsing.sympy_parser import parse_expr


def _wl_eval(code: str, to_string: bool = True) -> str:
    """Send code to wolfram script for evaluation and return the string.

    Args:
        code (str): The Wolfram Language code to evaluate.
        to_string (bool): If True, return the output as a string. Defaults to True.

    Returns:
        str: The output from the Wolfram Language evaluation.
    """

    cmd = ["wolframscript", "-noprompt", "-code", code]
    out = subprocess.check_output(cmd, text=True)
    return out.strip() if to_string else out

def eval_in_mma(
    expr: Expr,
    *,
    subs: Mapping[Expr, Any] | None = None,
    simplify: str | Iterable[str] | None = None,
    numeric: bool = False,
    precision: int | None = None,
    assumptions: str | None = None
) -> str:
    """
    Push a SymPy expression to Wolfram Engine (via wolframscript) and return
    the raw stdout produced by the kernel.

    Args:
        expr (Expr): The SymPy expression to evaluate.
        subs (Mapping[Expr, Any] | None): Substitutions to apply before evaluation.
        simplify (str | iterable[str]): WL function name(s) that wrap the code, e.g. "FullSimplify".
            If a collection is given they are applied inside-out in the order
            listed: `s = ["Expand", "TrigReduce"]` → `TrigReduce[Expand[...]]`.
        numeric (bool): Whether to wrap with N[...].
        precision (int): Digits for N[..., precision].  Ignored if numeric is False.

    Returns
    -------
    str
        The kernel’s textual output (strip()ped).

    Examples
    --------
    >>> eval_in_mma(expr, simplify="FullSimplify")
    >>> eval_in_mma(expr, subs={x: 1.0}, numeric=True, precision=50)
    """

    if subs:
        expr = expr.subs(subs)
    wl_code: str = str(mathematica_code(expr))

    # add simplifier(s)
    if simplify:
        if isinstance(simplify, str):
            wl_code = f"{simplify}[{wl_code}]"
        else:                        # iterable
            for fun in simplify:
                wl_code = f"{fun}[{wl_code}]"

    # numeric evaluation
    if numeric:
        prec = f", {precision}" if precision else ""
        wl_code = f"N[{wl_code}{prec}]"

    # add assumptions
    if assumptions:
        wl_code = wl_code[:-1] + f", Assumptions -> ({assumptions})]"

    return _wl_eval(wl_code)
