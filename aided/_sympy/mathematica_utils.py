"""
Utilities to invoke Mathematica via wolframscript in python.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import subprocess

def mathematica_eval(expr_str: str) -> str:
    """Evaluate a Mathematica expression using wolframscript

    Args:
        expr_str (str): The Mathematica expression to evaluate.

    Returns:
        str: The result of the evaluation.
    """
    wrapped_code = f"FullSimplify[{expr_str}]"
    result = subprocess.run(
        ["wolframscript", "-code", wrapped_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("Error executing Mathematica code:", result.stderr)
        return ""
    return result.stdout.strip()
