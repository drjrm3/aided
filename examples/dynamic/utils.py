"""
Notebook utilities for dynamic analysis.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

from aided.core.edwfn import EDWfn, EDWfnStatic, EDWfnDynamic

WFN_FILE = (
    Path(__file__).parent / "../../test/data/msda/g09/formamide.b3lyp.6311gss.wfn"
).resolve()
LOG_FILE = (
    Path(__file__).parent / "../../test/data/msda/g09/formamide.b3lyp.6311gss.log"
).resolve()

def main() -> Tuple[EDWfn, EDWfn]:
    """Main routine to run inside of Jupyterhub.

    I'm only using it for visualization XD.
    """

    ### General parameters to use.
    T = 0

    static, dynamic = get_wfn_objects(T=T)

    # Create an empty `Axes` object for plotting.
    fig, ax = plt.subplots(figsize=(8, 8))

    xs, ys = generate_line(static, "C2", "O6", npts=20, buffer=0.5)

    static_rho = [static.rho(x, y, z) for x, y, z in zip(xs, ys, [0] * len(xs))]
    dynamic_rho = [dynamic.rho(x, y, z) for x, y, z in zip(xs, ys, [0] * len(xs))]

    ax.plot(static_rho, "-r.")
    ax.plot(dynamic_rho, "-b.")

    return static, dynamic

def generate_line(wfn: EDWfn, at1: str, at2: str, npts: int, buffer: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a line between two atoms in the wfn object.

    Args:
        wfn (EDWfn): WFN object.
        at1 (str): Name of the first atom.
        at2 (str): Name of the second atom.
        buffer (float): Extra distance to extend the line beyond the atoms.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinates of the line.
    """
    at1_pos = wfn.atpos[wfn.atnames == at1][0]
    at2_pos = wfn.atpos[wfn.atnames == at2][0]
    direction = at2_pos - at1_pos
    length = np.linalg.norm(direction)
    direction /= length  # Normalize the direction vector
    # Generate points along the line
    xs = np.linspace(at1_pos[0] - buffer * direction[0], at2_pos[0] + buffer * direction[0], npts)
    ys = np.linspace(at1_pos[1] - buffer * direction[1], at2_pos[1] + buffer * direction[1], npts)

    return xs, ys


def get_wfn_objects(
    wfn_file: Path = WFN_FILE, log_file: Path = LOG_FILE, T: float = 300
) -> Tuple[EDWfn, EDWfn]:
    """Get wfn objects.

    Args:
        wfn_file (Path): Path to the WFN file.
        log_file (Path): Path to the log file.
        T (float): Temperature in Kelvin.

    Returns:
        Tuple[EDWfn, EDWfn]: Static and dynamic wfn objects.
    """
    static = EDWfnStatic(wfn_file.as_posix())
    dynamic = EDWfnDynamic(wfn_file.as_posix(), log_file=log_file.as_posix(), T=T)

    return static, dynamic


def plot_bonds(wfn: EDWfn, ax: Axes) -> Axes:
    """Plot bonds from wfn object."""
    ax.plot([wfn.atpos[0, 0], wfn.atpos[1, 0]], [wfn.atpos[0, 1], wfn.atpos[1, 1]], "k")  # H1, C2
    ax.plot([wfn.atpos[1, 0], wfn.atpos[5, 0]], [wfn.atpos[1, 1], wfn.atpos[5, 1]], "k")  # C2, O6
    ax.plot([wfn.atpos[2, 0], wfn.atpos[3, 0]], [wfn.atpos[2, 1], wfn.atpos[3, 1]], "k")  # N3, H4
    ax.plot([wfn.atpos[1, 0], wfn.atpos[2, 0]], [wfn.atpos[1, 1], wfn.atpos[2, 1]], "k")  # C2, N3
    ax.plot([wfn.atpos[2, 0], wfn.atpos[4, 0]], [wfn.atpos[2, 1], wfn.atpos[4, 1]], "k")  # N3, H5
    for atpos, atname in zip(wfn.atpos, wfn.atnames):
        ax.text(atpos[0], atpos[1], atname)

    return ax
