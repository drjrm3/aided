"""
Notebook utilities for dynamic analysis.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Tuple

from aided.core import EDWfn
from aided.core.edwfn import EDWfnStatic, EDWfnDynamic


WFN_FILE = (
    Path(__file__).parent / "../../test/data/msda/g09/formamide.b3lyp.6311gss.wfn"
).resolve()
LOG_FILE = (
    Path(__file__).parent / "../../test/data/msda/g09/formamide.b3lyp.6311gss.log"
).resolve()

print(f"Using WFN file: {WFN_FILE}")


def get_wfn_objects(
    wfn_file: Path = WFN_FILE, log_file: Path = LOG_FILE, T: float = 300
) -> Tuple[EDWfn, EDWfn]:
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
