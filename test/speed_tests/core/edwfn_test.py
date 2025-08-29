#!/usr/bin/env python3
"""
Speed tests for edwfn

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from datetime import datetime

import numpy as np

from aided.core.ed.wfn import EDWfn


def test_rho(wfnfile: str):
    wfn = EDWfn(wfnfile)

    wfn.rho(0, 0, 0)


def test_gen_gs(wfnfile: str, num_iters: int):
    """Test the speed of the _gen_gs function.

    Args:
        wfn: EDWfn object.
        num_iters: Number of iterations for speed tests.
    """
    wfn = EDWfn(wfnfile)
    ymin, ymax = float(np.min(wfn.atpos[1, :])), float(np.max(wfn.atpos[1, :]))
    xmin, xmax = float(np.min(wfn.atpos[0, :])), float(np.max(wfn.atpos[0, :]))

    for ider in range(3):
        wfn = EDWfn(wfnfile)
        t1 = datetime.now()
        for _ in range(num_iters):
            x, y = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
            wfn._gen_gs(x, y, 0, ider=ider)
        t2 = datetime.now()
        tdiff = (t2 - t1).total_seconds()
        rate = (num_iters / tdiff) / 1000
        print(f"_gen_gs runs at a speed of {rate:5.1f}K calls/sec for ider={ider}.")


def main(args):
    """Main routine for speed tests.

    Args:
        args: Command line arguments.
            - args.wfnfile: Directory containing test data.
            - args.num_iters: Number of iterations for speed tests.
    """
    test_gen_gs(args.wfnfile, args.num_iters)
