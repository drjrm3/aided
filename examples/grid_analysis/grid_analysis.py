#!/usr/bin/env python3
"""
Analyze properties on a grid.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from datetime import datetime
from numpy.linalg import norm
import numpy as np
from aided import EDWfn

def gen_rho(xs, ys, wfn):
    """Generate the density rho(x,y,0) from the wavefunction."""

    with open("rho.txt", "w") as fout:
        t1 = datetime.now()
        for x in xs:
            for y in ys:
                rho = wfn.rho(x, y, 0)
                print(f"{x:10.5e} {y:10.5e} {rho:10.5e}", file=fout)
        t2 = datetime.now()
        tdiff = (t2 - t1).total_seconds()
        print(f"[*] Time taken for rho .... {tdiff:10.5f} seconds")

def gen_grad(xs, ys, wfn):
    """Generate the density grad(x,y,0) from the wavefunction."""

    with open("grad.txt", "w") as fout:
        t1 = datetime.now()
        for x in xs:
            for y in ys:
                grad = wfn.grad(x, y, 0)
                grad_str = " ".join([f"{g:10.5e}" for g in grad])
                print(f"{x:10.5e} {y:10.5e} {grad_str}", file=fout)
        t2 = datetime.now()
        tdiff = (t2 - t1).total_seconds()
        print(f"[*] Time taken for grad ... {tdiff:10.5f} seconds")

def gen_hess(xs, ys, wfn):
    """Generate the density hess(x,y,0) from the wavefunction."""

    with open("hess.txt", "w") as fout:
        t1 = datetime.now()
        for x in xs:
            for y in ys:
                hess = wfn.hess(x, y, 0)
                hess_str = " ".join([f"{h:10.5e}" for h in hess])
                print(f"{x:10.5e} {y:10.5e} {hess_str}", file=fout)
        t2 = datetime.now()
        tdiff = (t2 - t1).total_seconds()
        print(f"[*] Time taken for hess ... {tdiff:10.5f} seconds")

def main():
    """Main routine."""

    # Read in the Gaussian output file
    filename = "../../test/data/wfns/formamide/formamide.6311gss.b3lyp.wfn"

    wfn = EDWfn(filename)

    N = 250
    xs = np.linspace(-5.0, 5.0, N)
    ys = np.linspace(-5.0, 5.0, N)

    gen_rho(xs, ys, wfn)
    gen_grad(xs, ys, wfn)
    gen_hess(xs, ys, wfn)

if __name__ == "__main__":
    main()
