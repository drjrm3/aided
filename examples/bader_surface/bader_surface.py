#!/usr/bin/env python3
"""
Generate bader surfaces.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from datetime import datetime
from numpy.linalg import norm
from aided import EDWfn


def main():
    """Main routine."""

    # Read in the Gaussian output file
    filename = "../../test/data/wfns/formamide/formamide.6311gss.b3lyp.wfn"

    wfn = EDWfn(filename)

    for atom_name in wfn.atnames:
        with open(f"{atom_name}.txt", "w") as fout:
            t1 = datetime.now()
            _, _, _xyzs = wfn.bader_surface_of_atom(
                atom_name, ntheta=3, nphi=25, step_size=0.2, tol=1e-12
            )
            t2 = datetime.now()
            tdiff = (t2 - t1).total_seconds()
            print(f"[*] Time taken for {atom_name}: {tdiff:10.5f} seconds")
            for xyz in _xyzs:
                print(f"{xyz[0]:12.6e} {xyz[1]:12.6e} {xyz[2]:12.6e}", file=fout)


if __name__ == "__main__":
    main()
