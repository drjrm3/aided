#!/usr/bin/env python3
"""
Analyzes Bond Critical Points (BCPs) from a Gaussian output file.

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

    # Get the BCPs
    bcps, bond_pairs = wfn.find_bcps()

    t1 = datetime.now()
    for bond_pair, bcp in zip(bond_pairs, bcps):
        print(f"[*] Bond {bond_pair[0]}-{bond_pair[1]}")
        bcp_coord = [x for x in bcp.x]
        print(f"bcp = {bcp_coord[0]:.6f}, {bcp_coord[1]:.6f}, {bcp_coord[2]:.6f}")
        rho = wfn.rho(*bcp.x)
        print(f"Rho at bcp .......... {rho:.6f}")
        grad = wfn.grad(*bcp.x)
        print(f"|Gradient| at bcp ... {norm(grad):.6e}")
        hess = wfn.hess(*bcp.x)
        print(f"Hessian at bcp ......")
        hxx = hess[0]
        hyy = hess[1]
        hzz = hess[2]
        hxy = hess[3]
        hxy = hess[4]
        hyz = hess[5]

        print(f"{hxx:10.6f} {hxy:10.6f} {hxy:10.6f}")
        print(f"{hxy:10.6f} {hyy:10.6f} {hyz:10.6f}")
        print(f"{hxy:10.6f} {hyz:10.6f} {hzz:10.6f}")
    t2 = datetime.now()
    tdiff = (t2 - t1).total_seconds()
    print(f"[*] Time taken: {tdiff:10.5f} seconds")

if __name__ == "__main__":
    main()
