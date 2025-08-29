"""
Generate bader surfaces.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from datetime import datetime
from aided import EDWfn

def get_args():
    """Get command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate bader surfaces.",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="../../test/data/wfns/formamide/formamide.6311gss.b3lyp.wfn",
        help="Gaussian output file name.",
    )
    parser.add_argument("--plot", action="store_true", help="Plot the results.")

    args = parser.parse_args()

    return args


def main():
    """Main routine."""

    args = get_args()

    # Read in the Gaussian output file
    filename = args.file

    wfn = EDWfn(filename)

    if args.plot:
        points_dict = {}

    for atom_name, atom_position in zip(wfn.atnames, wfn.atpos):
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

        if args.plot:
            points_dict[atom_name] = {"position": atom_position, "xyzs": _xyzs}

    if args.plot:
        import matplotlib.pyplot as plt
        for atom_name, atom_data in points_dict.items():
            atom_position = atom_data["position"]
            plt.plot(atom_position[0], atom_position[1], "o", label=atom_name)
            xyzs = atom_data["xyzs"]
            xs = [xyz[0] for xyz in xyzs]
            ys = [xyz[1] for xyz in xyzs]

            plt.plot(xs, ys, ".")

        plt.show()


if __name__ == "__main__":
    main()
