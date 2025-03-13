"""
aided.io.read_wfn

Read AIMfile WFN representation files. This reads it in both single file mode and also a batch of
files which it then returns as WFNRep or WFNsRep, respectively.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List, Tuple

import numpy

from concurrent.futures import ProcessPoolExecutor

from .. import np, npt
from .utils import is_number, convert_scientific_notation
from ..core.wfn import WFNRep, WFNsRep


def _read_wfn_worker(iwfn: int, wfn: str) -> Tuple:  # pragma: no cover
    """Read a single wfn file and return its data."""
    wfn_rep = read_wfn_file(wfn)
    return (
        iwfn,
        wfn_rep.atnames,
        wfn_rep.atpos,
        wfn_rep.atcharge,
        wfn_rep.centers,
        wfn_rep.expons,
        wfn_rep.types,
        wfn_rep.energies,
        wfn_rep.occs,
        wfn_rep.coeffs,
        wfn_rep.total_energy,
        wfn_rep.virial_energy,
    )


def read_wfn_file(wfn_file: str) -> WFNRep:
    """
    Read all of the parameters needed for a WFNRep from a .wfn file.

    Args:
        wfn_file: .wfn file representing an AIM file.

    Returns:
        wfn_rep: A wfn representation as a dataclass representation
    """

    def _extract_values(lines, key, dtype: npt.DTypeLike = np.int32):
        """Extract values from a list of lines that start with a key."""
        values = []
        while key in lines[0]:
            values.extend([float(x) for x in lines[0].split() if is_number(x)])
            del lines[0]
        return np.array(values, dtype)

    with open(wfn_file, "r", encoding="utf-8") as finp:
        lines = finp.read().splitlines()

    # Delete first line as it is just a comment.
    del lines[0]

    # Take any strings that have `D` in them which represent scientific notation and change to `E`.
    lines = convert_scientific_notation(lines)

    # Read nmos, nprims, nats:
    nmos, nprims, nats = [int(x) for x in lines[0].split() if x.isdigit()]
    del lines[0]

    # Read atnames, atpos, atcharge:
    atnames = numpy.array(["".join(line.split()[0:2]) for line in lines[:nats]])
    atpos = np.array(
        [[float(w) for w in line.split()[4:7]] for line in lines[:nats]]
    )  # * ANG_TO_AU
    atcharge = np.array([float(line.split()[-1]) for line in lines[:nats]])
    del lines[:nats]

    # Read center assignment integers, exponents, and types for each Gaussian primitive.
    centers = _extract_values(lines, "CENTRE") - 1
    types = _extract_values(lines, "TYPE") - 1
    exponents = _extract_values(lines, "EXPONENTS", dtype=float)

    # Read Molecular Orbitals
    energies = np.zeros(nmos, dtype=float)
    occs = np.zeros(nmos, dtype=float)
    coeffs = np.zeros((nmos, nprims), dtype=float)
    for imo in range(nmos):
        # Read the MO energy and occupation number
        line = lines.pop(0)
        occs[imo], energies[imo] = float(line.split()[7]), float(line.split()[11])

        # Read the MO coefficients
        iprim = 0
        while lines and "END DATA" not in lines[0] and "MO" not in lines[0]:
            for x in lines.pop(0).split():
                if is_number(x):
                    coeffs[imo, iprim] = float(x)
                    iprim += 1
    if "END DATA" in lines[0]:
        del lines[0]

    # print(f"DBG: {coeffs=}")

    # Read the total and virial energy
    tokens = lines.pop(0).split()
    total_energy, virial_energy = float(tokens[3]), float(tokens[6])

    # Now save data into WFNRep
    wfn_rep = WFNRep(
        nmos=nmos,
        nprims=nprims,
        nats=nats,
        atnames=atnames,
        atpos=atpos,
        atcharge=atcharge,
        centers=centers,
        types=types,
        expons=exponents,
        occs=occs,
        energies=energies,
        coeffs=coeffs,
        total_energy=total_energy,
        virial_energy=virial_energy,
    )

    return wfn_rep


def read_wfn_files(wfns: List[str], nprocs: int = 1) -> WFNsRep:
    """
    Read all of the parameters needed for a WFNRep from a .wfn file.

    Args:
        wfn_files: A list of .wfn files representing an AIM file.
        nprocs: Number of processors to read with.

    Returns:
        wfns_rep: A representation of multiple wfns as a dataclass
    """

    # Get the number of wfns for spacing.
    nwfns = len(wfns)

    # Read the first wfn to get an idea of sizing.
    _wfn_rep = read_wfn_file(wfns[0])
    nmos, nprims, nats = _wfn_rep.nmos, _wfn_rep.nprims, _wfn_rep.nats

    # Space for atnames, atpos, atcharge.
    atnames = numpy.empty((nwfns, nats), dtype=object)
    atpos = np.zeros((nwfns, nats, 3), dtype=float)
    atcharge = np.zeros((nwfns, nats), dtype=np.int32)

    # Space for center assignment integers, exponents, and types for each Gaussian primitive.
    centers = np.zeros((nwfns, nprims), dtype=np.int32)
    exponents = np.zeros((nwfns, nprims), dtype=float)
    types = np.zeros((nwfns, nprims), dtype=np.int32)

    # Read Molecular Orbitals
    energies = np.zeros((nwfns, nmos), dtype=float)
    occs = np.zeros((nwfns, nmos), dtype=float)
    coeffs = np.zeros((nwfns, nmos, nprims), dtype=float)

    # Total and virial energy.
    total_energies = np.zeros(nwfns, dtype=float)
    virial_energies = np.zeros(nwfns, dtype=float)

    # Split up work across cores.
    if nprocs > 1:
        # Parallel processing with deterministic mapping
        with ProcessPoolExecutor(max_workers=nprocs) as executor:
            # Map iwfn and corresponding wfn to the parallel function
            results = executor.map(_read_wfn_worker, range(nwfns), wfns)

            # Collect results and assign them to the correct array indices
            for (
                iwfn,
                atnames_res,
                atpos_res,
                atcharge_res,
                centers_res,
                exponents_res,
                types_res,
                energies_res,
                occs_res,
                coeffs_res,
                total_energy_res,
                virial_energy_res,
            ) in results:
                atnames[iwfn, :] = atnames_res
                atpos[iwfn, :, :] = atpos_res
                atcharge[iwfn, :] = atcharge_res
                centers[iwfn, :] = centers_res
                exponents[iwfn, :] = exponents_res
                types[iwfn, :] = types_res
                energies[iwfn, :] = energies_res
                occs[iwfn, :] = occs_res
                coeffs[iwfn, :, :] = coeffs_res
                total_energies[iwfn] = total_energy_res
                virial_energies[iwfn] = virial_energy_res
    else:

        # Read all of the wfn files.
        for iwfn, wfn in enumerate(wfns):
            wfn_rep = read_wfn_file(wfn)

            # Save the data into the arrays.
            atnames[iwfn, :] = wfn_rep.atnames
            atpos[iwfn, :, :] = wfn_rep.atpos
            atcharge[iwfn, :] = wfn_rep.atcharge
            centers[iwfn, :] = wfn_rep.centers
            exponents[iwfn, :] = wfn_rep.expons
            types[iwfn, :] = wfn_rep.types
            energies[iwfn, :] = wfn_rep.energies
            occs[iwfn, :] = wfn_rep.occs
            coeffs[iwfn, :, :] = wfn_rep.coeffs
            total_energies[iwfn] = wfn_rep.total_energy
            virial_energies[iwfn] = wfn_rep.virial_energy

    # Save as WFNsRep.
    wfns_rep = WFNsRep(
        nwfns=nwfns,
        nmos=nmos,
        nprims=nprims,
        nats=nats,
        atnames=atnames,
        atpos=atpos,
        atcharge=atcharge,
        centers=centers,
        types=types,
        expons=exponents,
        occs=occs,
        energies=energies,
        coeffs=coeffs,
        total_energies=total_energies,
        virial_energies=virial_energies,
        total_energy=0,
        virial_energy=0,
    )

    return wfns_rep


def _tst():  # pragma: no cover
    # pylint: disable=all
    import argparse
    import sys

    # Get one or more input files.
    parser = argparse.ArgumentParser(description="Test wfn reading.")
    parser.add_argument("-i", "--input", type=str, nargs="+", help="Input wfn file(s)")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if len(args.input) > 1:
        _wfn_rep = read_wfn_files(args.input)
    else:
        _wfns_rep = read_wfn_file(args.input[0])


if __name__ == "__main__":  # pragma: no cover
    _tst()
