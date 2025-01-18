"""
aided.fio.read_wfn

Read AIMfile WFN representation files. This reads it in both single file mode and also a batch of
files which it then returns as WFNRep or WFNsRep, respectively.

Copyright (C) J. Robert Michael PhD, 2025
"""

from typing import List

from ... import np, npt
from ..utils import is_number, convert_scientific_notation
from ...core.wfn import WFNRep, WFNsRep


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

    # FIXME: Add logger
    # print(f"Reading file: {wfn_file}")
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
    atnames = np.array(["".join(line.split()[0:2]) for line in lines[:nats]])
    atpos = np.array([line.split()[4:7] for line in lines[:nats]]).astype(np.float32)
    atcharge = np.array([np.float32(line.split()[-1]) for line in lines[:nats]])
    del lines[:nats]

    # Read center assignment integers, exponents, and types for each Gaussian primitive.
    centers = _extract_values(lines, "CENTRE")
    types = _extract_values(lines, "TYPE")
    exponents = _extract_values(lines, "EXPONENTS", dtype=np.float32)

    # Read Molecular Orbitals
    energies = np.zeros(nmos, dtype=np.float32)
    occupations = np.zeros(nmos, dtype=np.float32)
    coeffs = np.zeros((nmos, nprims), dtype=np.float32)
    for imo in range(nmos):
        # Read the MO energy and occupation number
        line = lines.pop(0)
        energies[imo], occupations[imo] = float(line.split()[7]), float(line.split()[11])

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
        occupations=occupations,
        energies=energies,
        coeffs=coeffs,
        total_energy=total_energy,
        virial_energy=virial_energy,
    )

    return wfn_rep


def read_wfn_files(wfns: List[str]) -> WFNsRep:
    """
    Read all of the parameters needed for a WFNRep from a .wfn file.

    Args:
        wfn_file: .wfn file representing an AIM file.

    Returns:
        wfn_rep: A wfn representation as a dataclass representation
    """

    # Get the number of wfns for spacing.
    nwfns = len(wfns)

    # Read the first wfn to get an idea of sizing.
    _wfn_rep = read_wfn_file(wfns[0])
    nmos, nprims, nats = _wfn_rep.nmos, _wfn_rep.nprims, _wfn_rep.nats

    # Space for atnames, atpos, atcharge.
    atnames = np.empty((nwfns, nats), dtype=object)
    atpos = np.zeros((nwfns, nats, 3), dtype=np.float32)
    atcharge = np.zeros((nwfns, nats), dtype=np.int32)

    # Space for center assignment integers, exponents, and types for each Gaussian primitive.
    centers = np.zeros((nwfns, nprims), dtype=np.float32)
    exponents = np.zeros((nwfns, nprims), dtype=np.float32)
    types = np.zeros((nwfns, nprims), dtype=np.int32)

    # Read Molecular Orbitals
    energies = np.zeros((nwfns, nmos), dtype=np.float32)
    occupations = np.zeros((nwfns, nmos), dtype=np.float32)
    coeffs = np.zeros((nwfns, nmos, nprims), dtype=np.float32)

    # Total and virial energy.
    total_energies = np.zeros(nwfns, dtype=np.float32)
    virial_energies = np.zeros(nwfns, dtype=np.float32)

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
        occupations[iwfn, :] = wfn_rep.occupations
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
        occupations=occupations,
        energies=energies,
        coeffs=coeffs,
        total_energies=total_energies,
        virial_energies=virial_energies,
    )

    return wfns_rep
