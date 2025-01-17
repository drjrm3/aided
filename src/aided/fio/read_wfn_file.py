"""
aided.fio.read_aimfile

Read Gaussian09 output files.

Copyright (C) J. Robert Michael PhD, 2025
"""

from dataclasses import dataclass

from .. import np, npt

from .utils import is_number, convert_scientific_notation


def read_wfn_file(wfn_file: str):
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
    print(f"Reading file: {wfn_file}")
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

    # Read Center assignment integers, Gaussian primitive types, and exponents
    centers = _extract_values(lines, "CENTRE")
    types = _extract_values(lines, "TYPE")
    exponents = _extract_values(lines, "EXPONENTS", dtype=np.float32)

    ### Read Molecular Orbitals
    energies = np.zeros(nmos)
    occupations = np.zeros(nmos)
    coeffs = np.zeros((nmos, nprims))
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

    ### Read the total energy
    tokens = lines.pop(0).split()
    total_energy, virial_energy = float(tokens[3]), float(tokens[6])

    ### Now save data into WFNRep
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


# pylint: disable=too-many-instance-attributes
@dataclass
class WFNRep:
    """
    Structure of Arrays of AIM file / representation using numpy arrays. If 'Nuclear shaking' alone
    occurs, the only thing to change here would be the `atpos` of each resulting molecule.
    """

    # fmt: off
    # Header information to define sizes of the rest
    nmos: int    # Number of Molecular Orbitals
    nprims: int  # Number of Gaussian Primitives
    nats: int    # Number of Nuclei (Atoms)

    # Specific to the Atoms. All have size as a function of the number of atoms.
    atnames: np.ndarray   # Nuclei names. Sized `nats`. (dtype=str or object)
    atpos: np.ndarray     # Nuclei positions. Sized `3*nats`. (dtype=float)
    atcharge: np.ndarray  # Nuclei charges. Sized `nats`. (dtype=int)

    # Specific to the Gaussian primitives. All have size `nprims`.
    centers: np.ndarray  # Center of each primitive. Sized `nprims`. (dtype=int)
    types: np.ndarray    # Gaussian primitive type for each atom. Sized `nprims`. (dtype=int)
    expons: np.ndarray   # Exponents for each basis function. Sized `nprims`. (dtype=float)

    # Specific to the molecular orbitals
    occupations: np.ndarray  # Occupancy number for each MO. Sized `nmos`. (dtype=float)
    energies: np.ndarray  # Energy of each MO. Sized `nmos`. (dtype=float)
    coeffs: np.ndarray    # Coefficients for each MO. Sized `nmos x nprims`. (dtype=float)

    # Energy in the system.
    total_energy: float
    virial_energy: float
    # fmt: on

    def __post_init__(self):
        # Validate sizes
        # fmt: off
        nat_params = [
            "atnames", "atcharge", "atpos", # Size based on nats.
            "centers", "expons", "types",   # Size based off of nprims,
            "occupations", "energies",      # Size is nmos
            "coeffs",                       # Size based on both nmos and nprims
            ]
        # fmt: on
        for param in nat_params:
            value = getattr(self, param)
            # fmt: off
            expected_size = {
                "atcharge": self.nats,
                "atnames":  self.nats,
                "atpos":    self.nats * 3,

                "centers": self.nprims,
                "expons":  self.nprims,
                "types":   self.nprims,

                "occupations": self.nmos,
                "energies": self.nmos,

                "coeffs": self.nmos * self.nprims,
            }[param]
            # fmt: on

            if value.size != expected_size:
                raise ValueError(f"`{param}` must have size {expected_size}, but got {value.size}.")


if __name__ == "__main__":
    FILE = "/home/drjrm3/code/aided/contrib/wfns/formamide.6311gss.b3lyp.wfn"
    wfnrep = read_wfn_file(FILE)
    print(wfnrep)
