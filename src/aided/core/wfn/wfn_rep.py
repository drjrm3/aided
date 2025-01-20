"""
aided.core.wfn.wfn_rep

Read Gaussian09 output files.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from dataclasses import dataclass

from ... import np


# pylint: disable=too-many-instance-attributes, R0801
@dataclass
class WFNRep:
    """Structure of Arrays of AIM file / representation using numpy arrays.

    nmos: Number of Molecular Orbitals
    nprims: Number of Gaussian Primitives
    nats: Number of Nuclei (Atoms)
    atnames: Atom names
    atpos: Atomic positions
    atcharge: Atomic charges
    centers: Atomic center upon which each primitive is based
    types: Gaussian primitive type for each atom
    expons: Exponents for each basis function
    occs: Occupancy number for each MO
    energies: Energy of each MO
    coeffs: Coefficients for each MO
    total_energy: Total energy of the system
    virial_energy: Virial energy of the system
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
    occs: np.ndarray      # Occupancy number for each MO. Sized `nmos`. (dtype=float)
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
            "occs", "energies",             # Size is nmos
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

                "occs": self.nmos,
                "energies": self.nmos,

                "coeffs": self.nmos * self.nprims,
            }[param]
            # fmt: on

            if value.size != expected_size:
                raise ValueError(f"`{param}` must have size {expected_size}, but got {value.size}.")
