"""
aided.core.wfn.wfns_rep

Read Gaussian09 output files.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from dataclasses import dataclass

from ... import np


# pylint: disable=too-many-instance-attributes, R0801
@dataclass
class WFNsRep:
    """
    Structure of Arrays of AIM file / representation using numpy arrays.

    Specificlaly this is every WFN kept as a unique element in the array. Even if some values are
    redundant like the locations of the atoms, they are kept so that this can be used arbirtarily.

    nwfns: Number of WFNs represented
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
    nwfns: int   # Number of WFNs represented.
    nmos: int    # Number of Molecular Orbitals
    nprims: int  # Number of Gaussian Primitives
    nats: int    # Number of Nuclei (Atoms)

    # Specific to the Atoms. All have size as a function of the number of atoms.
    atnames: np.ndarray   # Nuclei names.
    atpos: np.ndarray     # Nuclei positions.
    atcharge: np.ndarray  # Nuclei charges.

    # Specific to the Gaussian primitives.
    centers: np.ndarray  # Center of each primitive.
    types: np.ndarray    # Gaussian primitive type for each atom.
    expons: np.ndarray   # Exponents for each basis function.

    # Specific to the molecular orbitals
    occs: np.ndarray      # Occupancy number for each MO.
    energies: np.ndarray  # Energy of each MO. Sized `nmos`.
    coeffs: np.ndarray    # Coefficients for each MO.

    # Energy in the system.
    total_energies: np.ndarray
    virial_energies: np.ndarray
    # fmt: on

    def __post_init__(self):
        # Validate sizes
        # fmt: off
        nat_params = [
            "atnames", "atcharge", "atpos", # Size based on nats.
            "centers", "expons", "types",   # Size based off of nprims,
            "occs", "energies",      # Size is nmos
            "coeffs",                       # Size based on both nmos and nprims
            ]
        # fmt: on
        for param in nat_params:
            value = getattr(self, param)
            # fmt: off
            expected_size = {
                "atcharge": self.nwfns * self.nats,
                "atnames":  self.nwfns * self.nats,
                "atpos":    self.nwfns * self.nats * 3,

                "centers": self.nwfns * self.nprims,
                "expons":  self.nwfns * self.nprims,
                "types":   self.nwfns * self.nprims,

                "occs": self.nwfns * self.nmos,
                "energies": self.nwfns * self.nmos,

                "coeffs": self.nwfns * self.nmos * self.nprims,

                "total_energies": self.nwfns,
                "virial_energies": self.nwfns,
            }[param]
            # fmt: on

            if value.size != expected_size:
                raise ValueError(f"`{param}` must have size {expected_size}, but got {value.size}.")
