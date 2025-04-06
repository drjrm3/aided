"""
aided.core.wfn.wfn_rep

Read AIMfile output files.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from dataclasses import dataclass

from ... import np, npt


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
    atnames: npt.NDArray[np.object_]  # Nuclei names. Sized `nats`.
    atpos: npt.NDArray[np.float64]     # Nuclei positions. Sized `3*nats`.
    atcharge: npt.NDArray[np.int_]    # Nuclei charges. Sized `nats`.

    # Specific to the Gaussian primitives. All have size `nprims`.
    centers: npt.NDArray[np.int_]    # Center of each primitive. Sized `nprims`.
    types: npt.NDArray[np.int_]      # Gaussian primitive type for each atom. Sized `nprims`.
    expons: npt.NDArray[np.float64]   # Exponents for each basis function. Sized `nprims`.

    # Specific to the molecular orbitals
    occs: npt.NDArray[np.float64]      # Occupancy number for each MO. Sized `nmos`.
    energies: npt.NDArray[np.float64]  # Energy of each MO. Sized `nmos`.
    coeffs: npt.NDArray[np.float64]    # Coefficients for each MO. Sized `nmos x nprims`.

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

    def __eq__(self, other) -> bool:
        """Equality comparison for WFNRep."""
        if not isinstance(other, WFNRep):
            return False

        # fmt: off
        return (
            self.nmos == other.nmos and
            self.nprims == other.nprims and
            self.nats == other.nats and
            np.array_equal(self.atnames, other.atnames) and
            np.array_equal(self.atpos, other.atpos) and
            np.array_equal(self.atcharge, other.atcharge) and
            np.array_equal(self.centers, other.centers) and
            np.array_equal(self.types, other.types) and
            np.array_equal(self.expons, other.expons) and
            np.array_equal(self.occs, other.occs) and
            np.array_equal(self.energies, other.energies) and
            np.array_equal(self.coeffs, other.coeffs)
        )
        # fmt: on
