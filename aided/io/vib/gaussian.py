"""
aided.io.vib.gaussian

Read vibrational data from Gaussian log files.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List

from pathlib import Path

from aided import np, npt
from aided.core.units import THERM_FACTOR_CM_K, ZPE_PREF_ANG2
from aided import get_logger
from aided.math.primitives import coth

from .log_reader import LogReader

logger = get_logger()


class GaussianLogReader(LogReader):
    """Reads Gaussian log files and extracts vibrational data."""

    def __init__(self, logfile: str | Path):
        super().__init__(logfile)

        # Frequencies (cm**-1)
        self.freqs: npt.NDArray[np.float64]
        # Normal modes (mass-weighted, orthonormal)
        self.modes: npt.NDArray[np.float64]
        # IR intensities (km/mol)
        self.ir_intensities: npt.NDArray[np.float64]
        # Raman activities (A**4/amu)
        self.raman_activities: npt.NDArray[np.float64]
        # Reduced masses (amu)
        self.reduced_masses: npt.NDArray[np.float64]
        # Force constants (mDyne/Å)
        self.force_constants: npt.NDArray[np.float64]

        # Preprocess the file to get pertinent information.
        self.__read_input_orientation()
        self.__freq_blocks = self._get_frequency_blocks()
        self.__parse_values()

    @property
    def nmodes(self) -> int:
        """Number of vibrational modes."""
        last_freq_block = self.__freq_blocks[-1]
        assert len(last_freq_block) > 0, "Last frequency block is empty."

        return int(last_freq_block[0].split()[-1])

    def __read_input_orientation(self):
        """Reads the "Input orientation" section and returns atomic numbers and coordinates.

        Content will be of the form:
                                         Input orientation:
        ---------------------------------------------------------------------
        Center     Atomic      Atomic             Coordinates (Angstroms)
        Number     Number       Type             X           Y           Z
        ---------------------------------------------------------------------
             1          1           0       -0.461860    1.425760    0.000000
             2          6           0        0.000000    0.418660    0.000000
             3          7           0       -0.936360   -0.569290    0.000000
             4          1           0       -0.634660   -1.531610    0.000000
             5          1           0       -1.920810   -0.362470    0.000000
             6          8           0        1.196480    0.242680    0.000000
        ---------------------------------------------------------------------
        """
        logger.debug("Reading Input orientation section...")

        # If we have already found the atomic numbers and coordinates, skip.
        if self.atomic_numbers or self.coordinates:  # pragma: no cover
            return

        # Find the line number that contains 'Input orientation:'
        idx = next(i for i, line in enumerate(self.lines) if "Input orientation:" in line)

        # Increment idx until we find two lines with dashes
        dash_count = 0
        while dash_count < 2:
            if "---" in self.lines[idx]:
                dash_count += 1
            idx += 1

        # Now idx is at the first line of atomic data. Read atomic data until the next dashed line.
        while "---" not in self.lines[idx]:
            _, atomic_number, _, x, y, z = self.lines[idx].split()
            self.atomic_numbers.append(int(atomic_number))
            self.coordinates.append((float(x), float(y), float(z)))
            idx += 1

    def _get_frequency_blocks(self) -> List[List[str]]:
        """Reads the log file and extracts blocks of lines containing frequency data.

        Each frequency block will be of the form:

                                   1         2         3         4         5
                                  A"        A'        A"        A"        A'
               Frequencies ---   216.7639  567.5881  650.7612 1048.2940 1055.2946
            Reduced masses ---     1.2210    2.4393    1.2123    1.5463    1.7853
           Force constants ---     0.0338    0.4630    0.3025    1.0012    1.1714
            IR Intensities ---   227.2889   11.8185   20.1891    0.7114    4.0825
         Coord Atom Element:
           1     1     1          0.00000   0.05229   0.00000   0.00000  -0.30943
           2     1     1          0.00000   0.23299   0.00000   0.00000  -0.12980
           3     1     1          0.01703   0.00000   0.21736   0.96116   0.00000
           1     2     6          0.00000  -0.07220   0.00000   0.00000  -0.04413
           2     2     6          0.00000   0.18678   0.00000   0.00000  -0.03085
           3     2     6         -0.01324   0.00000   0.10343  -0.20870   0.00000
           1     3     7          0.00000   0.14792   0.00000   0.00000   0.19809
           2     3     7          0.00000   0.04012   0.00000   0.00000   0.02891
           3     3     7          0.12636   0.00000  -0.04917   0.03495   0.00000
           1     4     1          0.00000   0.72865   0.00000   0.00000  -0.38825
           2     4     1          0.00000   0.22064   0.00000   0.00000  -0.14404
           3     4     1         -0.51974   0.00000   0.77653   0.16834   0.00000
           1     5     1          0.00000   0.03663   0.00000   0.00000   0.35139
           2     5     1          0.00000  -0.50270   0.00000   0.00000   0.73108
           3     5     1         -0.84450   0.00000  -0.57700   0.01134   0.00000
           1     6     8          0.00000  -0.12685   0.00000   0.00000  -0.11849
           2     6     8          0.00000  -0.17216   0.00000   0.00000  -0.03097
           3     6     8         -0.01580   0.00000  -0.06082   0.05410   0.00000

        Returns:
            List of frequency blocks, each block is a list of strings (lines).
        """

        logger.debug("Extracting frequency blocks...")

        # Get the line numbers of lines that contain '   Frequencies ---
        freq_idxs = [i for i, line in enumerate(self.lines) if "   Frequencies ---" in line]

        # Loop over all frequency line numbers and extract the blocks.
        freq_blocks = []
        for idx in freq_idxs:
            block = []
            # Start at two lines before to get the eigenvector number. Go until natoms * 3 and also
            # include the next four lines (Reduced masses, Force constants, IR Intensities,
            # Coord Atom Element)
            for jdx in range(idx - 2, idx + 4 + self.natoms * 3 + 1):
                block.append(self.lines[jdx])
            freq_blocks.append(block)

        return freq_blocks

    def __parse_values(self):
        """Extract frequencies, normal modes, and other vibrational data from the frequency blocks.

        Args:
            T (float): Temperature in Kelvin for MSDA calculation. Default is 298.15 K.

        Returns:
            freqs (np.ndarray): Frequencies in cm‑1, shape (n_mode,).
            modes (np.ndarray): Eigenvectors, shape (3N, n_mode).
        """

        def __read_values(line: str) -> npt.NDArray[np.float64]:
            """Reads a line of the form 'Label --- val1 val2 ...'"""
            tokens = line.split()
            dash_idx = tokens.index("---")
            return np.array([float(s) for s in tokens[dash_idx + 1 :]])

        self.freqs = np.empty(self.nmodes)
        self.modes = np.zeros((self.natoms * 3, self.nmodes))
        self.reduced_masses = np.empty(self.nmodes)
        self.force_constants = np.empty(self.nmodes)
        self.ir_intensities = np.empty(self.nmodes)
        self.raman_activities = np.empty(self.nmodes)

        for frq_block in self.__freq_blocks:
            eig_nums = np.array([int(s) - 1 for s in frq_block[0].split()])

            self.freqs[eig_nums] = __read_values(frq_block[2])
            self.reduced_masses[eig_nums] = __read_values(frq_block[3])
            self.force_constants[eig_nums] = __read_values(frq_block[4])
            self.ir_intensities[eig_nums] = __read_values(frq_block[5])

            # Read normal modes.
            for i, line in enumerate(frq_block[7:]):
                self.modes[i, eig_nums] = np.array([float(s) for s in line.split()[3:]])

    def gen_msda(self, temperature: float) -> npt.NDArray[np.float64]:
        """Generate mean square displacement amplitudes (MSDA) at temperature T.

        Args:
            temperature (float): Temperature in Kelvin.

        Returns:
            msda (np.ndarray): Mean square displacement amplitudes in Å², shape (n_mode,).
        """

        delta = ZPE_PREF_ANG2 / self.freqs / self.reduced_masses  # Å²
        if temperature > 1e-9:
            delta *= coth(THERM_FACTOR_CM_K * self.freqs / temperature)

        # modes : (3N, n_mode)        columns = √amu-weighted eigen-vectors
        # delta : (n_mode,)           scalar prefactor for each mode
        msda = np.einsum("is,s,js->ij", self.modes, delta, self.modes)

        return msda
