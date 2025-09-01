"""
Dynamic Electron Density test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np

from aided.core.ed.wfn_dynamic import EDWfnDynamic, EDWfnStatic

from conftest import MSDA_FILES_DIR
from ut_helper import CxTestCase


NUM_ITERS = 100


class WfnDynamicMSDAGeneration(CxTestCase):
    """Tests MSDA generation for dynamic wavefunction."""

    def set_up(self):
        """Set up the test case."""
        form_base = "formamide.b3lyp.6311gss"
        self.wfn_file = MSDA_FILES_DIR / f"g09/{form_base}.wfn"
        self.log_file = MSDA_FILES_DIR / f"g09/{form_base}.log"
        self.msda_file = MSDA_FILES_DIR / f"g09/{form_base}.msda"

        self.edwfn_static = EDWfnStatic(self.wfn_file.as_posix())
        self.natoms = len(self.edwfn_static.atnames)
        self.msda_matrix = np.random.rand(3 * self.natoms, 3 * self.natoms) * 2 - 1

    def test_invalid_msda_entries(self):
        """
        Tests that we properly protect against building the ed from too many or too few MSDA args
        """

        # A floating value of T must be passed.
        with self.assertRaises(TypeError):
            # pylint: disable=no-value-for-parameter
            _edwfn = EDWfnDynamic(self.wfn_file.as_posix())

        # At least one argument must be passed for MSDA.
        with self.assertRaises(ValueError):
            _edwfn = EDWfnDynamic(self.wfn_file.as_posix(), 0.0)

        # Raise on bad shape entry for MSDA matrix.
        with self.assertRaises(ValueError):
            _edwfn = EDWfnDynamic(
                self.wfn_file.as_posix(),
                0.0,
                msda_matrix=np.random.rand(self.natoms, self.natoms),
            )

        # No more than one argument can be passed for MSDA.
        with self.assertRaises(ValueError):
            _edwfn = EDWfnDynamic(
                self.wfn_file.as_posix(),
                0.0,
                msda_matrix=self.msda_matrix,
                msda_file=self.msda_file,
            )

        # No more than one argument can be passed for MSDA.
        with self.assertRaises(ValueError):
            _edwfn = EDWfnDynamic(
                self.wfn_file.as_posix(),
                0.0,
                msda_matrix=self.msda_matrix,
                log_file=self.log_file,
            )

        # No more than one argument can be passed for MSDA.
        with self.assertRaises(ValueError):
            _edwfn = EDWfnDynamic(
                self.wfn_file.as_posix(),
                0.0,
                msda_file=self.msda_file,
                log_file=self.log_file,
            )

        # No more than one argument can be passed for MSDA.
        with self.assertRaises(ValueError):
            _edwfn = EDWfnDynamic(
                self.wfn_file.as_posix(),
                0.0,
                msda_matrix=self.msda_matrix,
                msda_file=self.msda_file,
                log_file=self.log_file,
            )

    def test_msda_from_matrix(self):
        """Tests MSDA generation from a specific matrix input."""

        edwfn = EDWfnDynamic(self.wfn_file.as_posix(), 0.0, msda_matrix=self.msda_matrix)

        # Assert that edwfn.msda is the same as self.msda_matrix
        np.testing.assert_array_almost_equal(edwfn.msda, self.msda_matrix, decimal=14)

    def test_msda_from_msda_file(self):
        """Tests the generation of MSDA from a file."""

        edwfn = EDWfnDynamic(self.wfn_file.as_posix(), 0.0, msda_file=self.msda_file)

        # Assert that edwfn.msda has correct shape.
        self.assert_block_diagonal_msda_entries(edwfn.msda, self.natoms)

    def test_msda_from_log_file(self):
        """Tests the generation of MSDA from a log file."""

        edwfn = EDWfnDynamic(self.wfn_file.as_posix(), 0.0, log_file=self.log_file)

        # Assert that edwfn.msda has correct shape.
        self.assert_block_diagonal_msda_entries(edwfn.msda, self.natoms)
