"""
Test functionality for Bader surface construction.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy
import unittest

from ...helper import CxTestCase, get_wfn_file

from aided.core.edwfn import EDWfn
from aided.math.geometry import spherical_angles_from_vector

class TestTraceGradientToPoint(CxTestCase):
    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()
        self.wfn = EDWfn(self.wfn_file)
        numpy.random.seed(0)

    def test_trace_gradient_to_point_bad_method(self):
        """Test attempt to trace gradient to a point with an invalid method."""

        with self.assertRaises(ValueError):
            self.wfn.trace_gradient_to_atom(0.0, 0.0, 0.0, method="bad_method")


    def test_trace_gradient_to_point(self):
        """Test tracing gradient to a point."""

        for i in range(10):
            for atom, atpos in zip(self.wfn.atnames, self.wfn.atpos):
                eps = 0.5

                x = atpos[0] + numpy.random.uniform(-eps, eps)
                y = atpos[1] + numpy.random.uniform(-eps, eps)
                z = atpos[2]

                grad = self.wfn.grad(x, y, z)

                _atom, _atpos = self.wfn.trace_gradient_to_atom(x, y, z)
                output = "\n\n"
                output += f"[*] {atom}: {atpos[0]:10.5f}, {atpos[1]:10.5f}, {atpos[2]:10.5f}\n"
                output += f"[+] {_atom}: {_atpos[0]:10.5f}, {_atpos[1]:10.5f}, {_atpos[2]:10.5f}\n"
                output += f"        {x:10.5f}, {y:10.5f}, {z:10.5f}\n"
                output += f"Dist({atom}) = {numpy.linalg.norm([x, y, z] - atpos):10.5f}\n"
                output += f"Dist({_atom}) = {numpy.linalg.norm([x, y, z] - _atpos):10.5f}\n"
                output += f"grad = {grad[0]:10.5f}, {grad[1]:10.5f}, {grad[2]:10.5f}\n"
                self.assertEqual(atom, _atom, output)
                self.assertEqual(numpy.linalg.norm(atpos - _atpos), 0.0)

class TestBaderSurfacePoint(CxTestCase):
    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()
        self.wfn = EDWfn(self.wfn_file)

    def test_bader_surface_point_interior(self):
        """Test Bader surface of a point on the interior of the molecule."""
        # Coordinates
        C2 = self.wfn.atpos[ self.wfn.atnames == "C2" ][0]
        N3 = self.wfn.atpos[ self.wfn.atnames == "N3" ][0]

        # Convert difference vectors to (theta, phi)
        theta_c2_n3, phi_c2_n3 = spherical_angles_from_vector(N3 - C2)
        theta_n3_o2, phi_n3_o2 = spherical_angles_from_vector(C2 - N3)

        # Choose a small “starting_radius”
        start_r = 0.3  
        max_r   = 2.5  
        step    = 0.1  
        tol     = 1e-12

        # Find bader surface point in C2 in the direction of N3
        start_pos_c2 = C2 + start_r * numpy.array([
           numpy.sin(theta_c2_n3)*numpy.cos(phi_c2_n3),
           numpy.sin(theta_c2_n3)*numpy.sin(phi_c2_n3),
           numpy.cos(theta_c2_n3)
        ])
        pt_02_n3 = self.wfn._find_bader_surface_point(start_pos_c2, start_r,
                                                      theta_c2_n3, phi_c2_n3,
                                                      max_r, step, tol)

        # Find bader surface point in N3 in the direction of C2
        start_pos_n3 = N3 + start_r * numpy.array([
           numpy.sin(theta_n3_o2)*numpy.cos(phi_n3_o2),
           numpy.sin(theta_n3_o2)*numpy.sin(phi_n3_o2),
           numpy.cos(theta_n3_o2)
        ])
        pt_n3_o2 = self.wfn._find_bader_surface_point(start_pos_n3, start_r,
                                                  theta_n3_o2, phi_n3_o2,
                                                  max_r, step, tol)

        # Ensure they are within tol of each other
        dist = numpy.linalg.norm(pt_02_n3 - pt_n3_o2)
        assert dist < tol, f"Points differ by {dist} Bohr!"

