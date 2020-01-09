# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests for pulse system generator functions
"""

import unittest
from test.terra.common import QiskitAerTestCase
from qiskit.providers.aer.openpulse.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.openpulse.hamiltonian_model import HamiltonianModel
from qiskit.providers.aer.openpulse import pulse_model_generators as model_gen

class TestPulseModelGenerators(QiskitAerTestCase):
    """Tests for functions in pulse_model_generators.py"""

    def setUp(self):
        pass

    def test_coupling_graph(self):
        """Test _coupling_graph class"""

        coupling_graph = model_gen._coupling_graph([(0,1), (1,0), (3,2), (1,2)])

        # test constructor, including catching of duplicate entries
        self.assertEqual(len(coupling_graph.graph), 3)
        self.assertEqual(coupling_graph.sorted_graph, [(0,1), (1,2), (2,3)])
        self.assertEqual(coupling_graph.sorted_two_way_graph,
                         [(0,1), (1,0), (1,2), (2,1), (2,3), (3,2)])
        self.assertEqual(coupling_graph.two_way_graph_dict,
                         {(0,1): 0, (1,0) : 1, (1,2) : 2, (2,1) : 3, (2,3) : 4, (3,2) : 5})

        # test sorted_edge_index, and that it treats (1,2) and (2,1) as the same edge
        self.assertEqual(coupling_graph.sorted_edge_index((1,2)), 1)
        self.assertEqual(coupling_graph.sorted_edge_index((2,1)), 1)

        # test two_way_edge_index, and that it treats (1,2) and (2,1) as different
        self.assertEqual(coupling_graph.two_way_edge_index((1,2)), 2)
        self.assertEqual(coupling_graph.two_way_edge_index((2,1)), 3)
