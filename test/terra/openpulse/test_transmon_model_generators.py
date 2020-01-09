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
from qiskit.providers.aer.openpulse import transmon_model_generators as model_gen

class TestTransmonModelGenerators(QiskitAerTestCase):
    """Tests for functions in pulse_model_generators.py"""

    def setUp(self):
        pass

    def test_transmon_hamiltonian_dict(self):
        """Test _transmon_hamiltonian_dict"""

        transmons = [0, 1]
        transmon_dims = [2, 2]
        transmon_freqs = [5.0, 5.1]
        freq_symbols = ['v0', 'v1']
        anharm_freqs = [-0.33, -0.33]
        anharm_symbols = ['a0', 'a1']
        drive_strengths = [1.1, 1.2]
        drive_symbols = ['r0', 'r1']
        ordered_coupling_edges = [(0,1)]
        coupling_strengths = [0.02]
        coupling_symbols = ['j']
        cr_idx_dict = {(0,1): 0}

        expected = {'h_str': ['np.pi*(2*v0-a0)*O0','np.pi*(2*v1-a1)*O1',
                              'np.pi*a0*O0*O0', 'np.pi*a1*O1*O1',
                              '2*np.pi*r0*X0||D0', '2*np.pi*r1*X1||D1',
                              '2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)',
                              '2*np.pi*r0*X0||U0'],
                    'vars': {'v0': 5.0, 'v1': 5.1,
                             'a0': -0.33, 'a1' : -0.33,
                             'r0': 1.1, 'r1' : 1.2,
                             'j' : 0.02},
                    'qub': {'0': 2, '1': 2}}
        output = model_gen._transmon_hamiltonian_dict(transmons,
                                                      transmon_dims,
                                                      transmon_freqs,
                                                      freq_symbols,
                                                      anharm_freqs,
                                                      anharm_symbols,
                                                      drive_strengths,
                                                      drive_symbols,
                                                      ordered_coupling_edges,
                                                      coupling_strengths,
                                                      coupling_symbols,
                                                      cr_idx_dict)
        self._compare_str_lists(output['h_str'], expected['h_str'])
        self.assertEqual(output['vars'], expected['vars'])
        self.assertEqual(output['qub'], expected['qub'])

        # test 3 transmons with mixed up inputs
        transmons = [0, 1, 2]
        transmon_dims = [2, 2, 3]
        transmon_freqs = [5.0, 5.1, 4.9]
        freq_symbols = ['v0', 'v1', 'x3']
        anharm_freqs = [-0.33, -0.33, 1.]
        anharm_symbols = ['a0', 'a1', 'z4']
        drive_strengths = [1.1, 1.2, 0.98]
        drive_symbols = ['r0', 'r1', 'sa']
        ordered_coupling_edges = [(0,1), (1,2), (2,0)]
        coupling_strengths = [0.02, 0.1, 0.33]
        coupling_symbols = ['j', 's', 'r']
        cr_idx_dict = {(0,1): 0, (2,0): 1, (1,2): 2}

        expected = {'h_str': ['np.pi*(2*v0-a0)*O0','np.pi*(2*v1-a1)*O1', 'np.pi*(2*x3-z4)*O2',
                              'np.pi*a0*O0*O0', 'np.pi*a1*O1*O1', 'np.pi*z4*O2*O2',
                              '2*np.pi*r0*X0||D0', '2*np.pi*r1*X1||D1', '2*np.pi*sa*X2||D2',
                              '2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)','2*np.pi*s*(Sp1*Sm2+Sm1*Sp2)',
                              '2*np.pi*r*(Sp0*Sm2+Sm0*Sp2)',
                              '2*np.pi*r0*X0||U0', '2*np.pi*sa*X2||U1', '2*np.pi*r1*X1||U2'],
                    'vars': {'v0': 5.0, 'v1': 5.1, 'x3': 4.9,
                             'a0': -0.33, 'a1' : -0.33, 'z4': 1.,
                             'r0': 1.1, 'r1' : 1.2, 'sa': 0.98,
                             'j' : 0.02, 's': 0.1, 'r': 0.33},
                    'qub': {'0': 2, '1': 2, '2': 3}}
        output = model_gen._transmon_hamiltonian_dict(transmons,
                                                      transmon_dims,
                                                      transmon_freqs,
                                                      freq_symbols,
                                                      anharm_freqs,
                                                      anharm_symbols,
                                                      drive_strengths,
                                                      drive_symbols,
                                                      ordered_coupling_edges,
                                                      coupling_strengths,
                                                      coupling_symbols,
                                                      cr_idx_dict)
        self._compare_str_lists(output['h_str'], expected['h_str'])
        self.assertEqual(output['vars'], expected['vars'])
        self.assertEqual(output['qub'], expected['qub'])



    def test_cr_lo_list(self):
        """Test _cr_lo_list"""

        cr_dict = {(0,1): 0, (1,0) : 1, (3,4) : 2}
        expected = [[{'scale' : [1.0, 0], 'q' : 1}],
                    [{'scale' : [1.0, 0], 'q' : 0}],
                    [{'scale' : [1.0, 0], 'q' : 4}]]
        self.assertEqual(model_gen._cr_lo_list(cr_dict), expected)

        cr_dict = {(0,1): 0, (3,4) : 2, (1,0) : 1}
        expected = [[{'scale' : [1.0, 0], 'q' : 1}],
                    [{'scale' : [1.0, 0], 'q' : 0}],
                    [{'scale' : [1.0, 0], 'q' : 4}]]
        self.assertEqual(model_gen._cr_lo_list(cr_dict), expected)

    def test_single_term_generators(self):
        """Test various functions for individual terms:
        _single_transmon_drift_terms, _drive_terms, _exchange_coupling_terms, _cr_terms
        """

        # single transmon terms
        self.assertEqual(model_gen._single_transmon_drift_terms(freq_symbols='v',
                                                                anharm_symbols='a',
                                                                transmon_list=0),
                         ['np.pi*(2*v-a)*O0', 'np.pi*a*O0*O0'])
        self.assertEqual(model_gen._single_transmon_drift_terms(freq_symbols=['v0','v1'],
                                                                anharm_symbols=['a0','a1'],
                                                                transmon_list=[2, 3]),
                         ['np.pi*(2*v0-a0)*O2',
                          'np.pi*(2*v1-a1)*O3',
                          'np.pi*a0*O2*O2',
                          'np.pi*a1*O3*O3'])

        # drive terms
        self.assertEqual(model_gen._drive_terms(drive_symbols='r', transmon_list=0),
                         ['2*np.pi*r*X0||D0'])
        self.assertEqual(model_gen._drive_terms(drive_symbols=['r0', 'r1'], transmon_list=[1, 2]),
                         ['2*np.pi*r0*X1||D1', '2*np.pi*r1*X2||D2'])

        # exchange coupling
        symbols = 'j'
        edges = [(0,1)]
        expected = ['2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)']
        self.assertEqual(model_gen._exchange_coupling_terms(symbols, edges), expected)
        symbols = ['j','k']
        edges = [(0,1), (3,2)]
        expected = ['2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)', '2*np.pi*k*(Sp3*Sm2+Sm3*Sp2)']
        self.assertEqual(model_gen._exchange_coupling_terms(symbols, edges), expected)

        # cross resonance terms
        symbols = 'r'
        driven_indices = 0
        u_channel_indices = [1]
        expected = ['2*np.pi*r*X0||U1']
        self.assertEqual(model_gen._cr_terms(symbols, driven_indices, u_channel_indices), expected)
        symbols = ['r','s']
        driven_indices = [0,3]
        u_channel_indices = [1,1]
        expected = ['2*np.pi*r*X0||U1', '2*np.pi*s*X3||U1']
        self.assertEqual(model_gen._cr_terms(symbols, driven_indices, u_channel_indices), expected)

    def test_str_list_generator(self):
        """Test _str_list_generator"""

        # test one argument
        template = 'First: {0}'
        self.assertEqual(model_gen._str_list_generator(template, 'a'), ['First: a'])
        self.assertEqual(model_gen._str_list_generator(template, ['a1', 'a2']),
                         ['First: a1', 'First: a2'])

        # test multiple arguments
        template = 'First: {0}, Second: {1}'
        self.assertEqual(model_gen._str_list_generator(template, 'a', 'b'),
                         ['First: a, Second: b'])
        self.assertEqual(model_gen._str_list_generator(template, ['a1', 'a2'], ['b1', 'b2']),
                         ['First: a1, Second: b1', 'First: a2, Second: b2'])


    def test_arg_to_iterable(self):
        """Test _arg_to_iterable."""

        self.assertEqual(model_gen._arg_to_iterable('a'), ['a'])
        self.assertEqual(model_gen._arg_to_iterable(['a']), ['a'])
        self.assertEqual(model_gen._arg_to_iterable(('a','b')), ('a','b'))
        self.assertEqual(model_gen._arg_to_iterable({'a','b'}), {'a','b'})

    def test_coupling_graph(self):
        """Test _coupling_graph class."""

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

    def _compare_str_lists(self, list1, list2):
        """Helper function for checking that the contents of string lists are the same when order
        doesn't matter.

        Args:
            list1 (list): A list of strings
            list2 (list): A list of strings
        """

        list1_copy = list1.copy()
        list2_copy = list1.copy()
        self.assertEqual(len(list1_copy), len(list2_copy))
        list1_copy.sort()
        list2_copy.sort()
        for str1, str2 in zip(list1_copy, list2_copy):
            self.assertEqual(str1, str2)
