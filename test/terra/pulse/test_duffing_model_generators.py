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
from numpy import array, array_equal, kron
from test.terra.common import QiskitAerTestCase
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.providers.aer.pulse.system_models import duffing_model_generators as model_gen
from qiskit.providers.aer.pulse.qutip_extra_lite.qobj_generators import get_oper
from qiskit.providers.models.backendconfiguration import UchannelLO

class TestDuffingModelGenerators(QiskitAerTestCase):
    """Tests for functions in duffing_model_generators.py"""

    def setUp(self):
        pass

    def test_duffing_system_model1(self):
        """First test of duffing_system_model, 2 qubits, 2 dimensional"""

        dim_oscillators = 2
        oscillator_freqs = [5.0, 5.1]
        anharm_freqs = [-0.33, -0.33]
        drive_strengths = [1.1, 1.2]
        coupling_dict = {(0,1): 0.02}
        dt = 1.3

        system_model = model_gen.duffing_system_model(dim_oscillators,
                                                      oscillator_freqs,
                                                      anharm_freqs,
                                                      drive_strengths,
                                                      coupling_dict,
                                                      dt)
        cr_idx_dict = {label: idx for idx, label in enumerate(system_model.control_channel_labels)}

        # check basic parameters
        self.assertEqual(system_model.subsystem_list, [0, 1])
        self.assertEqual(system_model.dt, 1.3)

        # check that cr_idx_dict is correct
        self.assertEqual(cr_idx_dict, {(0,1): 0, (1,0): 1})
        self.assertEqual(system_model.control_channel_index((0,1)), 0)

        # check u_channel_lo is correct
        self.assertEqual(system_model.u_channel_lo,
                         [[UchannelLO(1, 1.0+0.0j)], [UchannelLO(0, 1.0+0.0j)]])

        # check consistency of system_model.u_channel_lo with cr_idx_dict
        # this should in principle be redundant with the above two checks
        for q_pair, idx in cr_idx_dict.items():
            self.assertEqual(system_model.u_channel_lo[idx],
                             [UchannelLO(q_pair[1], 1.0+0.0j)])

        # check correct hamiltonian
        ham_model = system_model.hamiltonian
        expected_vars = {'v0': 5.0, 'v1': 5.1,
                         'alpha0': -0.33, 'alpha1': -0.33,
                         'r0': 1.1, 'r1': 1.2,
                         'j01': 0.02}
        self.assertEqual(ham_model._variables, expected_vars)
        self.assertEqual(ham_model._subsystem_dims, {0: 2, 1: 2})
        self._compare_str_lists(list(ham_model._channels), ['D0', 'D1', 'U0', 'U1'])

        # check that Hamiltonian terms have been imported correctly
        # constructing the expected_terms requires some knowledge of how the strings get generated
        # and then parsed
        O0 = self._operator_array_from_str(2, ['I', 'O'])
        O1 = self._operator_array_from_str(2, ['O', 'I'])
        OO0 = O0@O0
        OO1 = O1@O1
        X0 = self._operator_array_from_str(2, ['I', 'X'])
        X1 = self._operator_array_from_str(2, ['X', 'I'])
        exchange = self._operator_array_from_str(2, ['Sm', 'Sp']) + self._operator_array_from_str(2, ['Sp', 'Sm'])
        expected_terms = [('np.pi*(2*v0-alpha0)', O0),
                          ('np.pi*(2*v1-alpha1)', O1),
                          ('np.pi*alpha0', OO0),
                          ('np.pi*alpha1', OO1),
                          ('2*np.pi*r0*D0', X0),
                          ('2*np.pi*r1*D1', X1),
                          ('2*np.pi*j01', exchange),
                          ('2*np.pi*r0*U0', X0),
                          ('2*np.pi*r1*U1', X1)]

        # first check the number of terms is correct, then loop through
        # each expected term and verify that it is present and consistent
        self.assertEqual(len(ham_model._system), len(expected_terms))
        for expected_string, expected_op in expected_terms:
            idx = 0
            found = False
            while idx < len(ham_model._system) and found is False:
                op, string = ham_model._system[idx]
                if expected_string == string:
                    found = True
                    self.assertTrue(array_equal(expected_op, op))
                idx += 1
            self.assertTrue(found)

    def test_duffing_system_model2(self):
        """Second test of duffing_system_model, 3 qubits, 3 dimensional"""

        # do similar tests for different model
        dim_oscillators = 3
        oscillator_freqs = [5.0, 5.1, 5.2]
        anharm_freqs = [-0.33, -0.33, -0.32]
        drive_strengths = [1.1, 1.2, 1.3]
        coupling_dict = {(1,2): 0.03, (0,1): 0.02}
        dt = 1.3

        system_model = model_gen.duffing_system_model(dim_oscillators,
                                                      oscillator_freqs,
                                                      anharm_freqs,
                                                      drive_strengths,
                                                      coupling_dict,
                                                      dt)
        cr_idx_dict = {label: idx for idx, label in enumerate(system_model.control_channel_labels)}

        # check basic parameters
        self.assertEqual(system_model.subsystem_list, [0, 1, 2])
        self.assertEqual(system_model.dt, 1.3)

        # check that cr_idx_dict is correct
        self.assertEqual(cr_idx_dict, {(0,1): 0, (1,0): 1, (1,2): 2, (2,1): 3})
        self.assertEqual(system_model.control_channel_index((1,2)), 2)

        # check u_channel_lo is correct
        self.assertEqual(system_model.u_channel_lo,
                         [[UchannelLO(1, 1.0+0.0j)],
                          [UchannelLO(0, 1.0+0.0j)],
                          [UchannelLO(2, 1.0+0.0j)],
                          [UchannelLO(1, 1.0+0.0j)]])

        # check consistency of system_model.u_channel_lo with cr_idx_dict
        # this should in principle be redundant with the above two checks
        for q_pair, idx in cr_idx_dict.items():
            self.assertEqual(system_model.u_channel_lo[idx],
                             [UchannelLO(q_pair[1], 1.0+0.0j)])

        # check correct hamiltonian
        ham_model = system_model.hamiltonian
        expected_vars = {'v0': 5.0, 'v1': 5.1, 'v2': 5.2,
                         'alpha0': -0.33, 'alpha1': -0.33, 'alpha2': -0.32,
                         'r0': 1.1, 'r1': 1.2, 'r2': 1.3,
                         'j01': 0.02, 'j12': 0.03}
        self.assertEqual(ham_model._variables, expected_vars)
        self.assertEqual(ham_model._subsystem_dims, {0: 3, 1: 3, 2: 3})
        self._compare_str_lists(list(ham_model._channels), ['D0', 'D1', 'D3', 'U0', 'U1', 'U2', 'U3'])

        # check that Hamiltonian terms have been imported correctly
        # constructing the expected_terms requires some knowledge of how the strings get generated
        # and then parsed
        O0 = self._operator_array_from_str(3, ['I', 'I', 'O'])
        O1 = self._operator_array_from_str(3, ['I', 'O', 'I'])
        O2 = self._operator_array_from_str(3, ['O', 'I', 'I'])
        OO0 = O0@O0
        OO1 = O1@O1
        OO2 = O2@O2
        X0 = self._operator_array_from_str(3, ['I', 'I', 'A']) + self._operator_array_from_str(3, ['I', 'I', 'C'])
        X1 = self._operator_array_from_str(3, ['I', 'A', 'I']) + self._operator_array_from_str(3, ['I', 'C', 'I'])
        X2 = self._operator_array_from_str(3, ['A', 'I', 'I']) + self._operator_array_from_str(3, ['C', 'I', 'I'])
        exchange01 = self._operator_array_from_str(3, ['I', 'Sm', 'Sp']) + self._operator_array_from_str(3, ['I', 'Sp', 'Sm'])
        exchange12 = self._operator_array_from_str(3, ['Sm', 'Sp', 'I']) + self._operator_array_from_str(3, ['Sp', 'Sm', 'I'])
        expected_terms = [('np.pi*(2*v0-alpha0)', O0),
                          ('np.pi*(2*v1-alpha1)', O1),
                          ('np.pi*(2*v2-alpha2)', O2),
                          ('np.pi*alpha0', OO0),
                          ('np.pi*alpha1', OO1),
                          ('np.pi*alpha2', OO2),
                          ('2*np.pi*r0*D0', X0),
                          ('2*np.pi*r1*D1', X1),
                          ('2*np.pi*r2*D2', X2),
                          ('2*np.pi*j01', exchange01),
                          ('2*np.pi*j12', exchange12),
                          ('2*np.pi*r0*U0', X0),
                          ('2*np.pi*r1*U1', X1),
                          ('2*np.pi*r1*U2', X1),
                          ('2*np.pi*r2*U3', X2)]

        # first check the number of terms is correct, then loop through
        # each expected term and verify that it is present and consistent
        self.assertEqual(len(ham_model._system), len(expected_terms))
        for expected_string, expected_op in expected_terms:
            idx = 0
            found = False
            while idx < len(ham_model._system) and found is False:
                op, string = ham_model._system[idx]
                if expected_string == string:
                    found = True
                    self.assertTrue(array_equal(expected_op, op))
                idx += 1
            self.assertTrue(found)

    def test_duffing_system_model3(self):
        """Third test of duffing_system_model, 4 qubits, 2 dimensional"""

        # do similar tests for different model
        dim_oscillators = 2
        oscillator_freqs = [5.0, 5.1, 5.2, 5.3]
        anharm_freqs = [-0.33, -0.33, -0.32, -0.31]
        drive_strengths = [1.1, 1.2, 1.3, 1.4]
        coupling_dict = {(0,2): 0.03, (1,0): 0.02, (0,3): 0.14, (3,1): 0.18, (1,2) : 0.33}
        dt = 1.3

        system_model = model_gen.duffing_system_model(dim_oscillators,
                                                      oscillator_freqs,
                                                      anharm_freqs,
                                                      drive_strengths,
                                                      coupling_dict,
                                                      dt)
        cr_idx_dict = {label: idx for idx, label in enumerate(system_model.control_channel_labels)}

        # check basic parameters
        self.assertEqual(system_model.subsystem_list, [0, 1, 2, 3])
        self.assertEqual(system_model.dt, 1.3)

        # check that cr_idx_dict is correct
        self.assertEqual(cr_idx_dict, {(0,1): 0, (1,0): 1,
                                       (0,2): 2, (2,0): 3,
                                       (0,3): 4, (3,0): 5,
                                       (1,2): 6, (2,1): 7,
                                       (1,3): 8, (3,1): 9})
        self.assertEqual(system_model.control_channel_index((1,2)), 6)

        # check u_channel_lo is correct
        self.assertEqual(system_model.u_channel_lo,
                         [[UchannelLO(1, 1.0+0.0j)], [UchannelLO(0, 1.0+0.0j)],
                          [UchannelLO(2, 1.0+0.0j)], [UchannelLO(0, 1.0+0.0j)],
                          [UchannelLO(3, 1.0+0.0j)], [UchannelLO(0, 1.0+0.0j)],
                          [UchannelLO(2, 1.0+0.0j)], [UchannelLO(1, 1.0+0.0j)],
                          [UchannelLO(3, 1.0+0.0j)], [UchannelLO(1, 1.0+0.0j)]])

        # check consistency of system_model.u_channel_lo with cr_idx_dict
        # this should in principle be redundant with the above two checks
        for q_pair, idx in cr_idx_dict.items():
            self.assertEqual(system_model.u_channel_lo[idx],
                             [UchannelLO(q_pair[1], 1.0+0.0j)])

        # check correct hamiltonian
        ham_model = system_model.hamiltonian
        expected_vars = {'v0': 5.0, 'v1': 5.1, 'v2': 5.2, 'v3': 5.3,
                         'alpha0': -0.33, 'alpha1': -0.33, 'alpha2': -0.32, 'alpha3': -0.31,
                         'r0': 1.1, 'r1': 1.2, 'r2': 1.3, 'r3': 1.4,
                         'j01': 0.02, 'j02': 0.03, 'j03': 0.14, 'j12': 0.33, 'j13': 0.18}
        self.assertEqual(ham_model._variables, expected_vars)
        self.assertEqual(ham_model._subsystem_dims, {0: 2, 1: 2, 2: 2, 3: 2})
        self._compare_str_lists(list(ham_model._channels), ['D0', 'D1', 'D3', 'D4',
                                                            'U0', 'U1', 'U2', 'U3', 'U4',
                                                            'U5', 'U6', 'U7', 'U8', 'U9'])

        # check that Hamiltonian terms have been imported correctly
        # constructing the expected_terms requires some knowledge of how the strings get generated
        # and then parsed
        O0 = self._operator_array_from_str(2, ['I', 'I', 'I', 'O'])
        O1 = self._operator_array_from_str(2, ['I', 'I', 'O', 'I'])
        O2 = self._operator_array_from_str(2, ['I', 'O', 'I', 'I'])
        O3 = self._operator_array_from_str(2, ['O', 'I', 'I', 'I'])
        OO0 = O0@O0
        OO1 = O1@O1
        OO2 = O2@O2
        OO3 = O3@O3
        X0 = self._operator_array_from_str(2, ['I','I', 'I', 'A']) + self._operator_array_from_str(2, ['I', 'I', 'I', 'C'])
        X1 = self._operator_array_from_str(2, ['I', 'I', 'A', 'I']) + self._operator_array_from_str(2, ['I', 'I', 'C', 'I'])
        X2 = self._operator_array_from_str(2, ['I', 'A', 'I', 'I']) + self._operator_array_from_str(2, ['I', 'C', 'I', 'I'])
        X3 = self._operator_array_from_str(2, ['A', 'I', 'I', 'I']) + self._operator_array_from_str(2, ['C', 'I', 'I', 'I'])
        exchange01 = self._operator_array_from_str(2, ['I', 'I', 'Sm', 'Sp']) + self._operator_array_from_str(2, ['I', 'I', 'Sp', 'Sm'])
        exchange02 = self._operator_array_from_str(2, ['I', 'Sm', 'I', 'Sp']) + self._operator_array_from_str(2, ['I', 'Sp', 'I', 'Sm'])
        exchange03 = self._operator_array_from_str(2, ['Sm', 'I', 'I', 'Sp']) + self._operator_array_from_str(2, ['Sp', 'I', 'I', 'Sm'])
        exchange12 = self._operator_array_from_str(2, ['I', 'Sm', 'Sp', 'I']) + self._operator_array_from_str(2, ['I', 'Sp', 'Sm', 'I'])
        exchange13 = self._operator_array_from_str(2, ['Sm', 'I', 'Sp', 'I']) + self._operator_array_from_str(2, ['Sp', 'I', 'Sm', 'I'])
        expected_terms = [('np.pi*(2*v0-alpha0)', O0),
                          ('np.pi*(2*v1-alpha1)', O1),
                          ('np.pi*(2*v2-alpha2)', O2),
                          ('np.pi*(2*v3-alpha3)', O3),
                          ('np.pi*alpha0', OO0),
                          ('np.pi*alpha1', OO1),
                          ('np.pi*alpha2', OO2),
                          ('np.pi*alpha3', OO3),
                          ('2*np.pi*r0*D0', X0),
                          ('2*np.pi*r1*D1', X1),
                          ('2*np.pi*r2*D2', X2),
                          ('2*np.pi*r3*D3', X3),
                          ('2*np.pi*j01', exchange01),
                          ('2*np.pi*j02', exchange02),
                          ('2*np.pi*j03', exchange03),
                          ('2*np.pi*j12', exchange12),
                          ('2*np.pi*j13', exchange13),
                          ('2*np.pi*r0*U0', X0),
                          ('2*np.pi*r1*U1', X1),
                          ('2*np.pi*r0*U2', X0),
                          ('2*np.pi*r2*U3', X2),
                          ('2*np.pi*r0*U4', X0),
                          ('2*np.pi*r3*U5', X3),
                          ('2*np.pi*r1*U6', X1),
                          ('2*np.pi*r2*U7', X2),
                          ('2*np.pi*r1*U8', X1),
                          ('2*np.pi*r3*U9', X3)]

        # first check the number of terms is correct, then loop through
        # each expected term and verify that it is present and consistent
        self.assertEqual(len(ham_model._system), len(expected_terms))
        for expected_string, expected_op in expected_terms:
            idx = 0
            found = False
            while idx < len(ham_model._system) and found is False:
                op, string = ham_model._system[idx]
                if expected_string == string:
                    found = True
                    self.assertTrue(array_equal(expected_op, op))
                idx += 1
            self.assertTrue(found)

    def test_duffing_hamiltonian_dict(self):
        """Test _duffing_hamiltonian_dict"""

        oscillators = [0, 1]
        oscillator_dims = [2, 2]
        oscillator_freqs = [5.0, 5.1]
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
        output = model_gen._duffing_hamiltonian_dict(oscillators,
                                                      oscillator_dims,
                                                      oscillator_freqs,
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

        # test 3 oscillators with mixed up inputs
        oscillators = [0, 1, 2]
        oscillator_dims = [2, 2, 3]
        oscillator_freqs = [5.0, 5.1, 4.9]
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
        output = model_gen._duffing_hamiltonian_dict(oscillators,
                                                      oscillator_dims,
                                                      oscillator_freqs,
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


    def test_calculate_channel_frequencies(self):
        """test calculate_channel_frequencies of resulting PulseSystemModel objects"""

        dim_oscillators = 2
        oscillator_freqs = [5.0, 5.1]
        anharm_freqs = [-0.33, -0.33]
        drive_strengths = [1.1, 1.2]
        coupling_dict = {(0,1): 0.0}
        dt = 1.3

        system_model = model_gen.duffing_system_model(dim_oscillators,
                                                      oscillator_freqs,
                                                      anharm_freqs,
                                                      drive_strengths,
                                                      coupling_dict,
                                                      dt)

        channel_freqs = system_model.calculate_channel_frequencies([5.0, 5.1])
        expected = {'D0' : 5.0, 'D1' : 5.1, 'U0' : 5.1, 'U1': 5.0}
        self.assertEqual(dict(channel_freqs), expected)



    def test_cr_lo_list(self):
        """Test _cr_lo_list"""

        cr_dict = {(0,1): 0, (1,0) : 1, (3,4) : 2}
        expected = [[UchannelLO(1, 1.0+0.0j)],
                    [UchannelLO(0, 1.0+0.0j)],
                    [UchannelLO(4, 1.0+0.0j)]]
        self.assertEqual(model_gen._cr_lo_list(cr_dict), expected)

        cr_dict = {(0,1): 0, (3,4) : 2, (1,0) : 1}
        expected = [[UchannelLO(1, 1.0+0.0j)],
                    [UchannelLO(0, 1.0+0.0j)],
                    [UchannelLO(4, 1.0+0.0j)]]
        self.assertEqual(model_gen._cr_lo_list(cr_dict), expected)

    def test_single_term_generators(self):
        """Test various functions for individual terms:
        _single_duffing_drift_terms, _drive_terms, _exchange_coupling_terms, _cr_terms
        """

        # single duffing terms
        self.assertEqual(model_gen._single_duffing_drift_terms(freq_symbols='v',
                                                                anharm_symbols='a',
                                                                system_list=0),
                         ['np.pi*(2*v-a)*O0', 'np.pi*a*O0*O0'])
        self.assertEqual(model_gen._single_duffing_drift_terms(freq_symbols=['v0','v1'],
                                                                anharm_symbols=['a0','a1'],
                                                                system_list=[2, 3]),
                         ['np.pi*(2*v0-a0)*O2',
                          'np.pi*(2*v1-a1)*O3',
                          'np.pi*a0*O2*O2',
                          'np.pi*a1*O3*O3'])

        # drive terms
        self.assertEqual(model_gen._drive_terms(drive_symbols='r', system_list=0),
                         ['2*np.pi*r*X0||D0'])
        self.assertEqual(model_gen._drive_terms(drive_symbols=['r0', 'r1'], system_list=[1, 2]),
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

    def test_CouplingGraph(self):
        """Test CouplingGraph class."""

        coupling_graph = model_gen.CouplingGraph([(0,1), (1,0), (3,2), (1,2)])

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

    def _operator_array_from_str(self, dim, op_str_list):

        op = array([[1.]])
        for c in op_str_list:
            op = kron(op, get_oper(c, dim))

        return op
