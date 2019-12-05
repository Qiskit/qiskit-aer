# -*- coding: utf-8 -*-

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
An algorithm for taking the ExpectationValue of an operator with respect to a circuit in Aer.

"""

import logging

from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit import assemble

from ..extensions import SnapshotExpectationValue
from ..backends.qasm_simulator import QasmSimulator

logger = logging.getLogger(__name__)

class ExpectationValue():

    def __init__(self, operator, circuit, param_dict):
        """Constructor.

        Args:
            operator (WeightedPauliOperator): Qubit operator in the WeightedPauliBasis
            circuit (QuantumCircuit): Quantum circuit, optionally parameterized.
            param_dict (dict): Pairs of (Parameter, list(float)) parameterizations of the circuits
        """
        self._operator = operator
        self._circuit = circuit
        self._param_dict = param_dict

    def run(self, operator=None, circuit=None, param_dict=None):
        operator = operator or self._operator
        circuit = circuit or self._circuit
        param_dict = param_dict or self._param_dict

        if not operator or not circuit or not isinstance(circuit, QuantumCircuit):
            raise ValueError('operator and circuit must be set in the constructor or run() parameters.')
        if (not param_dict and len(circuit.parameters)) or not len(param_dict) == len(circuit.parameters):
            raise ValueError('If circuit is parameterized, param dict must have the same number of elements as '
                             'circuit has parameters. Found {} in param_dict and {} in circuit.parameters.'.format(
                len(param_dict), len(circuit.parameters)))

        snapshot = SnapshotExpectationValue('expval', operator.paulis, variance=True)
        # Add expectation value snapshot instruction
        circuit.append(snapshot)

        qasm_simulator = QasmSimulator()
        transpiled_circuit = transpile(circuits=circuit, backend=qasm_simulator)

        param_indices = []
        num_parameterizations_tmp = None
        for parameter in param_dict.keys():
            for (instr, param_index) in transpiled_circuit._parameter_table[parameter]:

                # Check that param list for each param is the same length as the previous one.
                if num_parameterizations_tmp and not num_parameterizations_tmp == len(param_dict[parameter]):
                    raise AssertionError('Parameterizations for each parameter must contain the same number of '
                                         'elements. Found {} and {} at instructions {} and {}, respectively.'.format(
                        num_parameterizations_tmp,
                        len(param_dict[parameter]),
                        transpiled_circuit._data.index(instr),
                        transpiled_circuit._data.index(instr) - 1))

                num_parameterizations_tmp = len(param_dict[parameter])
                instr_index = transpiled_circuit._data.index(instr)
                param_indices += [[(instr_index, param_index), param_dict[parameter]]]

        qobj = assemble(transpiled_circuit, {'parameterizations': param_indices})
        result = qasm_simulator.run(qobj).result()
        snapshot_data = result.data(circuit)['snapshots']
        expval = snapshot_data['expectation_value']['expval'][0]['value']
        avg = expval[0] + 1j * expval[1]
        return avg