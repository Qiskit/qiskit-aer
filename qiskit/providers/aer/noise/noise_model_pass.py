# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Transpiler pass to remove quanutm errors from circuit and add to noise model.
"""
import copy
import uuid
from qiskit.circuit import Gate
from qiskit.transpiler import TransformationPass
from qiskit.providers.aer.noise.noise_model import NoiseModel


class QuantumErrorLocation(Gate):
    """Instruction for representing a multi-qubit error location in Aer"""

    _directive = True

    def __init__(self, num_qubits, label=None):
        super().__init__("qerror_loc", num_qubits, [], label=label)


class NoiseModelPass(TransformationPass):
    """Remove quantum errors for circuit and add to a noise model"""

    def __init__(self, noise_model=None):
        """Initialize a build noise model pass instance.

        Args:
            noise_model (NoiseModel): Optional, noise model to add quantum errors to.
        """
        if noise_model is None:
            self._noise_model = NoiseModel()
        else:
            self._noise_model = copy.deepcopy(noise_model)
        self._updated = False

    @property
    def noise_model(self):
        """Return the noise model build by this pass"""
        return self._noise_model

    @property
    def updated(self):
        """Return True if the noise model as updated"""
        return self._updated

    def run(self, dag):
        """Build noise model from an input DAGCircuit.

        Args:
            dag (DAGCircuit): input dag

        Returns:
            DAGCircuit: translated circuit.
        """
        for node in dag.op_nodes():
            if node.name == "qerror":
                error = node.op._quantum_error
                qubits = list(dag.qubits.index(q) for q in node.qargs)
                label = str(uuid.uuid4())
                error_loc = QuantumErrorLocation(error.num_qubits, label=label)
                error_loc.condition = node.op.condition
                self._noise_model.add_quantum_error(error, error_loc, qubits)
                dag.substitute_node(node, error_loc)
                self._updated = True
        return dag
