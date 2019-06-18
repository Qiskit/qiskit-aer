# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Noise model inserter module
The goal of this module is to add QuantumError gates (Kraus gates) to a circuit
based on a given noise model. The resulting circuit cannot be ran on a quantum computer
but can be handled correctly by simulators
"""

def add_errors(circuit, noise_model):
    """
        This function gets a circuit and a noise model and returns a new circuit
        with the errors from the noise model inserted as Kraus gates in the new circuit
        Args:
            circuit (QuantumCircuit): The circuit to add the errors to
            noise_model (NoiseModel):  The noise model containing the errors to add
        Returns:
            QuantumCircuit: The new circuit with the added Kraus gates
        """
    error_dict = dict([(name, qubit_dict) for (name, qubit_dict) in noise_model._default_quantum_errors.items()])
    result_circuit = circuit.copy(name=circuit.name + '_with_errors')
    result_circuit.data = []
    for inst, qargs, cargs in circuit.data:
        result_circuit.data.append((inst, qargs, cargs))
        if inst.name in error_dict.keys():
            error = error_dict[inst.name]
            result_circuit.append(error.to_instruction(), qargs)
    return result_circuit