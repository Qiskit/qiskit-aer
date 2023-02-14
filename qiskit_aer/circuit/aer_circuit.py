"""Directly simulatable circuit in Aer."""
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017-2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import List, Optional
from copy import copy

from qiskit.circuit import QuantumCircuit, Clbit, ParameterExpression
from qiskit.tools.parallel import parallel_map
from qiskit.providers.options import Options
from qiskit.qobj import QobjExperimentHeader
from qiskit_aer.aererror import AerError
# pylint: disable=import-error, no-name-in-module
from qiskit_aer.backends.controller_wrappers import AerCircuit_


class AerCircuit:
    """"Aer Internal Circuit class"""

    def __init__(self,
                 circuit: QuantumCircuit,
                 shots: int,
                 seed: Optional[int] = None):

        self._circuit = circuit
        self._shots = shots
        self._seed = seed

        self._aer_circ = None

        self._num_qubits = 0
        self._num_memory = 0
        self._max_conditional_idx = 0

        qreg_sizes = []
        creg_sizes = []
        qubit_labels = []
        clbit_labels = []
        self._global_phase = float(circuit.global_phase)

        for qreg in circuit.qregs:
            qreg_sizes.append([qreg.name, qreg.size])
            for j in range(qreg.size):
                qubit_labels.append([qreg.name, j])
            self._num_qubits += qreg.size
        for creg in circuit.cregs:
            creg_sizes.append([creg.name, creg.size])
            for j in range(creg.size):
                clbit_labels.append([creg.name, j])
            self._num_memory += creg.size

        self._is_conditional_experiment = any(
            getattr(inst.operation, "condition", None) for inst in circuit.data
        )

        self._header = QobjExperimentHeader(
            qubit_labels=qubit_labels,
            n_qubits=self._num_qubits,
            qreg_sizes=qreg_sizes,
            clbit_labels=clbit_labels,
            memory_slots=self._num_memory,
            creg_sizes=creg_sizes,
            name=circuit.name,
            global_phase=self._global_phase,
        )

        if hasattr(circuit, "metadata") and circuit.metadata:
            self._header.metadata = copy(circuit.metadata)

        self._qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
        self._clbit_indices = {clbit: idx for idx, clbit in enumerate(circuit.clbits)}

    def _append_op(self, aer_circ, inst, conditional_reg_idx):
        operation = inst.operation
        qubits = [self._qubit_indices[qubit] for qubit in inst.qubits]
        clbits = [self._clbit_indices[clbit] for clbit in inst.clbits]
        name = operation.name
        label = operation.label
        params = operation.params if hasattr(operation, "params") else None
        copied = False

        for i in range(len(params)):
            if (isinstance(params[i], ParameterExpression)
                    and len(params[i].parameters) > 0):
                if not copied:
                    params = copy(params)
                    copied = True
                params[i] = 0.0

        if name in ('ccx', 'ccz', 'cp', 'cswap', 'csx', 'cx', 'cy', 'cz', 'delay', 'ecr',
                    'h', 'id', 'mcp', 'mcphase', 'mcr', 'mcrx', 'mcry', 'mcrz', 'mcswap',
                    'mcsx', 'mcu', 'mcu1', 'mcu2', 'mcu3', 'mcx', 'mcx_gray', 'mcy', 'mcz',
                    'p', 'r', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzx', 'rzz', 's', 'sdg', 'swap',
                    'sx', 'sxdg', 't', 'tdg', 'u', 'x', 'y', 'z', 'u1', 'u2', 'u3',
                    'cu', 'cu1', 'cu2', 'cu3'):
            aer_circ.gate(name, qubits, params, [],
                                conditional_reg_idx, label if label else name)
        elif name == 'pauli':
            aer_circ.gate(name, qubits, [], params,
                                conditional_reg_idx, label if label else name)
        elif name == 'measure':
            if self._is_conditional_experiment:
                aer_circ.measure(qubits, clbits, clbits)
            else:
                aer_circ.measure(qubits, clbits, [])
        elif name == 'reset':
            aer_circ.reset(qubits)
        elif name == 'diagonal':
            aer_circ.diagonal(qubits, params, label if label else 'diagonal')
        elif name == 'unitary':
            aer_circ.unitary(qubits, params[0], conditional_reg_idx,
                                   label if label else 'unitary')
        elif name == 'initialize':
            aer_circ.initialize(qubits, params)
        elif name == 'roerror':
            aer_circ.roerror(qubits, params)
        elif name == 'multiplexer':
            aer_circ.multiplexer(qubits, params, conditional_reg_idx, label if label else name)
        elif name == 'kraus':
            aer_circ.kraus(qubits, params, conditional_reg_idx)
        elif name in ('save_statevector', 'save_statevector_dict', 'save_clifford',
                      'save_probabilities', 'save_probabilities_dict', 'save_matrix_product_state',
                      'save_unitary', 'save_superop', 'save_density_matrix', 'save_state',
                      'save_stabilizer'):
            aer_circ.save_state(qubits, name, operation._subtype, label if label else name)
        elif name in ('save_amplitudes', 'save_amplitudes_sq'):
            aer_circ.save_amplitudes(qubits, name, params, operation._subtype,
                                     label if label else name)
        elif name in ('save_expval', 'save_expval_var'):
            paulis = []
            coeff_reals = []
            coeff_imags = []
            for pauli, coeff in operation.params:
                paulis.append(pauli)
                coeff_reals.append(coeff[0])
                coeff_imags.append(coeff[1])
            aer_circ.save_expval(qubits, name, paulis, coeff_reals, coeff_imags, operation._subtype,
                                 label if label else name)
        elif name == 'set_statevector':
            aer_circ.set_statevector(qubits, params)
        elif name == 'set_unitary':
            aer_circ.set_unitary(qubits, params[0])
        elif name == 'set_density_matrix':
            aer_circ.set_density_matrix(qubits, params[0])
        elif name == 'set_stabilizer':
            aer_circ.set_clifford(qubits, params)
        elif name == 'set_superop':
            aer_circ.set_superop(qubits, params[0])
        elif name == 'set_matrix_product_state':
            aer_circ.set_matrix_product_state(qubits, params)
        elif name == 'superop':
            aer_circ.superop(qubits, params[0], conditional_reg_idx)
        elif name == 'barrier':
            pass
        elif name == 'jump':
            aer_circ.jump(qubits, params, conditional_reg_idx)
        elif name == 'mark':
            aer_circ.mark(qubits, params)
        elif name == 'qerror_loc':
            aer_circ.set_qerror_loc(qubits, label if label else name, conditional_reg_idx)
        elif name in ('for_loop', 'while_loop', 'if_else'):
            raise AerError('control-flow instructions must be converted ' +
                           f'to jump and mark instructions: {name}')

        else:
            raise AerError(f'unknown instruction: {name}')

    @property
    def native_circuit(self):
        """a native circuit instance"""
        if self._aer_circ:
            return self._aer_circ

        self._aer_circ = AerCircuit_()
        self._aer_circ.set_header(self._header)
        self._aer_circ.num_qubits = self._num_qubits
        self._aer_circ.num_memory = self._num_memory
        self._aer_circ.shots = self._shots
        if self._seed:
            self._aer_circ.seed = self._seed
        else:
            self._seed = self._aer_circ.seed
        self._aer_circ.global_phase_angle = self._global_phase

        for inst in self._circuit.data:
            # To convert to a qobj-style conditional, insert a bfunc prior
            # to the conditional instruction to map the creg ?= val condition
            # onto a gating register bit.
            conditional_reg_idx = -1
            if hasattr(inst.operation, "condition") and inst.operation.condition:
                ctrl_reg, ctrl_val = inst.operation.condition
                mask = 0
                val = 0
                if isinstance(ctrl_reg, Clbit):
                    mask = 1 << self._clbit_indices[ctrl_reg]
                    val = (ctrl_val & 1) << self._clbit_indices[ctrl_reg]
                else:
                    for clbit, idx in self._clbit_indices.items():
                        if clbit in ctrl_reg:
                            mask |= 1 << idx
                            val |= ((ctrl_val >> list(ctrl_reg).index(clbit)) & 1) << idx
                conditional_reg_idx = self._num_memory + self._max_conditional_idx
                self._aer_circ.bfunc("0x%X" % mask, "0x%X" % val, "==", conditional_reg_idx)
                self._max_conditional_idx += 1

            self._append_op(self._aer_circ, inst, conditional_reg_idx)

        return self._aer_circ

    @property
    def num_memory(self):
        """a number of memory"""
        return self._num_memory

    @property
    def num_qubits(self):
        """a number of qubits"""
        return self._num_qubits


def generate_aer_circuit(
    circuit: QuantumCircuit,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> AerCircuit:
    """converts a Qiskit circuit into a circuit that Aer can directly run.

    Args:
        circuit: circuit to be converted
        shots: a number of shots
        seed: seed to generate random numbers

    Returns:
        circuit to be run on the Aer backends
    """

    return AerCircuit(circuit, shots, seed)


def generate_aer_circuits(
    circuits: List[QuantumCircuit]
) -> List[AerCircuit]:
    """converts a list of Qiskit circuits into circuits that Aer can directly run.

    Args:
        circuits: circuit(s) to be converted
        backend_options: backend options
        run_options: run options

    Returns:
        circuits to be run on the Aer backends

    Examples:

        .. code-block:: python

            from qiskit.circuit import QuantumCircuit
            from qiskit_aer_backends import generate_aer_circuits
            # Create a circuit to be simulated
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            # Generate AerCircuit from the input circuit
            aer_qc_list, config = generate_aer_circuits(circuits=[qc])
    """
    # generate aer circuits
    if len(circuits) == 1:
        return [generate_aer_circuit(circuits[0])]
    else:
        return parallel_map(generate_aer_circuit, circuits)

def generate_aer_config(
    circuits: List[QuantumCircuit],
    backend_options: Options,
    **run_options
) -> List[AerCircuit]:
    """generates a configuration to run simulation.

    Args:
        circuits: circuit(s) to be converted
        backend_options: backend options
        run_options: run options

    Returns:
        dict to run Aer
    """
    num_qubits = max(sum(qreg.size for qreg in circuit.qregs) for circuit in circuits)
    memory_slots = max(sum(creg.size for creg in circuit.cregs) for circuit in circuits)

    config = {
        'memory_slots': memory_slots,
        'n_qubits': num_qubits,
        **backend_options.__dict__,
        **run_options
    }
    return config
