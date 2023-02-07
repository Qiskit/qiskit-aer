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

from typing import List, Dict, Optional
from copy import copy

from qiskit.circuit import QuantumCircuit, Clbit
from qiskit.tools.parallel import parallel_map
from qiskit.providers.options import Options
from qiskit_aer.backends.controller_wrappers import AerCircuit_
from .aer_operation import AerOp, BinaryFuncOp, generate_aer_operation
from ..noise.noise_model import NoiseModel
from qiskit.qobj import QobjExperimentHeader

"""Directly simulatable circuit in Aer."""

class AerCircuit:
    """A class of an internal ciruict of Aer
    """
    def __init__(
        self,
        header: QobjExperimentHeader,
        num_qubits: int,
        num_memory: int,
        shots: int,
        seed: Optional[int] = None,
        global_phase: Optional[float] = None,
    ):
        self._header = header
        self._num_qubits = num_qubits
        self._num_memory = num_memory
        self._shots = shots
        if seed:
            self._set_seed = True
            self._seed = seed
        else:
            self._set_seed = False
        self._global_phase = global_phase
        self._is_conditional_experiment = False
        self._aer_ops = []

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def shots(self):
        return self._shots

    @shots.setter
    def shots(self, v):
        self._shots = v

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, v):
        self._set_seed = True
        self._seed = v

    def append(
        self,
        aer_op: AerOp
    ) -> None:
        self._aer_ops.append(aer_op)

    def assemble_native(
        self
    ) -> AerCircuit_:
        aer_circuit = AerCircuit_()
        aer_circuit.num_qubits = self._num_qubits
        aer_circuit.num_memory = self._num_memory
        aer_circuit.shots = self._shots
        if self._set_seed:
            aer_circuit.seed = self._seed
        if self._global_phase is None:
            aer_circuit.global_phase_angle = 0.0
        else:
            aer_circuit.global_phase_angle = self._global_phase
        aer_circuit.ops = [aer_op.assemble_native() for aer_op in self._aer_ops]
        aer_circuit.set_header(self._header)

        return aer_circuit

def generate_aer_circuit(
    circuit: QuantumCircuit,
    shots: int = 1024,
    seed: Optional[int] = None,
)-> AerCircuit:

    num_qubits = 0
    memory_slots = 0
    max_conditional_idx = 0

    qreg_sizes = []
    creg_sizes = []
    qubit_labels = []
    clbit_labels = []

    for qreg in circuit.qregs:
        qreg_sizes.append([qreg.name, qreg.size])
        for j in range(qreg.size):
            qubit_labels.append([qreg.name, j])
        num_qubits += qreg.size
    for creg in circuit.cregs:
        creg_sizes.append([creg.name, creg.size])
        for j in range(creg.size):
            clbit_labels.append([creg.name, j])
        memory_slots += creg.size

    is_conditional_experiment = any(
        getattr(inst.operation, "condition", None) for inst in circuit.data
    )

    header = QobjExperimentHeader(
        qubit_labels=qubit_labels,
        n_qubits=num_qubits,
        qreg_sizes=qreg_sizes,
        clbit_labels=clbit_labels,
        memory_slots=memory_slots,
        creg_sizes=creg_sizes,
        name=circuit.name,
        global_phase=float(circuit.global_phase),
    )

    if hasattr(circuit, "metadata"):
        header.metadata = copy(circuit.metadata)

    qubit_indices = {qubit: idx for idx, qubit in enumerate(circuit.qubits)}
    clbit_indices = {clbit: idx for idx, clbit in enumerate(circuit.clbits)}

    aer_circ = AerCircuit(
        header,
        num_qubits,
        memory_slots,
        shots,
        seed,
        circuit.global_phase,
        )
    
    for inst in circuit.data:
        aer_op = generate_aer_operation(inst,
                                           qubit_indices,
                                           clbit_indices,
                                           is_conditional_experiment)

        # To convert to a qobj-style conditional, insert a bfunc prior
        # to the conditional instruction to map the creg ?= val condition
        # onto a gating register bit.
        conditional_reg = -1
        if hasattr(inst.operation, "condition") and inst.operation.condition:
            ctrl_reg, ctrl_val = inst.operation.condition
            mask = 0
            val = 0
            if isinstance(ctrl_reg, Clbit):
                mask = 1 << clbit_indices[ctrl_reg]
                val = (ctrl_val & 1) << clbit_indices[ctrl_reg]
            else:
                for clbit in clbit_indices:
                    if clbit in ctrl_reg:
                        mask |= 1 << clbit_indices[clbit]
                        val |= ((ctrl_val >> list(ctrl_reg).index(clbit)) & 1) << clbit_indices[
                            clbit
                        ]
            conditional_reg_idx = memory_slots + max_conditional_idx
            aer_circ.append(BinaryFuncOp(mask, "==", val, conditional_reg_idx))
            max_conditional_idx += 1
            aer_op.set_conditional(conditional_reg_idx)

        aer_circ.append(aer_op)
    
    return aer_circ

def generate_aer_circuits(
    circuits: List[QuantumCircuit],
    backend_options: Options,
    **run_options    
) -> List[AerCircuit]:
    """generates a list of Qiskit circuits into circuits that Aer can directly run.

    Args:
        circuits: circuit(s) to be converted

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
    aer_circs = parallel_map(generate_aer_circuit, circuits)
    config = {
        'memory_slots': max([aer_circ._num_memory for aer_circ in aer_circs]),
        'n_qubits': max([aer_circ._num_qubits for aer_circ in aer_circs]),
        **backend_options.__dict__,
        **run_options
    }
    return aer_circs, config
