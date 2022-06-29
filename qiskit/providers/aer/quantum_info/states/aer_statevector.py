# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019, 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Statevector quantum state class.
"""
from qiskit.quantum_info.states import Statevector
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.providers.aer.quantum_info.states.aer_state import AerState


class AerStatevector(Statevector):
    """AerStatevector class"""

    def __init__(self, data, **configs):
        """Initialize a statevector object.

        Args:
            data (np.array or list or Statevector or Operator or QuantumCircuit or
                  qiskit.circuit.Instruction):
                Data from which the statevector can be constructed. This can be either a complex
                vector, another statevector, a ``Operator`` with only one column or a
                ``QuantumCircuit`` or ``Instruction``.  If the data is a circuit or instruction,
                the statevector is constructed by assuming that all qubits are initialized to the
                zero state.
            configs (kwargs): configurations of ```AerState`. All the keys are handeld as string

        Raises:
            QiskitError: if input data is not valid.

        """
        if isinstance(data, (QuantumCircuit, Instruction)):
            data, aer_state = AerStatevector.from_instruction(data, **configs)
        super().__init__(data)
        self._aer_state = aer_state
        self._result = None

    def _last_result(self):
        if self._result is None:
            self._result = self._aer_state.last_result()
        return self._result

    def _metadata(self):
        return self._last_result()['metadata']

    @property
    def method(self):
        return self._metadata()['method']

    @property
    def device(self):
        return self._metadata()['device']

    @property
    def fusion_config(self):
        return self._metadata()['fusion']

    @property
    def parallel_state_update(self):
        return self._metadata()['parallel_state_update']

    @classmethod
    def from_instruction(cls, inst, **configs):
        """Return the output statevector of an instruction.

        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of the
        same number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            inst (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit
            configs (kwargs): configurations of ```AerState`. All the keys are handeld as string

        Returns:
            Statevector: The final statevector.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
                         the statevector simulation.
        """
        aer_state = AerState()

        for config_key in configs:
            aer_state.configure(config_key, configs[config_key])

        if isinstance(inst, QuantumCircuit):
            circuit = inst
            aer_state.allocate_qubits(circuit.num_qubits)
            aer_state.initialize()
            AerStatevector._evolve_circuit(aer_state, circuit, range(circuit.num_qubits))
        else:
            aer_state.allocate_qubits(inst.num_qubits)
            AerStatevector._evolve_instruction(aer_state, inst, range(inst.num_qubits))

        return aer_state.move_to_ndarray(), aer_state

    @classmethod
    def _evolve_circuit(cls, aer_state, circuit, qubits):
        """Apply circuit into aer_state"""
        for inst, qargs, _ in circuit.data:
            AerStatevector._evolve_instruction(aer_state, inst,
                                               [qubits[circuit.find_bit(qarg).index]
                                                for qarg in qargs])

    @classmethod
    def _evolve_instruction(cls, aer_state, inst, qubits):
        """Apply instruction into aer_state"""
        params = inst.params
        if inst.name == 'u':
            aer_state.apply_mcu(qubits, params[0], params[1], params[2])
        elif inst.name in ['x', 'cx', 'ccx']:
            aer_state.apply_mcx(qubits)
        elif inst.name in ['y', 'cy']:
            aer_state.apply_mcy(qubits)
        elif inst.name in ['z', 'cz']:
            aer_state.apply_mcz(qubits)
        elif inst.name == 'unitary':
            aer_state.apply_unitary(qubits, inst.params[0])
        elif inst.name == 'diagonal':
            aer_state.apply_diagonal(qubits, inst.params[0])
        else:
            definition = inst.definition
            if definition is inst:
                raise AerError('cannot decompose ' + inst.name)
            AerStatevector._evolve_circuit(aer_state, definition, qubits)
