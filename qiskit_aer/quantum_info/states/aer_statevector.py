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
import copy
import numpy as np

from qiskit.quantum_info.states import Statevector
from qiskit.circuit import QuantumCircuit, Instruction
from .aer_state import AerState


class AerStatevector(Statevector):
    """AerStatevector class"""

    def __init__(self, data, **configs):
        """Initialize a statevector object.

        Args:
            data (np.array or list or Statevector or QuantumCircuit or qiskit.circuit.Instruction):
                Data from which the statevector can be constructed. This can be either a complex
                vector, another statevector or a ``QuantumCircuit`` or ``Instruction``
                (``Operator`` is not supportted in the current implementation).  If the data is
                a circuit or instruction, the statevector is constructed by assuming that all
                qubits are initialized to the zero state.
            configs (kwargs): configurations of ```AerState`. All the keys are handeld as string

        Raises:
            QiskitError: if input data is not valid.

        """
        if '_aer_state' in configs:
            self._aer_state = configs.pop('_aer_state')
        if isinstance(data, (QuantumCircuit, Instruction)):
            data, aer_state = AerStatevector._from_instruction(data, None, **configs)
            self._aer_state = aer_state
        elif isinstance(data, np.ndarray):
            data, aer_state = AerStatevector._from_ndarray(data, **configs)
            self._aer_state = aer_state
        else:
            self._aer_state = None
        super().__init__(data)
        self._result = None
        self._configs = configs

    def _aer(self):
        return self._aer_state is not None

    def _assert_aer_mode(self):
        if not self._aer():
            raise AerError('AerState was not used.')

    def _last_result(self):
        self._assert_aer_mode()
        if self._result is None:
            self._result = self._aer_state.last_result()
        return self._result

    def _metadata(self):
        self._assert_aer_mode()
        if self._last_result() is None:
            raise AerError('AerState was not used and metdata does not exist.')
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

    def __deepcopy__(self, _memo=None):
        if self._aer():
            data, aer_state = AerStatevector._from_instruction(
                QuantumCircuit(self._aer_state.num_qubits),
                self._data, **self._configs)
            return AerStatevector(data, _aer_state=aer_state, **self._configs)
        else:
            data = copy.deepcopy(self._data)
            return AerStatevector(data, **self._configs)

    def evolve(self, other, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            other (Operator): The operator to evolve by.
            qargs (list): a list of Statevector subsystem positions to apply
                           the operator on.

        Returns:
            Statevector: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified Statevector subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if isinstance(other, (QuantumCircuit, Instruction)):
            data, aer_state = AerStatevector._from_instruction(other, self._data, **self._configs)
            return AerStatevector(data, _aer_state=aer_state, **self._configs)

        ret = copy.copy(self)

        # Evolution by an Operator
        if not isinstance(other, Operator):
            other = Operator(other)

        # check dimension
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        return Statevector._evolve_operator(ret, other, qargs=qargs)

    @classmethod
    def _from_ndarray(cls, init_data, **configs):
        aer_state = AerState()

        for config_key in configs:
            aer_state.configure(config_key, configs[config_key])

        num_qubits = int(np.log2(len(init_data)))

        aer_state.allocate_qubits(num_qubits)
        aer_state.initialize(data=init_data)

        return aer_state.move_to_ndarray(), aer_state

    @classmethod
    def _from_instruction(cls, inst, init_data, **configs):
        aer_state = AerState()

        for config_key in configs:
            aer_state.configure(config_key, configs[config_key])

        if isinstance(inst, QuantumCircuit):
            circuit = inst
            aer_state.allocate_qubits(circuit.num_qubits)
            num_qubits = circuit.num_qubits
        else:
            aer_state.allocate_qubits(inst.num_qubits)
            num_qubits = inst.num_qubits

        if init_data is not None:
            aer_state.initialize(data=init_data)
        else:
            aer_state.initialize()

        AerStatevector._evolve_circuit(aer_state, circuit, range(num_qubits))

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
