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

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.states import Statevector

from qiskit_aer import AerSimulator
from .aer_state import AerState
from ...backends.aerbackend import AerError


class AerStatevector(Statevector):
    """AerStatevector class

    This class inherits :class:`Statevector`, which stores probability amplitudes
    in its `ndarray`. class:`AerStatevector` generates this `ndarray` by using the
    same runtime with :class:`AerSimulator`.
    """

    def __init__(self, data, dims=None, **configs):
        """
        Args:
            data (np.array or list or AerStatevector or QuantumCircuit or
                  qiskit.circuit.Instruction):
                Data from which the statevector can be constructed. This can be either a complex
                vector, another statevector or a ``QuantumCircuit`` or ``Instruction``
                (``Operator`` is not supportted in the current implementation).  If the data is
                a circuit or instruction, the statevector is constructed by assuming that all
                qubits are initialized to the zero state.
            dims (int or tuple or list): Optional. The subsystem dimension of
                                         the state (See additional information).
            configs (kwargs): configurations of :class:`AerSimulator`. `method` configuration must
                be `statevector` or `matrix_product_state`.

        Raises:
            AerError: if input data is not valid.

        Additional Information:
            The ``dims`` kwarg is used to ``Statevector`` constructor.

        """
        if '_aer_state' in configs:
            self._aer_state = configs.pop('_aer_state')
        else:
            if 'method' not in configs:
                configs['method'] = 'statevector'
            elif configs['method'] not in ('statevector', 'matrix_product_state'):
                method = configs['method']
                raise AerError(f'Method {method} is not supported')
            if isinstance(data, (QuantumCircuit, Instruction)):
                data, aer_state = AerStatevector._from_instruction(data, None, configs)
            elif isinstance(data, list):
                data, aer_state = AerStatevector._from_ndarray(np.array(data, dtype=complex),
                                                               configs)
            elif isinstance(data, np.ndarray):
                data, aer_state = AerStatevector._from_ndarray(data, configs)
            elif isinstance(data, AerStatevector):
                aer_state = data._aer_state
                if dims is None:
                    dims = data._op_shape._dims_l
                data = data._data.copy()
            else:
                raise AerError(f'Input data is not supported: type={data.__class__}, data={data}')

            self._aer_state = aer_state

        super().__init__(data, dims=dims)

        self._result = None
        self._configs = configs

    def _last_result(self):
        if self._result is None:
            self._result = self._aer_state.last_result()
        return self._result

    def metadata(self):
        """Return result metadata of an operation that executed lastly."""
        if self._last_result() is None:
            raise AerError('AerState was not used and metdata does not exist.')
        return self._last_result()['metadata']

    def __copy__(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, _memo=None):
        ret = AerStatevector(self._data.copy(), **self._configs)
        ret._op_shape = copy.deepcopy(self._op_shape)
        ret._rng_generator = copy.deepcopy(self._rng_generator)
        return ret

    def conjugate(self):
        return AerStatevector(np.conj(self._data), dims=self.dims())

    def sample_memory(self, shots, qargs=None):
        if qargs is None:
            qubits = np.arange(self._aer_state.num_qubits)
        else:
            qubits = np.array(qargs)
        self._aer_state.close()
        self._aer_state = AerState(**self._aer_state.configuration())
        self._aer_state.initialize(self._data, copy=False)
        samples = self._aer_state.sample_memory(qubits, shots)
        self._data = self._aer_state.move_to_ndarray()
        return samples

    @staticmethod
    def _from_ndarray(init_data, configs):
        aer_state = AerState()

        options = AerSimulator._default_options()
        for config_key, config_value in configs.items():
            if options.get(config_key):
                aer_state.configure(config_key, config_value)

        if len(init_data) == 0:
            raise AerError('initial data must be larger than 0')

        num_qubits = int(np.log2(len(init_data)))

        aer_state.allocate_qubits(num_qubits)
        aer_state.initialize(data=init_data)

        return aer_state.move_to_ndarray(), aer_state

    @classmethod
    def from_instruction(cls, instruction):
        return AerStatevector(instruction)

    @staticmethod
    def _from_instruction(inst, init_data, configs):
        aer_state = AerState()

        for config_key, config_value in configs.items():
            aer_state.configure(config_key, config_value)

        aer_state.allocate_qubits(inst.num_qubits)
        num_qubits = inst.num_qubits

        if init_data is not None:
            aer_state.initialize(data=init_data, copy=True)
        else:
            aer_state.initialize()

        if isinstance(inst, QuantumCircuit) and inst.global_phase != 0:
            aer_state.apply_global_phase(inst.global_phase)

        if isinstance(inst, QuantumCircuit):
            AerStatevector._aer_evolve_circuit(aer_state, inst, range(num_qubits))
        else:
            AerStatevector._aer_evolve_instruction(aer_state, inst, range(num_qubits))

        return aer_state.move_to_ndarray(), aer_state

    @staticmethod
    def _aer_evolve_circuit(aer_state, circuit, qubits):
        """Apply circuit into aer_state"""
        for instruction in circuit.data:
            if instruction.clbits:
                raise AerError(
                    f"Cannot apply instruction with classical bits: {instruction.operation.name}"
                )
            inst = instruction.operation
            qargs = instruction.qubits
            AerStatevector._aer_evolve_instruction(aer_state, inst,
                                                   [qubits[circuit.find_bit(qarg).index]
                                                    for qarg in qargs])

    @staticmethod
    def _aer_evolve_instruction(aer_state, inst, qubits):
        """Apply instruction into aer_state"""
        params = inst.params
        if inst.name in ['u', 'u3']:
            aer_state.apply_mcu(qubits[0:len(qubits) - 1], qubits[len(qubits) - 1],
                                params[0], params[1], params[2])
        elif inst.name in ['x', 'cx', 'ccx']:
            aer_state.apply_mcx(qubits[0:len(qubits) - 1], qubits[len(qubits) - 1])
        elif inst.name in ['y', 'cy']:
            aer_state.apply_mcy(qubits[0:len(qubits) - 1], qubits[len(qubits) - 1])
        elif inst.name in ['z', 'cz']:
            aer_state.apply_mcz(qubits[0:len(qubits) - 1], qubits[len(qubits) - 1])
        elif inst.name == 'unitary':
            aer_state.apply_unitary(qubits, inst.params[0])
        elif inst.name == 'diagonal':
            aer_state.apply_diagonal(qubits, inst.params[0])
        elif inst.name == 'reset':
            aer_state.apply_reset(qubits)
        else:
            definition = inst.definition
            if definition is inst:
                raise AerError('cannot decompose ' + inst.name)
            if not definition:
                raise AerError('cannot decompose ' + inst.name)
            AerStatevector._aer_evolve_circuit(aer_state, definition, qubits)

    @classmethod
    def from_label(cls, label):
        return AerStatevector(Statevector.from_label(label)._data)

    @staticmethod
    def from_int(i, dims):
        size = np.product(dims)
        state = np.zeros(size, dtype=complex)
        state[i] = 1.0
        return AerStatevector(state, dims=dims)
