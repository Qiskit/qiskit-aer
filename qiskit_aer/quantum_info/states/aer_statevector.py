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
from qiskit.quantum_info.operators.operator import Operator

from qiskit_aer import AerSimulator
from .aer_state import AerState
from ...backends.aerbackend import AerError


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
            configs (kwargs): configurations of ``AerState``. All the keys are handeld as string

        Raises:
            AerError: if input data is not valid.

        Additional Information:
            The ``dims`` kwarg is used to ``Statevector`` constructor.

        """
        if 'dims' in configs:
            dims = configs.pop('dims')
        else:
            dims = None

        if isinstance(data, (QuantumCircuit, Instruction)):
            data, aer_state = AerStatevector._from_instruction(data, None, **configs)
        elif isinstance(data, list):
            data, aer_state = AerStatevector._from_ndarray(np.array(data, dtype=complex), **configs)
        elif isinstance(data, np.ndarray):
            data, aer_state = AerStatevector._from_ndarray(data, **configs)
        elif isinstance(data, AerStatevector):
            aer_state = data._aer_state
            if dims is None:
                dims = data._op_shape._dims_l
            data = data._data
        else:
            raise AerError(f'Input data is not supported: type={data.__class__}, data={data}')

        super().__init__(data, dims=dims)
        self._aer_state = aer_state
        self._result = None
        self._configs = configs

    def _last_result(self):
        if self._result is None:
            self._result = self._aer_state.last_result()
        return self._result

    @property
    def metadata(self):
        """Return result metadata of an operation that executed lastly."""
        if self._last_result() is None:
            raise AerError('AerState was not used and metdata does not exist.')
        return self._last_result()['metadata']

    def __deepcopy__(self, _memo=None):
        data, aer_state = AerStatevector._from_instruction(
            QuantumCircuit(self._aer_state.num_qubits),
            self._data, **self._configs)
        return AerStatevector(data, _aer_state=aer_state, **self._configs)

    def __eq__(self, other):
        eq_quantumqtate = isinstance(other, Statevector) and self.dims() == other.dims()
        eq_statevector = np.allclose(self._data, other._data, rtol=self.rtol, atol=self.atol)
        return eq_quantumqtate and eq_statevector

    def evolve(self, other, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            other (Operator): The operator to evolve by.
            qargs (list): a list of Statevector subsystem positions to apply
                           the operator on.

        Returns:
            Statevector: the output quantum state.

        Raises:
            AerError: if the operator dimension does not match the
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
            raise AerError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        return Statevector._evolve_operator(ret, other, qargs=qargs)

    def _expectation_value_pauli(self, pauli, qargs=None):
        """Compute the expectation value of a Pauli with Aer implementation.

        Args:
            pauli (Pauli): a Pauli operator to evaluate expval of.
            qargs (None or list): subsystems to apply operator on.

        Returns:
            complex: the expectation value.
        """
        n_pauli = len(pauli)
        if qargs is None:
            qubits = np.arange(n_pauli)
        else:
            qubits = np.array(qargs)
        pauli_str = str(pauli)
        self._aer_state.close()
        self._aer_state = AerState(**self._aer_state.configuration())
        self._aer_state.initialize(self._data, copy=False)
        ret = self._aer_state.expectation_value_pauli(pauli_str, qubits)
        self._data = self._aer_state.move_to_ndarray()
        return ret

    def sample_memory(self, shots, qargs=None):
        """Sample a list of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            np.array: list of sampled counts if the order sampled.

        Additional Information:

            This function *samples* measurement outcomes using the measure
            :meth:`probabilities` for the current state and `qargs`. It does
            not actually implement the measurement so the current state is
            not modified.
        """
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

    @classmethod
    def _from_ndarray(cls, init_data, **configs):
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
    def _from_instruction(cls, inst, init_data, **configs):
        aer_state = AerState()

        for config_key, config_value in configs.items():
            aer_state.configure(config_key, config_value)

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

        if isinstance(inst, QuantumCircuit) and inst.global_phase != 0:
            aer_state.apply_global_phase(inst.global_phase)

        AerStatevector._aer_evolve_circuit(aer_state, circuit, range(num_qubits))

        return aer_state.move_to_ndarray(), aer_state

    @classmethod
    def _aer_evolve_circuit(cls, aer_state, circuit, qubits):
        """Apply circuit into aer_state"""
        for inst, qargs, _ in circuit.data:
            AerStatevector._aer_evolve_instruction(aer_state, inst,
                                                   [qubits[circuit.find_bit(qarg).index]
                                                    for qarg in qargs])

    @classmethod
    def _aer_evolve_instruction(cls, aer_state, inst, qubits):
        """Apply instruction into aer_state"""
        params = inst.params
        if inst.name == 'u':
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
        else:
            definition = inst.definition
            if definition is inst:
                raise AerError('cannot decompose ' + inst.name)
            AerStatevector._aer_evolve_circuit(aer_state, definition, qubits)

    @classmethod
    def from_label(cls, label):
        """Return a tensor product of Pauli X,Y,Z eigenstates.

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label
             - Statevector
           * - ``"0"``
             - :math:`[1, 0]`
           * - ``"1"``
             - :math:`[0, 1]`
           * - ``"+"``
             - :math:`[1 / \\sqrt{2},  1 / \\sqrt{2}]`
           * - ``"-"``
             - :math:`[1 / \\sqrt{2},  -1 / \\sqrt{2}]`
           * - ``"r"``
             - :math:`[1 / \\sqrt{2},  i / \\sqrt{2}]`
           * - ``"l"``
             - :math:`[1 / \\sqrt{2},  -i / \\sqrt{2}]`

        Args:
            label (string): a eigenstate string ket label (see table for
                            allowed values).

        Returns:
            AerStatevector: The N-qubit basis state density matrix.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.
        """
        sv = Statevector.from_label(label)
        return AerStatevector(sv._data)
