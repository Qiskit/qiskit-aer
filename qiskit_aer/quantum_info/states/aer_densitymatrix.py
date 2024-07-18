# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019, 2020, 2021, 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
DensityMatrix quantum state class.
"""
import copy
import numpy as np

from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import DensityMatrix
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix

from qiskit_aer import AerSimulator
from .aer_statevector import AerStatevector
from .aer_state import AerState
from ...backends.aerbackend import AerError
from ...backends.backend_utils import BASIS_GATES


class AerDensityMatrix(DensityMatrix):
    """AerDensityMatrix class
    This class inherits :class:`DensityMatrix`.
    """

    def __init__(self, data, dims=None, **configs):
        """
        Args:
            data (np.array or list or Statevector or AerStatevector or DensityMatrix or
                  AerDensityMatrix or QuantumCircuit or qiskit.circuit.Instruction):
                Data from which the densitymatrix can be constructed. This can be either a complex
                vector, another densitymatrix or statevector or a ``QuantumCircuit`` or
                ``Instruction`` (``Operator`` is not supported in the current implementation).
                If the data is a circuit or instruction, the densitymatrix is constructed by
                assuming that all qubits are initialized to the zero state.
            dims (int or tuple or list): Optional. The subsystem dimension of
                                         the state (See additional information).
            configs (kwargs): configurations of :class:`AerDensityMatrix`. `_aer_state` and `method`
                are valid.

        Raises:
            AerError: if input data is not valid.
        Additional Information:
            The ``dims`` kwarg is used to ``AerDensityMatrix`` constructor.
        """
        if "_aer_state" in configs:
            self._aer_state = configs.pop("_aer_state")
        else:
            if "method" not in configs:
                configs["method"] = "density_matrix"
            elif configs["method"] != "density_matrix":
                method = configs["method"]
                raise AerError(f"Method {method} is not supported")
            if isinstance(data, (QuantumCircuit, Instruction)):
                data, aer_state = AerDensityMatrix._from_instruction(data, None, configs)
            elif isinstance(data, list):
                data = self._from_1d_array(np.array(data, dtype=complex))
                data, aer_state = AerDensityMatrix._from_ndarray(data, configs)
            elif isinstance(data, np.ndarray):
                data = self._from_1d_array(data)
                data, aer_state = AerDensityMatrix._from_ndarray(data, configs)
            elif isinstance(data, AerDensityMatrix):
                aer_state = data._aer_state
                if dims is None:
                    dims = data._op_shape._dims_l
                data = data._data.copy()
            elif isinstance(data, DensityMatrix):
                data, aer_state = AerDensityMatrix._from_ndarray(
                    np.array(data.data, dtype=complex), configs
                )
            elif hasattr(data, "to_operator"):
                # If the data object has a 'to_operator' attribute this is given
                # higher preference than the 'to_matrix' method for initializing
                # an Operator object.
                op = data.to_operator()
                data, aer_state = AerDensityMatrix._from_ndarray(op.data, configs)
                if dims is None:
                    dims = op.output_dims()
            elif hasattr(data, "to_matrix"):
                # If no 'to_operator' attribute exists we next look for a
                # 'to_matrix' attribute to a matrix that will be cast into
                # a complex numpy matrix.
                data, aer_state = AerDensityMatrix._from_ndarray(
                    np.asarray(data.to_matrix(), dtype=complex), configs
                )
            else:
                raise AerError(f"Input data is not supported: type={data.__class__}, data={data}")

            self._aer_state = aer_state

        super().__init__(data, dims=dims)

        self._result = None
        self._configs = configs

    def seed(self, value=None):
        """Set the seed for the quantum state RNG."""
        if value is None or isinstance(value, int):
            self._aer_state.set_seed(value)
        else:
            raise AerError(f"This seed is not supported: type={value.__class__}, value={value}")

    def _last_result(self):
        if self._result is None:
            self._result = self._aer_state.last_result()
        return self._result

    def metadata(self):
        """Return result metadata of an operation that executed lastly."""
        if self._last_result() is None:
            raise AerError("AerState was not used and metdata does not exist.")
        return self._last_result()["metadata"]

    def __copy__(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, _memo=None):
        ret = AerDensityMatrix(self._data.copy(), **self._configs)
        ret._op_shape = copy.deepcopy(self._op_shape)
        ret._rng_generator = copy.deepcopy(self._rng_generator)
        return ret

    def conjugate(self):
        return AerDensityMatrix(np.conj(self._data), dims=self.dims())

    def tensor(self, other):
        """Return the tensor product state self ⊗ other.
        Args:
            other (AerDensityMatrix): a quantum state object.
        Returns:
            AerDensityMatrix: the tensor product operator self ⊗ other.
        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, AerDensityMatrix):
            other = AerDensityMatrix(other)
        ret = copy.copy(self)
        ret._data = np.kron(self._data, other._data)
        ret._op_shape = self._op_shape.tensor(other._op_shape)
        return ret

    def expand(self, other):
        """Return the tensor product state other ⊗ self.
        Args:
            other (AerDensityMatrix): a quantum state object.
        Returns:
            AerDensityMatrix: the tensor product state other ⊗ self.
        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, AerDensityMatrix):
            other = AerDensityMatrix(other)
        ret = copy.copy(self)
        ret._data = np.kron(other._data, self._data)
        ret._op_shape = self._op_shape.expand(other._op_shape)
        return ret

    def _add(self, other):
        """Return the linear combination self + other.
        Args:
            other (AerDensityMatrix): a quantum state object.
        Returns:
            AerDensityMatrix: the linear combination self + other.
        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        if not isinstance(other, AerDensityMatrix):
            other = AerDensityMatrix(other)
        self._op_shape._validate_add(other._op_shape)
        ret = copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def sample_memory(self, shots, qargs=None):
        if qargs is None:
            qubits = np.arange(self._aer_state.num_qubits)
        else:
            qubits = np.array(qargs)

        self._aer_state.close()
        self._aer_state.renew()
        self._aer_state.initialize(self._data, copy=False)
        samples = self._aer_state.sample_memory(qubits, shots)
        self._data = self._aer_state.move_to_ndarray()
        return samples

    @staticmethod
    def _from_1d_array(data):
        # Convert statevector into a density matrix
        ndim = data.ndim
        shape = data.shape
        if ndim == 2 and shape[0] == shape[1]:
            pass  # We good
        elif ndim == 1:
            data = np.outer(data, np.conj(data))
        elif ndim == 2 and shape[1] == 1:
            data = np.reshape(data, shape[0])
        else:
            raise QiskitError("Invalid AerDensityMatrix input: not a square matrix.")
        return data

    @staticmethod
    def _from_ndarray(init_data, configs):
        aer_state = AerState(method="density_matrix")

        options = AerSimulator._default_options()
        for config_key, config_value in configs.items():
            if options.get(config_key):
                aer_state.configure(config_key, config_value)

        if len(init_data) == 0:
            raise AerError("initial data must be larger than 0")

        num_qubits = int(np.log2(len(init_data)))

        aer_state.allocate_qubits(num_qubits)
        aer_state.initialize(data=init_data)

        return aer_state.move_to_ndarray(), aer_state

    @classmethod
    def from_instruction(cls, instruction):
        return AerDensityMatrix(instruction)

    @staticmethod
    def _from_instruction(inst, init_data, configs):
        aer_state = AerState(method="density_matrix")

        for config_key, config_value in configs.items():
            aer_state.configure(config_key, config_value)

        basis_gates = BASIS_GATES["density_matrix"]
        custom_insts = ["reset", "kraus", "barrier"]

        aer_state.allocate_qubits(inst.num_qubits)
        num_qubits = inst.num_qubits

        if init_data is not None:
            aer_state.initialize(data=init_data, copy=True)
        else:
            aer_state.initialize()

        if isinstance(inst, QuantumCircuit) and inst.global_phase != 0:
            aer_state.apply_global_phase(inst.global_phase)

        if isinstance(inst, QuantumCircuit):
            AerStatevector._aer_evolve_circuit(
                aer_state, inst, range(num_qubits), basis_gates, custom_insts
            )
        else:
            AerStatevector._aer_evolve_instruction(
                aer_state, inst, range(num_qubits), basis_gates, custom_insts
            )

        return aer_state.move_to_ndarray(), aer_state

    def reset(self, qargs=None):
        # Normally, DensityMatrix.reset returns DensityMatrix, which should
        # be converted to AerDensityMatrix if necessary.
        density_matrix = super().reset(qargs=qargs)
        if isinstance(density_matrix, DensityMatrix):
            density_matrix = AerDensityMatrix(density_matrix)
        return density_matrix

    @classmethod
    def from_label(cls, label):
        return AerDensityMatrix(AerStatevector.from_label(label))

    @staticmethod
    def from_int(i, dims):
        size = np.prod(dims)
        state = np.zeros((size, size), dtype=complex)
        state[i, i] = 1.0
        return AerDensityMatrix(state, dims=dims)

    def to_statevector(self, atol=None, rtol=None):
        """Return a statevector from a pure density matrix.
        Args:
            atol (float): Absolute tolerance for checking operation validity.
            rtol (float): Relative tolerance for checking operation validity.
        Returns:
            AerStatevector: The pure density matrix's corresponding statevector.
                Corresponds to the eigenvector of the only non-zero eigenvalue.
        Raises:
            QiskitError: if the state is not pure.
        """
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        if not is_hermitian_matrix(self.data, atol=atol, rtol=rtol):
            raise QiskitError("Not a valid density matrix (non-hermitian).")

        evals, evecs = np.linalg.eig(self.data)

        nonzero_evals = evals[abs(evals) > atol]
        if len(nonzero_evals) != 1 or not np.isclose(nonzero_evals[0], 1, atol=atol, rtol=rtol):
            raise QiskitError("Density matrix is not a pure state")

        psi = evecs[:, np.argmax(evals)]  # eigenvectors returned in columns.
        return AerStatevector(psi)
