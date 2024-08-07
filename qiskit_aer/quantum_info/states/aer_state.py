# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
State class that handles internal C++ state safely
"""
from enum import Enum
import numpy as np

# pylint: disable=import-error, no-name-in-module
from qiskit_aer.backends.controller_wrappers import AerStateWrapper
from ...backends.aerbackend import AerError


class _STATE(Enum):
    INITIALIZING = 1
    ALLOCATED = 2
    MAPPED = 3
    MOVED = 4
    RELEASED = 5
    CLOSED = 6


class AerState:
    """Internal class to access state of Aer."""

    _data_in_use = {}

    @staticmethod
    def _in_use(data):
        AerState._data_in_use[id(data)] = data

    @staticmethod
    def _not_in_use(data):
        del AerState._data_in_use[id(data)]

    @staticmethod
    def _is_in_use(data):
        return id(data) in AerState._data_in_use

    def __init__(self, **kwargs):
        """State that handles cpp quantum state safely"""
        self._state = _STATE.INITIALIZING
        self._native_state = AerStateWrapper()
        self._init_data = None
        self._moved_data = None
        self._last_qubit = -1
        self._configs = {}

        self._method = None

        for key, value in kwargs.items():
            self.configure(key, value)

    def renew(self):
        """Renew AerState for reuse"""
        self._assert_closed()
        self._state = _STATE.INITIALIZING
        self._init_data = None
        self._moved_data = None
        self._last_qubit = -1

    def _assert_initializing(self):
        if self._state != _STATE.INITIALIZING:
            raise AerError("AerState was already initialized.")

    def _assert_allocated_or_mapped_or_moved(self):
        if self._state == _STATE.INITIALIZING:
            raise AerError("AerState has not been initialized yet.")
        if self._state == _STATE.CLOSED:
            raise AerError("AerState has already been closed.")

    def _assert_allocated_or_mapped(self):
        if self._state == _STATE.INITIALIZING:
            raise AerError("AerState has not been initialized yet.")
        if self._state == _STATE.MOVED:
            raise AerError("AerState has already been moved.")
        if self._state == _STATE.CLOSED:
            raise AerError("AerState has already been closed.")

    def _assert_mapped_or_moved(self):
        if self._state == _STATE.INITIALIZING:
            raise AerError("AerState has not been initialized yet.")
        if self._state == _STATE.ALLOCATED:
            raise AerError("AerState has not been moved yet.")
        if self._state == _STATE.CLOSED:
            raise AerError("AerState has already been closed.")

    def _assert_closed(self):
        if self._state != _STATE.CLOSED:
            raise AerError("AerState is not closed.")

    def _allocated(self):
        if self._state != _STATE.INITIALIZING:
            raise AerError("unexpected state transition: {self._state}->{_STATE.ALLOCATED}")
        self._state = _STATE.ALLOCATED

    def _mapped(self):
        if self._state != _STATE.INITIALIZING:
            raise AerError("unexpected state transition: {self._state}->{_STATE.MAPPED}")
        self._state = _STATE.MAPPED

    def _released(self):
        if self._state != _STATE.MAPPED:
            raise AerError("unexpected state transition: {self._state}->{_STATE.RELEASED}")
        self._state = _STATE.RELEASED

    def _moved(self):
        if self._state != _STATE.ALLOCATED:
            raise AerError("unexpected state transition: {self._state}->{_STATE.MOVED}")
        self._state = _STATE.MOVED

    def _closed(self):
        if self._state not in (_STATE.MOVED, _STATE.MAPPED, _STATE.RELEASED):
            raise AerError("unexpected state transition: {self._state}->{_STATE.CLOSED}")
        self._state = _STATE.CLOSED

    def configure(self, key, value):
        """configure AerState with options of `AerSimulator`."""
        self._assert_initializing()

        if not isinstance(key, str):
            raise AerError("AerState is configured with a str key")
        if not isinstance(value, str):
            value = str(value)

        self._configs[key] = value
        self._native_state.configure(key, value)

        if key == "method":
            self._method = value

    def configuration(self):
        """return configuration"""
        return self._configs.copy()

    def initialize(self, data=None, copy=True):
        """initialize state."""
        self._assert_initializing()

        if not self._method:
            raise AerError("method is not configured yet.")

        if data is None:
            self._native_state.initialize()
            self._allocated()
            return True
        elif isinstance(data, np.ndarray):
            return self._initialize_with_ndarray(data, copy)
        else:
            raise AerError(f"unsupported init data: {data.__class__}")

    def _initialize_with_ndarray(self, data, copy):
        if AerState._is_in_use(data) and not copy:
            raise AerError("another AerState owns this data")

        num_of_qubits = int(np.log2(len(data)))
        if len(data) != np.power(2, num_of_qubits):
            raise AerError("length of init data must be power of two")

        init = False
        if self._method == "statevector":
            init = self._native_state.initialize_statevector(num_of_qubits, data, copy)
        elif self._method == "density_matrix":
            if data.shape != (len(data), len(data)):
                raise AerError("shape of init data must be a pair of power of two")
            init = self._native_state.initialize_density_matrix(num_of_qubits, data, copy)

        if init:
            if not copy:
                self._init_data = data
                AerState._in_use(data)
                self._mapped()
            else:
                self._allocated()
        else:
            # slow path
            self._native_state.reallocate_qubits(num_of_qubits)
            self._native_state.initialize()
            if not data.flags.c_contiguous and not data.flags.f_contiguous:
                data = np.ascontiguousarray(data)
            if self._method == "statevector":
                self._native_state.apply_initialize(range(num_of_qubits), data)
            elif self._method == "density_matrix":
                self._native_state.set_density_matrix(range(num_of_qubits), data)
            else:
                self._native_state.apply_initialize(range(num_of_qubits), data)
            self._allocated()
            copy = True

        self._last_qubit = num_of_qubits - 1
        return copy

    def method(self):
        """return method to simulate"""
        return self._method

    def set_seed(self, value=None):
        """initialize seed with a specified value"""
        if value is None:
            self._native_state.set_random_seed()
        else:
            self._native_state.set_seed(value)

    def close(self):
        """Safely release all releated memory."""
        self._assert_allocated_or_mapped_or_moved()
        if self._state == _STATE.ALLOCATED:
            self._native_state.move_to_ndarray()

        self._assert_mapped_or_moved()
        if self._state == _STATE.MAPPED:
            # native memory will be freed when self._init_data is collected.
            # this call of move_to_buffer() is to avoid free in C++
            self._native_state.move_to_buffer()
            AerState._not_in_use(self._init_data)

        self._native_state.clear()
        self._closed()

    def move_to_ndarray(self):
        """move memory to ndarray if it is allocated, otherwise return mapped ndarray."""
        self._assert_allocated_or_mapped_or_moved()

        if self._state == _STATE.MAPPED:
            ret = self._init_data
            # native memory will be freed when self._init_data is collected.
            # this call of move_to_buffer() is to avoid free in C++
            self._native_state.move_to_buffer()
            AerState._not_in_use(self._init_data)
            self._released()
        elif self._state == _STATE.RELEASED:
            ret = self._init_data
        elif self._state == _STATE.MOVED:
            ret = self._moved_data
        else:
            if self._method == "density_matrix":
                self._moved_data = self._native_state.move_to_matrix()
            else:
                self._moved_data = self._native_state.move_to_ndarray()
            ret = self._moved_data
            self._moved()
        return ret

    def allocate_qubits(self, num_of_qubits):
        """allocate qubits."""
        self._assert_initializing()
        if num_of_qubits <= 0:
            raise AerError(f"invalid number of qubits: {num_of_qubits}")
        allocated = self._native_state.allocate_qubits(num_of_qubits)
        self._last_qubit = allocated[len(allocated) - 1]

    def _assert_in_allocated_qubits(self, qubit):
        if hasattr(qubit, "__iter__"):
            for q in qubit:
                self._assert_in_allocated_qubits(q)
        elif qubit < 0 or qubit > self._last_qubit:
            raise AerError(f"invalid qubit: index={qubit}")

    @property
    def num_qubits(self):
        """return a number of allocate qubits."""
        return self._last_qubit + 1

    def flush(self):
        """apply all buffered operations.
        Some gate operations are not evaluated immediately.
        This method guarantees that all called operations are evaluated.
        """
        self._assert_allocated_or_mapped()
        return self._native_state.flush()

    def last_result(self):
        """return a result of a operation fluhsed in the last."""
        self._assert_allocated_or_mapped_or_moved()
        return self._native_state.last_result()

    def apply_global_phase(self, phase):
        """apply global phase"""
        self._assert_allocated_or_mapped()
        self._native_state.apply_global_phase(phase)

    def apply_unitary(self, qubits, data):
        """apply a unitary gate."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(qubits)
        # Convert to numpy array in case not already an array
        data = np.array(data, dtype=complex)
        # Check input is N-qubit matrix
        input_dim = data.shape[0]
        output_dim = data.shape[1]
        num_qubits = int(np.log2(input_dim))
        if input_dim != output_dim or 2**num_qubits != input_dim:
            raise AerError("Input matrix is not an N-qubit operator.")
        if len(qubits) != num_qubits:
            raise AerError("Input matrix and qubits are insonsistent.")
        # update state
        self._native_state.apply_unitary(qubits, data)

    def apply_multiplexer(self, control_qubits, target_qubits, mats):
        """apply a multiplexer operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubits)
        if not isinstance(mats, list):
            raise AerError("Input must be a list of ndarray.")
        if len(mats) != (2 ** len(control_qubits)):
            raise AerError("Input length must be 2 ** number of control gates.")
        # Convert to np array in case not already an array
        mats = [np.array(mat, dtype=complex) for mat in mats]
        # Check input is N-qubit matrix
        input_dim, output_dim = mats[0].shape
        num_target_qubits = int(np.log2(input_dim))
        if input_dim != output_dim or 2**num_target_qubits != input_dim:
            raise AerError("Input matrix is not an N-qubit operator.")
        for mat in mats[1:]:
            if mat.shape != mats[0].shape:
                raise AerError("Input matrix is not an N-qubit operator.")
        if len(target_qubits) != num_target_qubits:
            raise AerError("Input matrix and qubits are insonsistent.")
        # update state
        self._native_state.apply_multiplexer(control_qubits, target_qubits, mats)

    def apply_diagonal(self, qubits, diag):
        """apply a diagonal matrix operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(qubits)
        # Convert to numpy array in case not already an array
        diag = np.array(diag, dtype=complex)
        # Check input is N-qubit vector
        input_dim = diag.shape[0]
        num_qubits = int(np.log2(input_dim))
        if 2**num_qubits != input_dim:
            raise AerError("Input vector is not an N-qubit operator.")
        if len(qubits) != num_qubits:
            raise AerError("Input vector and qubits are insonsistent.")
        # update state
        self._native_state.apply_diagonal(qubits, diag)

    def apply_x(self, target_qubit):
        """apply a x operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_x(target_qubit)

    def apply_cx(self, control_qubit, target_qubit):
        """apply a cx operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubit)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_cx([control_qubit, target_qubit])

    def apply_mcx(self, control_qubits, target_qubit):
        """apply a mcx operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_mcx(control_qubits + [target_qubit])

    def apply_y(self, target_qubit):
        """apply a y operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_y(target_qubit)

    def apply_cy(self, control_qubit, target_qubit):
        """apply a cy operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubit)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_cy([control_qubit, target_qubit])

    def apply_mcy(self, control_qubits, target_qubit):
        """apply a mcy operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_mcy(control_qubits + [target_qubit])

    def apply_z(self, target_qubit):
        """apply a z operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_z(target_qubit)

    def apply_cz(self, control_qubit, target_qubit):
        """apply a cz operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubit)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_cz([control_qubit, target_qubit])

    def apply_mcz(self, control_qubits, target_qubit):
        """apply a mcz operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_mcz(control_qubits + [target_qubit])

    def apply_mcphase(self, control_qubits, target_qubit, phase):
        """apply a mcphase operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_mcphase(control_qubits + [target_qubit], phase)

    def apply_h(self, target_qubit):
        """apply a h operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_h(target_qubit)

    def apply_u(self, target_qubit, theta, phi, lamb):
        """apply a u operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_u(target_qubit, theta, phi, lamb)

    def apply_cu(self, control_qubit, target_qubit, theta, phi, lamb, gamma):
        """apply a cu operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubit)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_cu([control_qubit, target_qubit], theta, phi, lamb, gamma)

    def apply_mcu(self, control_qubits, target_qubit, theta, phi, lamb, gamma):
        """apply a mcu operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._native_state.apply_mcu(control_qubits + [target_qubit], theta, phi, lamb, gamma)

    def apply_mcswap(self, control_qubits, qubit0, qubit1):
        """apply a mcswap operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(qubit0)
        self._assert_in_allocated_qubits(qubit1)
        # update state
        self._native_state.apply_mcswap(control_qubits + [qubit0, qubit1])

    def apply_measure(self, qubits):
        """apply a measure operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(qubits)
        # measure and update state
        return self._native_state.apply_measure(qubits)

    def apply_initialize(self, qubits, vec):
        """apply an initialize operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(qubits)
        # update state
        return self._native_state.apply_initialize(qubits, vec)

    def apply_reset(self, qubits):
        """apply a reset operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(qubits)
        # update state
        return self._native_state.apply_reset(qubits)

    def apply_kraus(self, qubits, krausops):
        """apply a kraus operation."""
        self._assert_allocated_or_mapped()
        self._assert_in_allocated_qubits(qubits)
        # update state
        return self._native_state.apply_kraus(qubits, krausops)

    def probability(self, outcome):
        """return a probability of `outcome`."""
        self._assert_allocated_or_mapped()
        # retrieve probability
        return self._native_state.probability(outcome)

    def probabilities(self, qubits=None):
        """return probabilities of `qubits`."""
        self._assert_allocated_or_mapped()
        if qubits is None:
            qubits = range(self._last_qubit + 1)
        else:
            self._assert_in_allocated_qubits(qubits)

        # retrieve probability
        return self._native_state.probabilities(qubits)

    def sample_counts(self, qubits=None, shots=1024):
        """samples all the qubits."""
        self._assert_allocated_or_mapped()
        if qubits is None:
            qubits = range(self._last_qubit + 1)
        else:
            self._assert_in_allocated_qubits(qubits)
        return self._native_state.sample_counts(qubits, shots)

    def sample_memory(self, qubits=None, shots=1024):
        """samples all the qubits."""
        self._assert_allocated_or_mapped()
        if qubits is None:
            qubits = range(self._last_qubit + 1)
        else:
            self._assert_in_allocated_qubits(qubits)
        return self._native_state.sample_memory(qubits, shots)
