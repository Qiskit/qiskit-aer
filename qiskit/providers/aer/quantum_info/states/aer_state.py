# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021, 2022
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
import numpy as np
from qiskit.providers.aer.backends.controller_wrappers import AerStateWrapper
from ...backends.aerbackend import AerError


class AerState:

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

    def __init__(self):
        """State that handles cpp quantum state safely"""
        self._shift_to_created_state()
        self._state = AerStateWrapper()
        self._method = 'statevector'
        self._init_data = None
        self._last_qubit = -1

    def _is_created_state(self):
        return (not self._initialized) and (not self._closed)

    def _is_initialized_state(self):
        return self._initialized and (not self._closed)

    def _is_closed_state(self):
        return self._initialized and self._closed

    def _shift_to_created_state(self):
        self._initialized = False
        self._closed = False

    def _assert_created_state(self):
        if self._is_initialized_state():
            raise AerError('AerState was already initialized.')
        if self._is_closed_state():
            raise AerError('AerState has already been closed.')

    def _shift_to_initialized_state(self):
        if not self._is_created_state():
            raise AerError('unexpected state transition')
        self._initialized = True
        self._closed = False

    def _assert_initialized_state(self):
        if not self._is_initialized_state():
            raise AerError('AerState has not been initialized yet.')
        if self._is_closed_state():
            raise AerError('AerState has already been closed.')

    def _shift_to_closed_state(self):
        if (not self._is_initialized_state()) and (not self._is_created_state()):
            raise AerError('unexpected state transition')
        self._initialized = True
        self._closed = True

    def configure(self, key, value):
        """configure AerState"""
        self._assert_created_state()

        if not isinstance(key, str):
            raise AerError('AerState is configured with a str key')
        if not isinstance(value, str):
            value = str(value)
        self._state.configure(key, value)

    def initialize(self, data=None):
        """Initialize state"""
        self._assert_created_state()

        if data is None:
            self._state.initialize()
        elif isinstance(data, np.ndarray):
            self._initialize_with_ndarray(data)
        else:
            raise AerError('unsupported init data.')

        self._shift_to_initialized_state()

    def _initialize_with_ndarray(self, data):
        if AerState._is_in_use(data):
            raise AerError('another AerState owns this data')

        num_of_qubits = int(np.log2(len(data)))
        if len(data) != np.power(2, num_of_qubits):
            raise AerError('length of init data must be power of two')
        qubits = [qubit for qubit in range(num_of_qubits)]

        initialized = False
        if(isinstance(data, np.ndarray) and
           self._method == 'statevector' and
           self._state.initialize_statevector(num_of_qubits, data)):
            self._init_data = data
            AerState._in_use(data)
            initialized = True

        if not initialized:
            self._state.reallocate_qubits(num_of_qubits)
            self._state.initialize()
            self._state.apply_initialize(qubits, data)

    def close(self):
        """Safely release all releated memory"""
        self._unbind_data()
        self._shift_to_closed_state()

    def _unbind_data(self):
        if self._init_data is not None:
            # intentional memory leak
            # memory of self._state will be freed when self._init_data is collected
            self._state.move_to_buffer()
            AerState._not_in_use(self._init_data)
            self._init_data = None
        else:
            self._state.clear()

    def move_to_ndarray(self):
        if self._init_data is not None:
            ret = self._init_data
            self._unbind_data()
        else:
            ret = self._state.move_to_ndarray()
        return ret

    def allocate_qubits(self, num_of_qubits):
        self._assert_created_state()
        if num_of_qubits <= 0:
            raise AerError('invalid number of qubits: {}'.format(num_of_qubits))
        allocated = self._state.allocate_qubits(num_of_qubits)
        self._last_qubit = allocated[len(allocated) - 1]

    def _assert_in_allocated_qubits(self, qubit):
        if isinstance(qubit, list):
            for q in qubit:
                self._assert_in_allocated_qubits(q)
        elif qubit < 0 or qubit > self._last_qubit:
            raise AerError('invalid qubit: index={}'.format(qubit))

    def apply_unitary(self, qubits, data):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(qubits)
        # Convert to numpy array in case not already an array
        data = np.array(data, dtype=complex)
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qubits = int(np.log2(input_dim))
        if input_dim != output_dim or 2**num_qubits != input_dim:
            raise AerError("Input matrix is not an N-qubit operator.")
        if len(qubits) != num_qubits:
            raise AerError("Input matrix and qubits are insonsistent.")
        # update state
        self._state.apply_unitary(qubits, data)

    def apply_multiplexer(self, control_qubits, target_qubits, mats):
        self._assert_initialized_state()
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
        self._state.apply_multiplexer(control_qubits, target_qubits, mats)

    def apply_diagonal(self, qubits, diag):
        self._assert_initialized_state()
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
        self._state.apply_diagonal(qubits, diag)

    def apply_mcx(self, control_qubits, target_qubit):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._state.apply_mcx(control_qubits + [target_qubit])

    def apply_mcy(self, control_qubits, target_qubit):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._state.apply_mcy(control_qubits + [target_qubit])

    def apply_mcz(self, control_qubits, target_qubit):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._state.apply_mcz(control_qubits + [target_qubit])

    def apply_mcphase(self, control_qubits, target_qubit, phase):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._state.apply_mcphase(control_qubits + [target_qubit], phase)

    def apply_mcu(self, control_qubits, target_qubit, theta, phi, lamb):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(target_qubit)
        # update state
        self._state.apply_mcu(control_qubits + [target_qubit], theta, phi, lamb)

    def apply_mcswap(self, control_qubits, qubit0, qubit1):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(control_qubits)
        self._assert_in_allocated_qubits(qubit0)
        self._assert_in_allocated_qubits(qubit1)
        # update state
        self._state.apply_mcswap(control_qubits + [qubit0, qubit1], theta, phi, lamb)

    def apply_measure(self, qubits):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(qubits)
        # measure and update state
        return self._state.apply_measure(qubits)

    def apply_reset(self, qubits):
        self._assert_initialized_state()
        self._assert_in_allocated_qubits(qubits)
        # update state
        return self._state.apply_reset(qubits)

    def probability(self, outcome):
        self._assert_initialized_state()
        # retrieve probability
        return self._state.probability(outcome)

    def probabilities(self, qubits=None):
        self._assert_initialized_state()
        if qubits is None:
            qubits = [q for q in range(self._last_qubit + 1)]
        else:
            self._assert_in_allocated_qubits(qubits)

        # retrieve probability
        return self._state.probabilities(qubits)
