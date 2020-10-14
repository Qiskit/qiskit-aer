# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
QasmSimulator Integration Tests for circuit library standard gates
"""

from ddt import ddt, unpack, data
from numpy.random import default_rng

from qiskit import execute
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.extensions import *

from qiskit.circuit.library.standard_gates import (
    CXGate, CYGate, CZGate, DCXGate, HGate, IGate, SGate, SXGate, SXdgGate,
    SdgGate, SwapGate, XGate, YGate, ZGate, TGate, TdgGate, iSwapGate, C3XGate,
    C4XGate, CCXGate, CHGate, CSXGate, CSwapGate, CPhaseGate, CRXGate, CRYGate,
    CRZGate, CU1Gate, CU3Gate, CUGate, PhaseGate, RC3XGate, RCCXGate, RGate,
    RXGate, RXXGate, RYGate, RYYGate, RZGate, RZXGate, RZZGate, U1Gate, U2Gate,
    U3Gate, UGate)

GATES = [
    # Clifford Gates
    (CXGate, 0),
    (CYGate, 0),
    (CZGate, 0),
    (DCXGate, 0),
    (HGate, 0),
    (IGate, 0),
    (SGate, 0),
    (SXGate, 0),
    (SXdgGate, 0),
    (SdgGate, 0),
    (SwapGate, 0),
    (XGate, 0),
    (YGate, 0),
    (ZGate, 0),
    (TGate, 0),
    # Non-Clifford Gates
    (TdgGate, 0),
    (iSwapGate, 0),
    (C3XGate, 0),
    (C4XGate, 0),
    (CCXGate, 0),
    (CHGate, 0),
    (CSXGate, 0),
    (CSwapGate, 0),
    # Parameterized Gates
    (CPhaseGate, 1),
    (CRXGate, 1),
    (CRYGate, 1),
    (CRZGate, 1),
    (CU1Gate, 1),
    (CU3Gate, 3),
    (CUGate, 4),
    (PhaseGate, 1),
    (RC3XGate, 1),
    (RCCXGate, 1),
    (RGate, 2),
    (RXGate, 1),
    (RXXGate, 1),
    (RYGate, 1),
    (RYYGate, 1),
    (RZGate, 1),
    (RZXGate, 1),
    (RZZGate, 1),
    (U1Gate, 1),
    (U2Gate, 2),
    (U3Gate, 3),
    (UGate, 3)
]


@ddt
class QasmStandardGateStatevectorTests:
    """QasmSimulator standard gate library tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}
    SEED = 8181
    RNG = default_rng(seed=SEED)

    @data(*GATES)
    @unpack
    def test_gate_statevector(self, gate_cls, num_params):
        """Test standard gate simulation test."""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state'
        ]

        circuit = self.gate_circuit(gate_cls,
                                    num_params=num_params,
                                    rng=self.RNG)
        target = Statevector.from_instruction(circuit)

        # Add snapshot and execute
        circuit.snapshot_statevector('final')
        backend_options = self.BACKEND_OPTS
        method = backend_options.pop('method', 'automatic')
        backend = self.SIMULATOR
        backend.set_options(method=method)
        result = execute(circuit, backend, shots=1, **backend_options).result()

        # Check results
        success = getattr(result, 'success', False)
        msg = '{}, method = {}'.format(gate_cls.__name__, method)
        if method not in SUPPORTED_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success, msg=msg)
            self.assertSuccess(result)
            snapshots = result.data(0).get("snapshots", {}).get("statevector", {})
            value = snapshots.get('final', [None])[0]
            fidelity = state_fidelity(target, value)
            self.assertGreater(fidelity, 0.99999, msg=msg)


@ddt
class QasmStandardGateDensityMatrixTests:
    """QasmSimulator standard gate library tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}
    SEED = 9997
    RNG = default_rng(seed=SEED)

    @data(*GATES)
    @unpack
    def test_gate_density_matrix(self, gate_cls, num_params):
        """Test standard gate simulation test."""
        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust'
        ]
        circuit = self.gate_circuit(gate_cls,
                                    num_params=num_params,
                                    rng=self.RNG)
        target = Statevector.from_instruction(circuit)

        # Add snapshot and execute
        circuit.snapshot_density_matrix('final')
        backend_options = self.BACKEND_OPTS
        method = backend_options.pop('method', 'automatic')
        backend = self.SIMULATOR
        backend.set_options(method=method)
        result = execute(circuit, backend, shots=1, **backend_options).result()

        # Check results
        success = getattr(result, 'success', False)
        msg = '{}, method = {}'.format(gate_cls.__name__, method)
        if method not in SUPPORTED_METHODS:
            self.assertFalse(success)
        else:
            self.assertTrue(success, msg=msg)
            self.assertSuccess(result)
            snapshots = result.data(0).get("snapshots",
                                           {}).get("density_matrix", {})
            value = snapshots.get('final', [{'value': None}])[0]['value']
            fidelity = state_fidelity(target, value)
            self.assertGreater(fidelity, 0.99999, msg=msg)
