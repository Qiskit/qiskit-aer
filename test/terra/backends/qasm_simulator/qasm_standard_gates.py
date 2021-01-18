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

from itertools import product
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
    U3Gate, UGate, MCXGate, MCPhaseGate, MCXGrayCode)

GATES = [
    # Clifford Gates
    (CXGate, 0, False),
    (CYGate, 0, False),
    (CZGate, 0, False),
    (DCXGate, 0, False),
    (HGate, 0, False),
    (IGate, 0, False),
    (SGate, 0, False),
    (SXGate, 0, False),
    (SXdgGate, 0, False),
    (SdgGate, 0, False),
    (SwapGate, 0, False),
    (XGate, 0, False),
    (YGate, 0, False),
    (ZGate, 0, False),
    (TGate, 0, False),
    # Non-Clifford Gates
    (TdgGate, 0, False),
    (iSwapGate, 0, False),
    (C3XGate, 0, False),
    (C4XGate, 0, False),
    (CCXGate, 0, False),
    (CHGate, 0, False),
    (CSXGate, 0, False),
    (CSwapGate, 0, False),
    # Parameterized Gates
    (CPhaseGate, 1, False),
    (CRXGate, 1, False),
    (CRYGate, 1, False),
    (CRZGate, 1, False),
    (CU1Gate, 1, False),
    (CU3Gate, 3, False),
    (CUGate, 4, False),
    (PhaseGate, 1, False),
    (RC3XGate, 1, False),
    (RCCXGate, 1, False),
    (RGate, 2, False),
    (RXGate, 1, False),
    (RXXGate, 1, False),
    (RYGate, 1, False),
    (RYYGate, 1, False),
    (RZGate, 1, False),
    (RZXGate, 1, False),
    (RZZGate, 1, False),
    (U1Gate, 1, False),
    (U2Gate, 2, False),
    (U3Gate, 3, False),
    (UGate, 3, False),
    (MCXGate, 0, True),
    (MCPhaseGate, 1, True),
    (MCXGrayCode, 0, True)
]
BASIS_GATES = [
    None,
    ['id', 'u1', 'u2', 'u3', 'cx'],  # Waltz
    ['id', 'rz', 'sx', 'x', 'cx']
]


@ddt
class QasmStandardGateStatevectorTests:
    """QasmSimulator standard gate library tests."""

    SIMULATOR = QasmSimulator()
    BACKEND_OPTS = {}
    SEED = 8181
    RNG = default_rng(seed=SEED)

    @data(*[(gate_params[0], gate_params[1], gate_params[2], basis_gates)
        for gate_params, basis_gates in product(GATES, BASIS_GATES)])
    @unpack
    def test_gate_statevector(self, gate_cls, num_angles, has_ctrl_qubits, basis_gates):
        """Test standard gate simulation test."""

        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'matrix_product_state'
        ]

        backend_options = self.BACKEND_OPTS.copy()
        method = backend_options.pop('method', 'automatic')
        backend = self.SIMULATOR
        backend.set_options(method=method)

        circuits = self.gate_circuits(gate_cls,
                                      num_angles=num_angles,
                                      has_ctrl_qubits=has_ctrl_qubits,
                                      rng=self.RNG)

        for circuit in circuits:
            target = Statevector.from_instruction(circuit)

            # Add snapshot and execute
            circuit.snapshot_statevector('final')
            result = execute(circuit, backend, shots=1, basis_gates=basis_gates,
                             optimization_level=0, **backend_options).result()

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

    @data(*[(gate_params[0], gate_params[1], gate_params[2], basis_gates)
        for gate_params, basis_gates in product(GATES, BASIS_GATES)])
    @unpack
    def test_gate_density_matrix(self, gate_cls, num_angles, has_ctrl_qubits, basis_gates):
        """Test standard gate simulation test."""
        SUPPORTED_METHODS = [
            'automatic', 'statevector', 'statevector_gpu', 'statevector_thrust',
            'density_matrix', 'density_matrix_gpu', 'density_matrix_thrust'
        ]

        backend_options = self.BACKEND_OPTS.copy()
        method = backend_options.pop('method', 'automatic')
        backend = self.SIMULATOR
        backend.set_options(method=method)
        
        circuits = self.gate_circuits(gate_cls,
                                      num_angles=num_angles,
                                      has_ctrl_qubits=has_ctrl_qubits,
                                      rng=self.RNG)

        for circuit in circuits:
            target = Statevector.from_instruction(circuit)

            # Add snapshot and execute
            circuit.snapshot_density_matrix('final')

            result = execute(circuit, backend, shots=1, basis_gates=basis_gates,
                             optimization_level=0, **backend_options).result()

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
            
