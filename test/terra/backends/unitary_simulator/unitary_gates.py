# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
UnitarySimulator standard gate tests
"""

from itertools import product
from ddt import ddt, unpack, data
from numpy.random import default_rng

from qiskit import execute
from qiskit.quantum_info import Operator
from qiskit.providers.aer import UnitarySimulator

from qiskit.circuit.library.standard_gates import (
    CXGate, CYGate, CZGate, DCXGate, HGate, IGate, SGate, SXGate, SXdgGate,
    SdgGate, SwapGate, XGate, YGate, ZGate, TGate, TdgGate, iSwapGate, C3XGate,
    C4XGate, CCXGate, CHGate, CSXGate, CSwapGate, CPhaseGate, CRXGate, CRYGate,
    CRZGate, CU1Gate, CU3Gate, CUGate, PhaseGate, RC3XGate, RCCXGate, RGate,
    RXGate, RXXGate, RYGate, RYYGate, RZGate, RZXGate, RZZGate, U1Gate, U2Gate,
    U3Gate, UGate)


@ddt
class UnitaryGateTests:
    """UnitarySimulator circuit library standard gate tests."""

    SIMULATOR = UnitarySimulator()
    BACKEND_OPTS = {}

    SEED = 8181
    RNG = default_rng(seed=SEED)
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
    BASIS_GATES = [
        None,
        ['id', 'u1', 'u2', 'u3', 'cx'],
        ['id', 'u', 'cx'],
        ['id', 'r', 'cz'],
        ['id', 'rz', 'rx', 'cz'],
        ['id', 'p', 'sx', 'cx']
    ]

    @data(*[(gate_params[0], gate_params[1], basis_gates)
            for gate_params, basis_gates in product(GATES, BASIS_GATES)])
    @unpack
    def test_gate(self, gate_cls, num_params, basis_gates):
        """Test standard gate simulation."""
        circuit = self.gate_circuit(gate_cls,
                                    num_params=num_params,
                                    rng=self.RNG)
        target = Operator(circuit)
        result = execute(circuit, self.SIMULATOR,
                         basis_gates=basis_gates).result()
        self.assertSuccess(result)
        value = Operator(result.get_unitary(0))
        self.assertTrue(target.equiv(value),
                        msg='{}, basis_gates = {}'.format(
                            gate_cls.__name__, basis_gates))
