# -*- coding: utf-8 -*

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

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

import math

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer
from qiskit.compiler import transpile, assemble


def build_model_circuit(qreg, creg, circuit=None):
    """Create quantum fourier transform circuit on quantum register qreg."""
    if circuit is None:
        circuit = QuantumCircuit(qreg, creg, name="qft")

    n = len(qreg)

    for i in range(n):
        for j in range(i):
            circuit.cu1(math.pi/float(2**(i-j)), qreg[i], qreg[j])
        circuit.h(qreg[i])
    circuit.measure(qreg, creg)

    return circuit


class QftQasmSimulatorBench:
    params = ([1, 2, 3, 5, 8, 15, 20, 53],
              ['statevector', 'density_matrix', 'stabilizer',
               'extended_stabilizer', 'matrix_product_state'])
    param_names = ['n_qubits', 'simulator_method']

    def setup(self, n, _):
        qr = QuantumRegister(n)
        cr = ClassicalRegister(n)
        self.circuit = build_model_circuit(qr, cr)
        self.sim_backend = Aer.get_backend('qasm_simulator')
        new_circ = transpile(self.circuit, self.sim_backend)
        self.qobj = assemble(new_circ, backend=self.sim_backend,
                             shots=1000)

    def time_qasm_simulator(self, _, simulator_method):
        backend_options = {
            'method': simulator_method,
        }
        job = self.sim_backend.run(self.qobj,
                                   backend_options=backend_options)
        job.result()
