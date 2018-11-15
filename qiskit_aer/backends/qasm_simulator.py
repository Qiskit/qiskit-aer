# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

from .aerbackend import AerBackend
from qasm_controller_wrapper import QasmControllerWrapper


class QasmSimulator(AerBackend):
    """Aer quantum circuit simulator"""

    DEFAULT_CONFIGURATION = {
        'name': 'qasm_simulator',
        'url': 'NA',
        'simulator': True,
        'local': True,
        'description': 'A C++ statevector simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,ccx,swap'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy(),
                         QasmControllerWrapper(), provider=provider)

    def _validate(self, qobj):
        # TODO
        return
