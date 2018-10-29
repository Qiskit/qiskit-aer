# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

import json
import numpy as np

from .aerbackend import AerBackend
from qv_wrapper import QvSimulatorWrapper


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
                         QvSimulatorWrapper(), provider=provider,
                         json_decoder=QasmSimulatorJSONDecoder)

    def _validate(self, qobj):
        # TODO
        return


class QasmSimulatorJSONDecoder(json.JSONDecoder):
    """
    JSON decoder for the output with complex vector snapshots.

    This converts complex vectors and matrices into numpy arrays
    for the following keys.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def decode_complex(self, obj):
        if isinstance(obj, list) and len(obj) == 2:
            if obj[1] == 0.:
                obj = obj[0]
            else:
                obj = obj[0] + 1j * obj[1]
        return obj

    def decode_complex_vector(self, obj):
        if isinstance(obj, list):
            obj = np.array([self.decode_complex(z) for z in obj], dtype=complex)
        return obj

    # pylint: disable=method-hidden
    def object_hook(self, obj):
        """Special decoding rules for simulator output."""

        # Decode snapshots
        if 'snapshots' in obj:
            # Decode state
            if 'state' in obj['snapshots']:
                for key in obj['snapshots']['state']:
                    tmp = [self.decode_complex_vector(vec)
                           for vec in obj['snapshots']['state'][key]]
                    obj['snapshots']['state'][key] = tmp
            # Decode observables
            if 'observables' in obj['snapshots']:
                for key in obj['snapshots']['observables']:
                    for j, val in enumerate(obj['snapshots']['observables'][key]):
                        obj['snapshots']['observables'][key][j]['value'] = self.decode_complex(val['value'])
        return obj
