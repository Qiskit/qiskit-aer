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


import unittest
import copy
import pickle
import os
from multiprocessing import Pool

from qiskit.providers.aer.backends.controller_wrappers import (qasm_controller_execute,
                                                               statevector_controller_execute,
                                                               unitary_controller_execute)
from test.terra.reference import ref_cache


class TestControllerExecuteWrappers(unittest.TestCase):
    """Basic functionality tests for pybind-generated wrappers"""

    CFUNCS = [qasm_controller_execute(), statevector_controller_execute(), unitary_controller_execute()]

    def test_deepcopy(self):
        """Test that the functors are deepcopy-able."""
        for cfunc in self.CFUNCS:
            cahpy = copy.deepcopy(cfunc)

    def test_pickleable(self):
        """Test that the functors are pickle-able (directly)."""
        for cfunc in self.CFUNCS:
            bites = pickle.dumps(cfunc)
            cahpy = pickle.loads(bites)

    def test_mappable(self):
        """Test that the functors can be used in a multiprocessing.pool.map call."""
        qobjs = [ref_cache.get_obj(fn) for fn in ['qobj_qasm', 'qobj_statevector', 'qobj_unitary']]
        n = max(os.cpu_count(), 2)
        results = []
        with Pool(processes=n) as p:
            for qobj, cfunc in zip(qobjs, self.CFUNCS):
                rs = p.map(cfunc, [copy.deepcopy(qobj) for _ in range(n)])
                results.append(rs)

        for res in results:
            self.assertEqual(len(res), n)
            for r in res:
                self.assertTrue(r['success'])

if __name__ == '__main__':
    unittest.main()
