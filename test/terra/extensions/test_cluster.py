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
import logging
import concurrent.futures
import numpy as np

from qiskit import Aer
from qiskit.providers.aer.backends.cluster.utils import split
from qiskit.circuit.random import random_circuit
from qiskit import execute, transpile, assemble
from qiskit.assembler.disassemble import disassemble
from qiskit.providers.aer.backends.cluster import ClusterBackend
from qiskit.providers.aer.backends import PulseSimulator
from qiskit.providers.aer.pulse.system_models.pulse_system_model import PulseSystemModel
from qiskit.providers.aer.pulse.system_models.hamiltonian_model import HamiltonianModel
from qiskit.pulse import (Schedule, Play, ShiftPhase, SetPhase, Delay, Acquire,
                          Waveform, DriveChannel, ControlChannel,
                          AcquireChannel, MemorySlot)
from test.terra import common
from test.terra.reference import ref_algorithms
from test.terra.reference import ref_pauli_noise
from test.terra.reference.ref_snapshot_expval import (
    snapshot_expval_circuits, snapshot_expval_counts, snapshot_expval_labels,
    snapshot_expval_pre_meas_values, snapshot_expval_circuit_parameterized,
    snapshot_expval_final_statevecs)
from test.terra.reference import ref_1q_clifford
from test.terra.backends.qasm_simulator.qasm_noncliffords import QasmNonCliffordTestsTGate
from test.terra.backends.statevector_simulator.statevector_gates import StatevectorGateTests
from test.terra.backends.unitary_simulator.unitary_gates import UnitaryGateTests

DASK_TESTS=False
try:
    from dask.distributed import LocalCluster, Client
    DASK_TESTS=True
except ImportError:
    logging.warn('Dask not installed: skipping Dask tests')

def dump(obj):
    for attr in dir(obj):
        if hasattr( obj, attr ):
            print( "obj.%s = %s" % (attr, getattr(obj, attr)))
            print()


class CBFixture(common.QiskitAerTestCase):
    """ClusterBackend extension tests"""

    @classmethod
    def setUpClass(cls):
        """Override me with an executor init."""
        cls._test_executor = None

    @classmethod
    def tearDownClass(cls):
        if cls._test_executor:
            cls._test_executor.shutdown()

class ThreadPoolFixture(CBFixture):
    @classmethod
    def setUpClass(cls):
        super(ThreadPoolFixture, cls).setUpClass()
        cls._test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

class DaskFixture(CBFixture):
    @classmethod
    def setUpClass(cls):
        super(DaskFixture, cls).setUpClass()
        if DASK_TESTS:
            cls._test_executor = Client(address=LocalCluster(n_workers=1, processes=True))

    def setUp(self):
        super(DaskFixture, self).setUp()
        if not DASK_TESTS:
            self.skipTest('Dask not installed, skipping ClusterBackend-dask tests')

#class ProcessPoolFixture(CBFixtor):
#    @classmethod
#    def setUpClass(cls):
#        super(TestProcessPoolExecutor, cls).setUpClass()
#        cls._test_executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

class TestQasmDask(DaskFixture,
                   QasmNonCliffordTestsTGate):
    @classmethod
    def setUpClass(cls):
        super(TestQasmDask, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('qasm_simulator'), cls._test_executor)

class TestQasmThread(ThreadPoolFixture,
                     QasmNonCliffordTestsTGate):
    @classmethod
    def setUpClass(cls):
        super(TestQasmThread, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('qasm_simulator'), cls._test_executor)

class TestStatevectorDask(DaskFixture,
                          StatevectorGateTests):
    @classmethod
    def setUpClass(cls):
        super(TestStatevectorDask, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('statevector_simulator'), cls._test_executor)

class TestStatevectorThread(ThreadPoolFixture,
                          StatevectorGateTests):
    @classmethod
    def setUpClass(cls):
        super(TestStatevectorThread, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('statevector_simulator'), cls._test_executor)

class TestUnitaryDask(DaskFixture,
                      UnitaryGateTests):
    @classmethod
    def setUpClass(cls):
        super(TestUnitaryDask, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('unitary_simulator'), cls._test_executor)

class TestUnitaryThread(ThreadPoolFixture,
                        UnitaryGateTests):
    @classmethod
    def setUpClass(cls):
        super(TestUnitaryThread, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('unitary_simulator'), cls._test_executor)

class TestClusterBackendUtils(ThreadPoolFixture):

    @classmethod
    def setUpClass(cls):
        super(TestClusterBackendUtils, cls).setUpClass()
        cls.SIMULATOR = ClusterBackend(Aer.get_backend('qasm_simulator'), cls._test_executor)

    @staticmethod
    def parameterized_circuits(shots=1000, measure=True, snapshot=False):
        """Return ParameterizedQobj for settings."""
        single_shot = shots == 1
        pcirc1, param1 = snapshot_expval_circuit_parameterized(single_shot=single_shot,
                                                               measure=measure,
                                                               snapshot=snapshot)
        circuits2to4 = snapshot_expval_circuits(pauli=True,
                                                skip_measure=(not measure),
                                                single_shot=single_shot)
        pcirc2, param2 = snapshot_expval_circuit_parameterized(single_shot=single_shot,
                                                               measure=measure,
                                                               snapshot=snapshot)
        circuits = [pcirc1] + circuits2to4 + [pcirc2]
        params = [param1, [], [], [], param2]
        return circuits, params

    def split_compare(self, circs, parameterizations=None, **shared_assemble_args):
        qobj = assemble(circs, parameterizations=parameterizations, qobj_id='testing', **shared_assemble_args)
        if parameterizations:
            qobjs = [assemble(c, parameterizations=[p], qobj_id='testing', **shared_assemble_args) for (c,p) in zip(circs, parameterizations)]
        else:
            qobjs = [assemble(c, qobj_id='testing', **shared_assemble_args) for c in circs]

        test_qobjs = split(qobj, _id='testing')
        self.assertEqual(len(test_qobjs), len(qobjs))
        for ref, test in zip(qobjs, test_qobjs):
            self.assertEqual(ref, test)

    def test_split(self):
        backend = Aer.get_backend('qasm_simulator')
        circs = [random_circuit(num_qubits=3, depth=4, measure=True) for _ in range(2)]
        circs = transpile(circs, backend)
        backend_options = {'shots': 4000}
        self.split_compare(circs, backend=backend, **backend_options)

    def test_parameterized_split(self):
        backend = Aer.get_backend('qasm_simulator')
        backend_options = {'shots': 4000}
        noise_model = None
        circs, params = self.parameterized_circuits(shots=backend_options['shots'], measure=True, snapshot=True)
        self.split_compare(circs, parameterizations=params, backend=backend, **backend_options)

    def test_set_assemble_config(self):
        """Test snapshot label must be str"""
        backend_options = None
        noise_model = None
        shots=4000

        backend = ClusterBackend(Aer.get_backend('qasm_simulator'), self._test_executor)
        circs = [random_circuit(num_qubits=3, depth=4, measure=True) for _ in range(2)]
        circs = transpile(circs, backend)
        
        combined_result = backend.set_assemble_config(shots=shots).run(circs, backend_options, noise_model).results(raises=True).combine_results()
        self.assertTrue(combined_result.success)

    def test_cluster_dispatch(self):
        """Test snapshot label must be str"""
        backend_options = None
        noise_model = None
        shots=4000

        backend = ClusterBackend(Aer.get_backend('qasm_simulator'), self._test_executor, shots=shots)
        circuits = ref_pauli_noise.pauli_gate_error_circuits()
        noise_models = ref_pauli_noise.pauli_gate_error_noise_models()
        targets = ref_pauli_noise.pauli_gate_error_counts(shots)
 
        qobj = assemble(circuits, backend, shots=shots)
        combined_result_list = backend.run(circuits, backend_options, noise_models[0]).results(raises=True).combine_results()
        combined_result_qobj = backend.run(qobj, backend_options, noise_models[0]).results(raises=True).combine_results()
        self.assertSuccess(combined_result_list)
        self.assertSuccess(combined_result_qobj)
        self.compare_counts(combined_result_list, [circuits[0]], [targets[0]], delta=0.05*shots)
        self.compare_counts(combined_result_qobj, [circuits[0]], [targets[0]], delta=0.05*shots)

if __name__ == '__main__':
    unittest.main()
