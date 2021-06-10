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

# pylint: disable=arguments-differ

import unittest
import logging
import concurrent.futures

from test.terra import common
from test.terra.reference import ref_pauli_noise
from test.terra.reference.ref_snapshot_expval import (
    snapshot_expval_circuits, snapshot_expval_circuit_parameterized)
from test.terra.backends.test_qasm_simulator_extended_stabilizer import (
    TestQasmExtendedStabilizerSimulator)
from test.terra.backends.statevector_simulator.statevector_gates import StatevectorGateTests
from test.terra.backends.unitary_simulator.unitary_gates import UnitaryGateTests
from qiskit import transpile, assemble
from qiskit import Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.backends.cluster.utils import split
from qiskit.circuit.random import random_circuit

DASK_TESTS = False

try:
    from dask.distributed import LocalCluster, Client
    DASK_TESTS = True
except ImportError:
    logging.warning('Dask not installed: skipping Dask tests')


def dump(obj):
    """object dump"""
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))
            print()


class CBFixture(common.QiskitAerTestCase):
    """Extension tests for Aerbackend with cluster backend"""
    @classmethod
    def setUpClass(cls):
        """Override me with an executor init."""
        cls._test_executor = None

    @classmethod
    def tearDownClass(cls):
        if cls._test_executor:
            cls._test_executor.shutdown()


class ThreadPoolFixture(CBFixture):
    """Setup of ThreadPool execution tests"""
    @classmethod
    def setUpClass(cls, sim_name):
        super(ThreadPoolFixture, cls).setUpClass()
        cls._test_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        simulator = Aer.get_backend(sim_name)
        simulator.set_options(executor=cls._test_executor)
        return simulator


class DaskFixture(CBFixture):
    """Setup of Dask execution tests"""
    @classmethod
    def setUpClass(cls, sim_name):
        super(DaskFixture, cls).setUpClass()
        if DASK_TESTS:
            cls._test_executor = Client(address=LocalCluster(n_workers=1, processes=True))
            simulator = Aer.get_backend(sim_name)
            simulator.set_options(executor=cls._test_executor)
            return simulator
        else:
            return None

    def setUp(self):
        super(DaskFixture, self).setUp()
        if not DASK_TESTS:
            self.skipTest('Dask not installed, skipping ClusterBackend-dask tests')
        else:
            return


class TestQasmDask(DaskFixture,
                   TestQasmExtendedStabilizerSimulator):
    """qasm simulator test with Dask"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestQasmDask, cls).setUpClass('qasm_simulator')


class TestQasmThread(ThreadPoolFixture,
                     TestQasmExtendedStabilizerSimulator):
    """qasm simulator test with threadpool"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestQasmThread, cls).setUpClass('qasm_simulator')


class TestStatevectorDask(DaskFixture,
                          StatevectorGateTests):
    """statevector simulator test with Dask"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestStatevectorDask, cls).setUpClass('statevector_simulator')


class TestStatevectorThread(ThreadPoolFixture,
                            StatevectorGateTests):
    """statevector simulator test with thread pool"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestStatevectorThread, cls).setUpClass('statevector_simulator')


class TestUnitaryDask(DaskFixture,
                      UnitaryGateTests):
    """unitary simulator test with Dask"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestUnitaryDask, cls).setUpClass('unitary_simulator')


class TestUnitaryThread(ThreadPoolFixture,
                        UnitaryGateTests):
    """unitary simulator test with thread pool"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestUnitaryThread, cls).setUpClass('unitary_simulator')


class TestClusterBackendUtils(ThreadPoolFixture):
    """Test cluster utils"""
    @classmethod
    def setUpClass(cls):
        cls.SIMULATOR = super(TestClusterBackendUtils, cls).setUpClass('qasm_simulator')

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

    def split_compare(self,
                      circs,
                      parameterizations=None,
                      **shared_assemble_args):
        """Qobj split test"""
        qobj = assemble(circs,
                        parameterizations=parameterizations,
                        qobj_id='testing',
                        **shared_assemble_args)
        if parameterizations:
            qobjs = [assemble(c, parameterizations=[p],
                              qobj_id='testing',
                              **shared_assemble_args) for (c, p) in zip(circs, parameterizations)]
        else:
            qobjs = [assemble(c, qobj_id='testing', **shared_assemble_args) for c in circs]

        test_qobjs = split(qobj, _id='testing')
        self.assertEqual(len(test_qobjs), len(qobjs))
        for ref, test in zip(qobjs, test_qobjs):
            self.assertEqual(ref, test)

    def test_split(self):
        """Circuits split test"""
        backend = Aer.get_backend('qasm_simulator')
        circs = [random_circuit(num_qubits=3, depth=4, measure=True) for _ in range(2)]
        circs = transpile(circs, backend)
        backend_options = {'shots': 4000}
        self.split_compare(circs, backend=backend, **backend_options)

    def test_parameterized_split(self):
        """Parameterized circuits split test"""
        backend = Aer.get_backend('qasm_simulator')
        backend_options = {'shots': 4000}
        circs, params = self.parameterized_circuits(shots=backend_options['shots'],
                                                    measure=True,
                                                    snapshot=True)
        self.split_compare(circs, parameterizations=params, backend=backend, **backend_options)

    def test_pauli_gate_noise(self):
        """Test simulation with Pauli gate error noise model."""

        sim = QasmSimulator()
        backend_opts = {}
        shots = 1000
        circuits = ref_pauli_noise.pauli_gate_error_circuits()
        noise_models = ref_pauli_noise.pauli_gate_error_noise_models()
        targets = ref_pauli_noise.pauli_gate_error_counts(shots)

        for circuit, noise_model, target in zip(circuits, noise_models,
                                                targets):
            qobj = assemble(circuit, self.SIMULATOR, shots=shots)
            result = sim.run(
                qobj,
                noise_model=noise_model, **backend_opts).result()
            self.assertSuccess(result)
            self.compare_counts(result, [circuit], [target], delta=0.05 * shots)

    def test_cluster_dispatch(self):
        """Test snapshot label must be str"""
        backend_options = {'shots': 4000}
        shots = 4000

        backend = Aer.get_backend('qasm_simulator')
        backend.set_options(executor=self._test_executor)
        circuits = ref_pauli_noise.pauli_gate_error_circuits()
        noise_models = ref_pauli_noise.pauli_gate_error_noise_models()
        targets = ref_pauli_noise.pauli_gate_error_counts(shots)

        qobj = assemble(circuits, backend, shots=shots)
        combined_result_list = backend.run(circuits,
                                           noise_model=noise_models[0],
                                           **backend_options).result()
        combined_result_qobj = backend.run(qobj,
                                           noise_model=noise_models[0],
                                           **backend_options).result()
        self.assertSuccess(combined_result_list)
        self.assertSuccess(combined_result_qobj)
        self.compare_counts(combined_result_list,
                            [circuits[0]],
                            [targets[0]],
                            delta=0.05 * shots)
        self.compare_counts(combined_result_qobj,
                            [circuits[0]],
                            [targets[0]],
                            delta=0.05 * shots)


if __name__ == '__main__':
    unittest.main()
