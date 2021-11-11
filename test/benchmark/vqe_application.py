# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Base class of Qiskit Aer Benchmarking
"""
import sys
import numpy as np
import multiprocessing
from time import time

class UCCSDBenchmarkSuite:

    def __init__(self,
                 name = 'uccsd_benchmark'):

        self.mol_strings = {
            'H2': ('H .0 .0 .0; H .0 .0 0.735', 2),                    # qubits: 2
            'LiH': ('H .0 .0 .0; Li .0 .0 2.5', 10),                   # qubits: 10
            'HF': ('H .0 .0 .0; F .0 .0 1.25', 10),                    # qubits: 10
            }
        
        self.timeout = 60 * 60
        self.__name__ = name
        self.params = ([mol_name for mol_name in self.mol_strings])
        self.param_names = ["mol"]

    def _run_uccsd_vqe(self, mol_string, method, simulator, threads):
        
        from qiskit_nature.circuit.library import HartreeFock, UCCSD
        from qiskit_nature.converters.second_quantization import QubitConverter
        from qiskit_nature.drivers import UnitsType, Molecule
        from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
        from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
        from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
        
        driver = PySCFDriver(atom=mol_string, unit=UnitsType.ANGSTROM, basis='sto3g')
        es_problem = ElectronicStructureProblem(driver)
        qubit_converter = QubitConverter(JordanWignerMapper())
        max_evals_grouped = 1024
        
        from qiskit.algorithms.optimizers import SLSQP
        optimizer = SLSQP(maxiter=5000)
        
        from qiskit.utils import QuantumInstance
        
        quantum_instance = QuantumInstance(backend=simulator)
        
        # import logging
        # logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
        #                     level=logging.DEBUG,
        #                     datefmt='%Y-%m-%d %H:%M:%S')
        
        from qiskit_nature.algorithms import VQEUCCFactory
        
        vqe_solver = VQEUCCFactory(quantum_instance,
                                   optimizer=optimizer,
                                   include_custom=True,
                                   max_evals_grouped=max_evals_grouped)
        
        from qiskit_nature.algorithms import GroundStateEigensolver
        
        calc = GroundStateEigensolver(qubit_converter, vqe_solver)
        
        res = calc.solve(es_problem)
        
    def time_statevector(self, mol_name):
        from qiskit.providers.aer import AerSimulator
        threads = multiprocessing.cpu_count()
        mol_string = self.mol_strings[mol_name][0]
        qubit = self.mol_strings[mol_name][1]
        self._run_uccsd_vqe(mol_string, 'statevector_cpu', AerSimulator(device='CPU'), threads)

    def time_statevector_gpu(self, mol_name):
        from qiskit.providers.aer import AerSimulator
        threads = 1
        mol_string = self.mol_strings[mol_name][0]
        qubit = self.mol_strings[mol_name][1]
        self._run_uccsd_vqe(mol_string, 'statevector_gpu', AerSimulator(device='GPU'), threads)

    def run_manual(self):
        import timeout_decorator
        @timeout_decorator.timeout(self.timeout)
        def run_with_timeout (suite, method, molstring):
            start = time()
            eval(f'suite.time_{method}("{molstring}")')
            return time() - start
        
        #for runtime in self.runtime_names:
        for method in [ 'statevector' ]:
            for mol_string in self.mol_strings:
                print (f'{self.__name__},uccsd,{method},{mol_string},', end="")
                try:
                    elapsed = run_with_timeout(self, method, mol_string)
                    print ('{0}'.format(elapsed))
                except ValueError as e:
                    print ('{0}'.format(e))
                except:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    print ('unknown error')

if __name__ == "__main__":
    benrhmarks = [ UCCSDBenchmarkSuite() ]
    for benrhmark in benrhmarks:
        benrhmark.run_manual()
