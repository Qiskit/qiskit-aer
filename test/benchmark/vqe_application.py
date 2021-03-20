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

    def _run_uccsd_vqe(self, mol_string, method, threads):
        from qiskit import Aer
        from qiskit.algorithms import VQE
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit_nature.circuit.library import HartreeFock
        from qiskit_nature.components.variational_forms import UCCSD
        from qiskit_nature.drivers import PySCFDriver, UnitsType
        from qiskit_nature.mappers.second_quantization import ParityMapper
        from qiskit_nature.operators.second_quantization.qubit_converter import QubitConverter
        from qiskit_nature.transformations import (FermionicTransformation,
                                                   FermionicTransformationType,
                                                   FermionicQubitMappingType)
        
        driver = PySCFDriver(atom=mol_string,
                                  unit=UnitsType.ANGSTROM,
                                  charge=0,
                                  spin=0,
                                  basis='sto3g')
        qubit_converter = QubitConverter(mappers=ParityMapper())
        fermionic_transformation = \
            FermionicTransformation(transformation=FermionicTransformationType.FULL,
                                    qubit_mapping=FermionicQubitMappingType.PARITY,
                                    two_qubit_reduction=True,
                                    freeze_core=True,
                                    orbital_reduction=[])
 
        qubit_op, _ = fermionic_transformation.transform(driver)
             
        optimizer = SLSQP(maxiter=5000)
         
        num_spin_orbitals = fermionic_transformation.molecule_info['num_orbitals']
        num_particles = fermionic_transformation.molecule_info['num_particles']
        z2_symmetries = fermionic_transformation.molecule_info['z2_symmetries']
         
        init_state = HartreeFock(
            num_spin_orbitals=num_spin_orbitals,
            num_particles=num_particles,
            qubit_converter=qubit_converter)
        var_form = UCCSD(
            num_orbitals=num_spin_orbitals,
            num_particles=num_particles,
            initial_state=initial_state,
            qubit_mapping=fermionic_transformation._qubit_mapping,
            two_qubit_reduction=fermionic_transformation._two_qubit_reduction,
            max_evals_grouped=256,
            z2_symmetries=z2_symmetries)
         
        quantum_instance = QuantumInstance(
                         backend=Aer.get_backend('statevector_simulator'))
        quantum_instance.backend_options['backend_options'] = {'max_parallel_experiments':threads, 'method': method}
         
        solver = VQE(var_form=var_form,
                     optimizer=optimizer,
                     quantum_instance=quantum_instance)
         
        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)
         
        result = gsc.solve(self.driver)

            
    def _time_statevector(self, mol_name):
        threads = multiprocessing.cpu_count()
        mol_string = self.mol_strings[mol_name][0]
        qubit = self.mol_strings[mol_name][1]
        self._run_uccsd_vqe(mol_string, 'statevector', threads)

    def _time_statevector_gpu(self, mol_name):
        threads = 1
        mol_string = self.mol_strings[mol_name][0]
        qubit = self.mol_strings[mol_name][1]
        self._run_uccsd_vqe(mol_string, 'statevector_gpu', threads)

    def run_manual(self):
        import timeout_decorator
        @timeout_decorator.timeout(self.timeout)
        def run_with_timeout (suite, method, molstring):
            start = time()
            return eval(f'suite._time_{method}("{molstring}")')
        
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
