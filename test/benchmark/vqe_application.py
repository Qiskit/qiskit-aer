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
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.applications import MolecularGroundStateEnergy
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit_aer import AerProvider


class UCCSDBenchmarkSuite:
    def __init__(self, name="uccsd_benchmark"):
        self.mol_strings = {
            "H2": ("H .0 .0 .0; H .0 .0 0.735", 2),  # qubits: 2
            "LiH": ("H .0 .0 .0; Li .0 .0 2.5", 10),  # qubits: 10
            "HF": ("H .0 .0 .0; F .0 .0 1.25", 10),  # qubits: 10
        }

        self.timeout = 60 * 60
        self.__name__ = name
        self.params = [mol_name for mol_name in self.mol_strings]
        self.param_names = ["mol"]

    def _run_uccsd_vqe(self, mol_string, method, threads):
        driver = PySCFDriver(
            atom=mol_string, unit=UnitsType.ANGSTROM, charge=0, spin=0, basis="sto3g"
        )

        def cb_create_solver(
            num_particles, num_orbitals, qubit_mapping, two_qubit_reduction, z2_symmetries
        ):
            initial_state = HartreeFock(
                num_orbitals,
                num_particles,
                qubit_mapping,
                two_qubit_reduction,
                z2_symmetries.sq_list,
            )
            var_form = UCCSD(
                num_orbitals=num_orbitals,
                num_particles=num_particles,
                initial_state=initial_state,
                qubit_mapping=qubit_mapping,
                two_qubit_reduction=two_qubit_reduction,
                z2_symmetries=z2_symmetries,
            )
            vqe = VQE(
                var_form=var_form,
                include_custom=True,
                optimizer=SLSQP(maxiter=5000),
                max_evals_grouped=256,
            )
            vqe.quantum_instance = AerProvider.get_backend("qasm_simulator")
            vqe.quantum_instance.backend_options["backend_options"] = {
                "max_parallel_experiments": threads,
                "method": method,
            }

            return vqe

        mgse = MolecularGroundStateEnergy(driver)
        result = mgse.compute_energy(cb_create_solver)

    def time_statevector(self, mol_name):
        threads = multiprocessing.cpu_count()
        mol_string = self.mol_strings[mol_name][0]
        qubit = self.mol_strings[mol_name][1]
        self._run_uccsd_vqe(mol_string, "statevector", threads)

    def time_statevector_gpu(self, mol_name):
        threads = 1
        mol_string = self.mol_strings[mol_name][0]
        qubit = self.mol_strings[mol_name][1]
        self._run_uccsd_vqe(mol_string, "statevector_gpu", threads)
