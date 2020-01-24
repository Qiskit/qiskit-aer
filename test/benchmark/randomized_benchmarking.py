# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member,invalid-name,missing-docstring,no-name-in-module
# pylint: disable=attribute-defined-outside-init,unsubscriptable-object

import numpy as np

from qiskit import assemble
from qiskit import transpile
from qiskit import Aer
import qiskit.ignis.verification.randomized_benchmarking as rb

from .tools import kraus_noise_model, no_noise, mixed_unitary_noise_model, \
                   reset_noise_model


def build_rb_circuit(nseeds=1, length_vector=None,
                     rb_pattern=None, length_multiplier=1,
                     seed_offset=0, align_cliffs=False, seed=None):
    """
    Randomized Benchmarking sequences.
    """
    if not seed:
        np.random.seed(10)
    else:
        np.random.seed(seed)
    rb_opts = {}
    rb_opts['nseeds'] = nseeds
    rb_opts['length_vector'] = length_vector
    rb_opts['rb_pattern'] = rb_pattern
    rb_opts['length_multiplier'] = length_multiplier
    rb_opts['seed_offset'] = seed_offset
    rb_opts['align_cliffs'] = align_cliffs

    # Generate the sequences
    try:
        rb_circs, _ = rb.randomized_benchmarking_seq(**rb_opts)
    except OSError:
        skip_msg = ('Skipping tests because '
                    'tables are missing')
        raise NotImplementedError(skip_msg)
    all_circuits = []
    for seq in rb_circs:
        all_circuits += seq
    return all_circuits


class RandomizedBenchmarkingQasmSimBenchmark:
    # parameters for RB (1&2 qubits):
    params = ([[[0]], [[0, 1]], [[0, 2], [1]]],
              ['statevector', 'density_matrix', 'stabilizer',
               'extended_stabilizer', 'matrix_product_state'],
              [no_noise(), mixed_unitary_noise_model(), reset_noise_model(),
               kraus_noise_model()])
    param_names = ['rb_pattern', 'simulator_method', 'noise_model']
    version = '0.2.0'
    timeout = 600

    def setup(self, rb_pattern, _, __):
        length_vector = np.arange(1, 200, 4)
        nseeds = 1
        self.seed = 10
        self.circuits = build_rb_circuit(nseeds=nseeds,
                                         length_vector=length_vector,
                                         rb_pattern=rb_pattern,
                                         seed=self.seed)
        self.sim_backend = Aer.get_backend('qasm_simulator')
        trans_circ = transpile(self.circuits, backend=self.sim_backend,
                               seed_transpiler=self.seed)
        self.qobj = assemble(trans_circ, backend=self.sim_backend)

    def time_run_rb_circuit(self, _, simulator_method, noise_model):
        backend_options = {
            'method': simulator_method,
            'noise_model': noise_model(),
        }
        job = self.sim_backend.run(self.qobj,
                                   backend_options=backend_options)
        job.result()

    def peakmem_run_rb_circuit(self, _, simulator_method, noise_model):
        backend_options = {
            'method': simulator_method,
            'noise_model': noise_model(),
        }
        job = self.sim_backend.run(self.qobj,
                                   backend_options=backend_options)
        job.result()
