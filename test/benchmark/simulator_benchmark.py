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
from time import time
from qiskit.compiler import transpile, assemble
from qiskit_aer import AerSimulator, UnitarySimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

from benchmark.circuit_library_circuits import CircuitLibraryCircuits

QOBJS = {}
SIMULATOR = AerSimulator()


class SimulatorBenchmarkSuite(CircuitLibraryCircuits):
    RUNTIME_STATEVECTOR_CPU = "statevector"
    RUNTIME_STATEVECTOR_GPU = "statevector_gpu"
    RUNTIME_MPS_CPU = "matrix_product_state"
    RUNTIME_DENSITY_MATRIX_CPU = "density_matrix"
    RUNTIME_DENSITY_MATRIX_GPU = "density_matrix_gpu"
    RUNTIME_STABILIZER_CPU = "stabilizer"
    RUNTIME_EXTENDED_STABILIZER_CPU = "extended_stabilizer"
    RUNTIME_UNITARY_MATRIX_CPU = "unitary_matrix"
    RUNTIME_UNITARY_MATRIX_GPU = "unitary_matrix_gpu"

    RUNTIME_CPU = [
        RUNTIME_STATEVECTOR_CPU,
        RUNTIME_MPS_CPU,
        RUNTIME_DENSITY_MATRIX_CPU,
        RUNTIME_STABILIZER_CPU,
        RUNTIME_EXTENDED_STABILIZER_CPU,
        RUNTIME_UNITARY_MATRIX_CPU,
    ]

    RUNTIME_GPU = [RUNTIME_STATEVECTOR_GPU, RUNTIME_DENSITY_MATRIX_GPU, RUNTIME_UNITARY_MATRIX_GPU]

    DEFAULT_RUNTIME = [
        RUNTIME_STATEVECTOR_CPU,
        RUNTIME_MPS_CPU,
        RUNTIME_DENSITY_MATRIX_CPU,
        RUNTIME_STATEVECTOR_GPU,
    ]

    TRANSPLIERS = {RUNTIME_MPS_CPU: "self.transpile_for_mps"}

    DEFAULT_QUBITS = [10, 15, 20, 25]

    MEASUREMENT_SAMPLING = "sampling"
    MEASUREMENT_EXPVAL = "expval"

    DEFAULT_MEASUREMENT_METHODS = [MEASUREMENT_SAMPLING]
    DEFAULT_MEASUREMENT_COUNTS = [1000]

    NOISE_IDEAL = "ideal"
    NOISE_DAMPING = "damping"
    NOISE_DEPOLARIZING = "depolarizing"

    DEFAULT_NOISE_MODELS = [NOISE_IDEAL]

    def __init__(
        self,
        name="simulator_benchmark",
        apps={},
        qubits=DEFAULT_QUBITS,
        runtime_names=DEFAULT_RUNTIME,
        measures=DEFAULT_MEASUREMENT_METHODS,
        measure_counts=DEFAULT_MEASUREMENT_COUNTS,
        noise_model_names=DEFAULT_NOISE_MODELS,
    ):
        self.timeout = 60 * 10
        self.__name__ = name

        self.apps = apps if isinstance(apps, list) else [app for app in apps]
        self.app2rep = {} if isinstance(apps, list) else apps
        self.qubits = qubits
        self.runtime_names = runtime_names
        self.measures = measures
        self.measure_counts = measure_counts
        self.noise_model_names = noise_model_names

        self.params = (
            self.apps,
            self.measures,
            self.measure_counts,
            self.noise_model_names,
            self.qubits,
        )
        self.param_names = ["application", "measure_method", "measure_counts", "noise", "qubit"]

        all_simulators = [SIMULATOR]

        self.simulators = {}
        self.backend_options_list = {}
        self.backend_qubits = {}

        self.noise_models = {}
        self.noise_models[self.NOISE_IDEAL] = None
        if self.NOISE_DAMPING in self.noise_model_names:
            noise_model = NoiseModel()
            error = amplitude_damping_error(1e-3)
            noise_model.add_all_qubit_quantum_error(error, ["u3"])
            self.noise_models[self.NOISE_DAMPING] = noise_model
        if self.NOISE_DEPOLARIZING in self.noise_model_names:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ["u3"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cx"])
            self.noise_models[self.NOISE_DEPOLARIZING] = noise_model

        if self.RUNTIME_STATEVECTOR_CPU in runtime_names:
            self.simulators[self.RUNTIME_STATEVECTOR_CPU] = SIMULATOR
            self.backend_options_list[self.RUNTIME_STATEVECTOR_CPU] = {
                "method": self.RUNTIME_STATEVECTOR_CPU
            }
            self.backend_qubits[self.RUNTIME_STATEVECTOR_CPU] = self.qubits

        if self.RUNTIME_STATEVECTOR_GPU in runtime_names:
            self.simulators[self.RUNTIME_STATEVECTOR_GPU] = SIMULATOR
            self.backend_options_list[self.RUNTIME_STATEVECTOR_GPU] = {
                "method": self.RUNTIME_STATEVECTOR_GPU
            }
            self.backend_qubits[self.RUNTIME_STATEVECTOR_GPU] = self.qubits

        if self.RUNTIME_MPS_CPU in runtime_names:
            self.simulators[self.RUNTIME_MPS_CPU] = SIMULATOR
            self.backend_options_list[self.RUNTIME_MPS_CPU] = {"method": self.RUNTIME_MPS_CPU}
            self.backend_qubits[self.RUNTIME_MPS_CPU] = self.qubits

        if self.RUNTIME_DENSITY_MATRIX_CPU in runtime_names:
            self.simulators[self.RUNTIME_DENSITY_MATRIX_CPU] = SIMULATOR
            self.backend_options_list[self.RUNTIME_DENSITY_MATRIX_CPU] = {
                "method": self.RUNTIME_DENSITY_MATRIX_CPU
            }
            self.backend_qubits[self.RUNTIME_DENSITY_MATRIX_CPU] = [
                qubit for qubit in qubits if qubit <= 15
            ]

        if self.RUNTIME_DENSITY_MATRIX_GPU in runtime_names:
            self.simulators[self.RUNTIME_DENSITY_MATRIX_GPU] = SIMULATOR
            self.backend_options_list[self.RUNTIME_DENSITY_MATRIX_GPU] = {
                "method": self.RUNTIME_DENSITY_MATRIX_GPU
            }
            self.backend_qubits[self.RUNTIME_DENSITY_MATRIX_GPU] = [
                qubit for qubit in qubits if qubit <= 15
            ]

    def gen_qobj(self, runtime, app, measure, measure_count, qubit):
        def add_measure_all(base):
            circuit = base.copy()
            circuit.measure_all()
            return circuit

        def add_expval(base, num_terms):
            circuit = base.copy()
            from qiskit_aer.extensions import snapshot_expectation_value
            from numpy.random import default_rng

            rng = default_rng(1)
            paulis = ["".join(s) for s in rng.choice(["I", "X", "Y", "Z"], size=(num_terms, qubit))]
            pauli_op = [(1 / num_terms, pauli) for pauli in paulis]
            circuit.snapshot_expectation_value("expval", pauli_op, range(qubit))

        circuit = eval("self.{0}".format(app))(
            qubit, None if app not in self.app2rep else self.app2rep[app]
        )
        if len(circuit.parameters) > 0:
            param_binds = {}
            for param in circuit.parameters:
                param_binds[param] = np.random.random()
            circuit = circuit.assign_parameters(param_binds)

        simulator = self.simulators[runtime]
        if measure == self.MEASUREMENT_SAMPLING:
            if runtime in self.TRANSPLIERS:
                runtime_circuit = eval(self.TRANSPLIERS[runtime])(circuit)
                if (runtime, app, measure, measure_count, qubit) not in QOBJS:
                    QOBJS[(runtime, app, measure, measure_count, qubit)] = assemble(
                        runtime_circuit, simulator, shots=measure_count
                    )
                return QOBJS[(runtime, app, measure, measure_count, qubit)]
            else:
                runtime_circuit = circuit
                if (simulator, app, measure, measure_count, qubit) not in QOBJS:
                    QOBJS[(simulator, app, measure, measure_count, qubit)] = assemble(
                        runtime_circuit, simulator, shots=measure_count
                    )
                return QOBJS[(simulator, app, measure, measure_count, qubit)]
        elif measure == self.MEASUREMENT_EXPVAL:
            if runtime in self.TRANSPLIERS:
                runtime_circuit = eval(self.TRANSPLIERS[runtime])(circuit)
                if (runtime, app, measure, measure_count, qubit) not in QOBJS:
                    QOBJS[(runtime, app, measure, measure_count, qubit)] = assemble(
                        runtime_circuit, simulator, shots=1
                    )
                return QOBJS[(runtime, app, measure, measure_count, qubit)]
            else:
                runtime_circuit = circuit
                if (simulator, app, measure, measure_count, qubit) not in QOBJS:
                    QOBJS[(simulator, app, measure, measure_count, qubit)] = assemble(
                        runtime_circuit, simulator, shots=1
                    )
                return QOBJS[(simulator, app, measure, measure_count, qubit)]

    def _transpile(self, circuit, basis_gates):
        from qiskit import transpile

        return transpile(circuit, basis_gates=basis_gates)

    def transpile(self, circuit):
        return self._transpile(
            circuit,
            [
                "u1",
                "u2",
                "u3",
                "cx",
                "cz",
                "id",
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdg",
                "t",
                "tdg",
                "swap",
                "ccx",
                "unitary",
                "diagonal",
                "initialize",
                "cu1",
                "cu2",
                "cu3",
                "cswap",
                "mcx",
                "mcy",
                "mcz",
                "mcu1",
                "mcu2",
                "mcu3",
                "mcswap",
                "multiplexer",
                "kraus",
                "roerror",
            ],
        )

    def transpile_for_mps(self, circuit):
        return self._transpile(
            circuit,
            [
                "u1",
                "u2",
                "u3",
                "cx",
                "cz",
                "id",
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdg",
                "t",
                "tdg",
                "swap",
                "ccx"  # , 'unitary', 'diagonal', 'initialize',
                "cu1",  #'cu2', 'cu3', 'cswap', 'mcx', 'mcy', 'mcz',
                #'mcu1', 'mcu2', 'mcu3', 'mcswap', 'multiplexer', 'kraus',
                "roerror",
            ],
        )

    def _run(self, runtime, app, measure, measure_count, noise_name, qubit):
        if runtime not in self.simulators or runtime not in self.backend_options_list:
            raise ValueError("unknown runtime: {0}".format(runtime))
        simulator = self.simulators[runtime]
        backend_options = self.backend_options_list[runtime]
        noise_model = self.noise_models[noise_name]

        if qubit not in self.backend_qubits[runtime]:
            raise ValueError(
                "out of qubit range: qubit={0}, list={1}".format(
                    qubit, self.backend_qubits[runtime]
                )
            )

        qobj = self.gen_qobj(runtime, app, measure, measure_count, qubit)
        if qobj is None:
            raise ValueError(
                "no qobj: measure={0}:{1}, qubit={2}".format(measure, measure_count, qubit)
            )

        start = time()
        result = simulator.run(qobj, noise_model=noise_model, **backend_options).result()
        if result.status != "COMPLETED":
            try:
                reason = None
                ret_dict = result.to_dict()
                if "results" in ret_dict:
                    if len(ret_dict["results"]) > 0 and "status" in ret_dict["results"][0]:
                        reason = ret_dict["results"][0]["status"]
                if reason is None and "status" in ret_dict:
                    reason = ret_dict["status"]
                if reason is None:
                    reason = "unknown"
            except:
                reason = "unknown"
            raise ValueError("simulation error ({0})".format(reason))
        return time() - start

    def run_manual(self):
        import timeout_decorator

        @timeout_decorator.timeout(self.timeout)
        def run_with_timeout(suite, runtime, app, measure, measure_count, noise_name, qubit):
            start = time()
            return eval("suite.track_{0}".format(runtime))(
                app, measure, measure_count, noise_name, qubit
            )

        # for runtime in self.runtime_names:
        for noise_name in self.noise_model_names:
            for runtime in self.runtime_names:
                for app in self.apps:
                    repeats = None if app not in self.app2rep else self.app2rep[app]
                    app_name = app if repeats is None else "{0}:{1}".format(app, repeats)
                    for qubit in self.qubits:
                        for measure in self.measures:
                            for measure_count in self.measure_counts:
                                print(
                                    "{0},{1},{2},{3},{4},{5},{6},".format(
                                        self.__name__,
                                        app_name,
                                        runtime,
                                        measure,
                                        measure_count,
                                        noise_name,
                                        qubit,
                                    ),
                                    end="",
                                )
                                try:
                                    elapsed = run_with_timeout(
                                        self,
                                        runtime,
                                        app,
                                        measure,
                                        measure_count,
                                        noise_name,
                                        qubit,
                                    )
                                    print("{0}".format(elapsed))
                                except ValueError as e:
                                    print("{0}".format(e))
                                except:
                                    import traceback

                                    traceback.print_exc(file=sys.stderr)
                                    print("unknown error")
