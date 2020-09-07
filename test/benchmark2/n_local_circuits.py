from qiskit.circuit.library import RealAmplitudes, EfficientSU2, ExcitationPreserving

from benchmark2.simulator_benchmark import SimulatorBenchmarkSuite

DEFAULT_APPS = {
    'real_amplitudes': 10,
    'real_amplitudes_linear': 10,
    'efficient_su2': 10,
    'efficient_su2_linear': 10,
    'excitation_preserving': 10,
    'excitation_preserving_linear': 10
    }

DEFAULT_QUBITS = SimulatorBenchmarkSuite.DEFAULT_QUBITS

DEFAULT_RUNTIME = [
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_CPU,
    SimulatorBenchmarkSuite.RUNTIME_MPS_CPU,
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_GPU
    ]

DEFAULT_MEASUREMENT_METHODS = [
    SimulatorBenchmarkSuite.MEASUREMENT_SAMPLING
    ]

DEFAULT_MEASUREMENT_COUNTS = SimulatorBenchmarkSuite.DEFAULT_MEASUREMENT_COUNTS

DEFAULT_NOISE_MODELS = [
    SimulatorBenchmarkSuite.NOISE_IDEAL
]

class NLocalCircuits(SimulatorBenchmarkSuite):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = DEFAULT_NOISE_MODELS):
        super().__init__('n_local_circuits', 
                         apps,
                         qubits=qubits,
                         runtime_names=runtime_names,
                         measures=measures, 
                         measure_counts=measure_counts, 
                         noise_model_names=noise_model_names)

    def real_amplitudes(self, qubit, repetition):
        return self.transpile(RealAmplitudes(qubit, reps=repetition))

    def real_amplitudes_linear(self, qubit, repetition):
        return self.transpile(RealAmplitudes(qubit, reps=repetition, entanglement='linear'))
    
    def efficient_su2(self, qubit, repetition):
        return self.transpile(EfficientSU2(qubit).decompose())

    def efficient_su2_linear(self, qubit, repetition):
        return self.transpile(EfficientSU2(qubit, reps=repetition, entanglement='linear'))
    
    def excitation_preserving(self, qubit, repetition):
        return self.transpile(ExcitationPreserving(qubit, reps=repetition).decompose())

    def excitation_preserving_linear(self, qubit, repetition):
        return self.transpile(ExcitationPreserving(qubit, reps=repetition, entanglement='linear'))
    
    
if __name__ == "__main__":
    NLocalCircuits().run_manual()