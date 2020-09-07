from benchmark2.simulator_benchmark import SimulatorBenchmarkSuite
from benchmark2.particular_quantum_circuits import ParticularQuantumCircuits

DEFAULT_APPS = [
#    'fourier_checking',
#    'graph_state',
#    'hidden_linear_function',
#    'iqp',
    'quantum_volume'
#    'phase_estimation'
    ]

DEFAULT_QUBITS = [10, 11, 12, 13, 14]

DEFAULT_RUNTIME = [
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_CPU,
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_GPU,
    SimulatorBenchmarkSuite.RUNTIME_DENSITY_MATRIX_CPU,
    SimulatorBenchmarkSuite.RUNTIME_DENSITY_MATRIX_GPU
    ]

DEFAULT_MEASUREMENT_METHODS = [
    SimulatorBenchmarkSuite.MEASUREMENT_SAMPLING
    ]

DEFAULT_MEASUREMENT_COUNTS = [ 1, 10, 100, 1000, 10000 ]

DEFAULT_NOISE_MODELS = [
    SimulatorBenchmarkSuite.NOISE_DAMPING,
    SimulatorBenchmarkSuite.NOISE_DEPOLARIZING
    ]

class NoiseParticularQuantumCircuits(ParticularQuantumCircuits):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = DEFAULT_NOISE_MODELS):
        super().__init__( apps, 
                          qubits=qubits, 
                          runtime_names=runtime_names, 
                          measures=measures, 
                          measure_counts=measure_counts, 
                          noise_model_names=noise_model_names)
        self.__name__ = 'noise_particular_quantum_circuits'
        
    
if __name__ == "__main__":
    NoiseParticularQuantumCircuits().run_manual()