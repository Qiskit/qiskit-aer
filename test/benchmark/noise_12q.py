from benchmark.simulator_benchmark import SimulatorBenchmarkSuite

DEFAULT_APPS = {
#    'fourier_checking',
#    'graph_state',
#    'hidden_linear_function',
#    'iqp',
    'quantum_volume': 1
#    'phase_estimation'
    }

DEFAULT_QUBITS = [12]

DEFAULT_RUNTIME = [
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_CPU,
    SimulatorBenchmarkSuite.RUNTIME_STATEVECTOR_GPU,
    SimulatorBenchmarkSuite.RUNTIME_DENSITY_MATRIX_CPU,
    SimulatorBenchmarkSuite.RUNTIME_DENSITY_MATRIX_GPU
    ]

DEFAULT_MEASUREMENT_METHODS = [
    SimulatorBenchmarkSuite.MEASUREMENT_SAMPLING
    ]

DEFAULT_MEASUREMENT_COUNTS = [ 1000 ]

DEFAULT_NOISE_MODELS = [
    SimulatorBenchmarkSuite.NOISE_DAMPING,
    SimulatorBenchmarkSuite.NOISE_DEPOLARIZING
    ]

class DampingError(SimulatorBenchmarkSuite):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = [SimulatorBenchmarkSuite.NOISE_DAMPING] ):
        super().__init__('damping_error',
                         apps, 
                         qubits=qubits, 
                         runtime_names=runtime_names, 
                         measures=measures, 
                         measure_counts=measure_counts, 
                         noise_model_names=noise_model_names)
        self.__name__ = 'damping_error'
        
class DepolarizingError(SimulatorBenchmarkSuite):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = [SimulatorBenchmarkSuite.NOISE_DEPOLARIZING] ):
        super().__init__('depolarizing_error',
                         apps, 
                         qubits=qubits, 
                         runtime_names=runtime_names, 
                         measures=measures, 
                         measure_counts=measure_counts, 
                         noise_model_names=noise_model_names)
        self.__name__ = 'depolarizing_error'
    
if __name__ == "__main__":
    DampingError().run_manual()
    DepolarizingError().run_manual()