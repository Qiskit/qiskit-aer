from qiskit.circuit.library import QFT

from benchmark2.simulator_benchmark import SimulatorBenchmarkSuite

DEFAULT_APPS = [
    'qft'
    ]

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

class BasicChangeCircuits(SimulatorBenchmarkSuite):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = DEFAULT_NOISE_MODELS):
        
        super().__init__('basic_change_circuits', 
                         apps, qubits=qubits, 
                         runtime_names=runtime_names, 
                         measures=measures, 
                         measure_counts=measure_counts, 
                         noise_model_names=noise_model_names)
        
    def qft(self, qubit, repetition):
        return QFT(qubit)
    
if __name__ == "__main__":
    BasicChangeCircuits().run_manual()