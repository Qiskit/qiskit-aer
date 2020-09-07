from qiskit.circuit.library import IntegerComparator, WeightedAdder, QuadraticForm

from benchmark2.simulator_benchmark import SimulatorBenchmarkSuite

DEFAULT_APPS = {
    'integer_comparator': 10,
    'weighted_adder': 1,
    'quadratic_form': 10
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

class ArithmeticCircuits(SimulatorBenchmarkSuite):

    def __init__(self,
                 apps = DEFAULT_APPS,
                 qubits = DEFAULT_QUBITS,
                 runtime_names = DEFAULT_RUNTIME,
                 measures = DEFAULT_MEASUREMENT_METHODS,
                 measure_counts = DEFAULT_MEASUREMENT_COUNTS,
                 noise_model_names = DEFAULT_NOISE_MODELS):
        super().__init__('arithmetic_circuits',
                          apps, 
                          qubits=qubits, 
                          runtime_names=runtime_names, 
                          measures=measures, 
                          measure_counts=measure_counts, 
                          noise_model_names=noise_model_names)
        
    def integer_comparator(self, qubit, repetition):
        if qubit < 2:
            raise ValueError('qubit is too small: {0}'.format(qubit))
        half = int(qubit / 2)
        circ = IntegerComparator(num_state_qubits=half, value=1)
        if repetition is not None:
            circ = circ.repeat(repetition).decompose()
        return circ
    
    def weighted_adder(self, qubit, repetition):
        if qubit > 20:
            raise ValueError('qubit is too big: {0}'.format(qubit))
        circ = WeightedAdder(num_state_qubits=qubit).decompose()
        if repetition is not None:
            circ = circ.repeat(repetition).decompose()
        return circ
    
    def quadratic_form(self, qubit, repetition):
        if qubit < 6:
            raise ValueError('qubit is too small: {0}'.format(qubit))
        circ = QuadraticForm(num_result_qubits=(qubit - 6), linear=[1, 1, 1], little_endian=True).decompose()
        if repetition is not None:
            circ = circ.repeat(repetition).decompose()
        return circ
    
if __name__ == "__main__":
    ArithmeticCircuits().run_manual()