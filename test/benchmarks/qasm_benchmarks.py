"""
Run quantum volume benchmarks on AerQvSimulator.
"""

from multiprocessing import cpu_count
from itertools import repeat
import matplotlib.pyplot as plt

from quantumvolume import quantum_volume_circuit
from benchmark_tools import benchmark_circuits_qasm_simulator
from benchmark_tools import mixed_unitary_noise_model
from benchmark_tools import reset_noise_model
from benchmark_tools import kraus_noise_model


def run_benchmarks(qubits_list, thread_list, depth,
                   shots=1, num_circ=1, circuit_seed=42,
                   test_parallel_shot=True,
                   test_parallel_circuit=True,
                   ideal_tests=True,
                   unitary_noise_tests=True,
                   reset_noise_tests=True,
                   kraus_noise_tests=True):
    """
    Run quantum volume circuit benchmarks.

    Runs benchmarks for ideal circuits and noisy circuits with mixed unitary,
    reset, and kraus noise models.

    Args:
        qubit_list (list[int]): list of qubits to generate circuits for.
        thread_list (list[int]): maximum thread counts for parallelization.
        depth (int): depth of circuit for each qubit count.
        shots (int): number of shots for benchmarks.
        num_circs (int): number of different circuits for each qubit number.
        circuit_seed (int): seed for random quantum volume circuit generation,
                            if None a random seed will be generated (default 42).
        test_parallel_shot (bool): run tests using parallel shot evaluation.
        test_parallel_circuit (bool): run tests using parallel circuit evaluation.
        ideal_tests (bool): run ideal circuit tests.
        unitary_noise_tests (bool): run test circuits with mixed unitary noise.
        reset_noise_tests (bool): run test circuits with rest noise.
        kraus_noise_tests (bool): run test circuits with kraus noise.

    Returns:
        dict: benchmark results.
    """

    # Quantum Volume benchmark circuits
    test_circuits = []
    for n in qubits_list:
        # Generate ciruits
        circuits_n = []
        for _ in repeat(None, num_circ):
            circuits_n.append(quantum_volume_circuit(n, depth, measure=True))
        test_circuits.append(circuits_n)

    # threads
    threads = []
    for n in thread_list:
        if n <= cpu_count():
            threads.append(n)
        else:
            print('Threads {}'.format(n) +
                  " is greater than cpu_count {}".format(cpu_count()))

    # Dicts indexed by core number

    # Ideal circuits
    times_ideal_par_state = {}
    times_ideal_par_circuit = {}

    # Mixed unitary noisy circuits
    unitary_noise = mixed_unitary_noise_model()
    times_unitary_par_state = {}
    times_unitary_par_shot = {}
    times_unitary_par_circuit = {}

    # Mixed reset noisy circuits
    reset_noise = reset_noise_model()
    times_reset_par_state = {}
    times_reset_par_shot = {}
    times_reset_par_circuit = {}

    # Kraus noisy circuits
    kraus_noise = kraus_noise_model()
    times_kraus_par_state = {}
    times_kraus_par_shot = {}
    times_kraus_par_circuit = {}

    # Loop over thread limits
    for th in threads:

        # Ideal circuits
        t_ideal_par_state = []
        t_ideal_par_circuit = []

        # Mixed unitary noisy circuits
        t_unitary_par_state = []
        t_unitary_par_shot = []
        t_unitary_par_circuit = []

        # Mixed reset noisy circuits
        t_reset_par_state = []
        t_reset_par_shot = []
        t_reset_par_circuit = []

        # Kraus noisy circuits
        t_kraus_par_state = []
        t_kraus_par_shot = []
        t_kraus_par_circuit = []

        # Loop over qubit number
        for circuits in test_circuits:

            # Ideal circuits
            if ideal_tests is True:
                t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                      parallel_mode='state')
                t_ideal_par_state.append(t)
                if test_parallel_circuit is True:
                    if th == 1:
                        t_ideal_par_circuit.append(t)
                    else:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              parallel_mode='circuit')
                        t_ideal_par_circuit.append(t)

            # Mixed unitary noise circuits
            if unitary_noise_tests is True:
                t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                      noise_model=unitary_noise,
                                                      parallel_mode='state')
                t_unitary_par_state.append(t)
                if th == 1:
                    if test_parallel_shot is True:
                        t_unitary_par_shot.append(t)
                    if test_parallel_circuit is True:
                        t_unitary_par_circuit.append(t)
                else:
                    if test_parallel_shot is True:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              noise_model=unitary_noise,
                                                              parallel_mode='shot')
                        t_unitary_par_shot.append(t)
                    if test_parallel_circuit is True:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              noise_model=unitary_noise,
                                                              parallel_mode='circuit')
                        t_unitary_par_circuit.append(t)

            # reset noise circuits
            if reset_noise_tests is True:
                t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                      noise_model=reset_noise,
                                                      parallel_mode='state')
                t_reset_par_state.append(t)
                if th == 1:
                    if test_parallel_shot is True:
                        t_reset_par_shot.append(t)
                    if test_parallel_circuit is True:
                        t_reset_par_circuit.append(t)
                else:
                    if test_parallel_shot is True:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              noise_model=reset_noise,
                                                              parallel_mode='shot')
                        t_reset_par_shot.append(t)
                    if test_parallel_circuit is True:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              noise_model=reset_noise,
                                                              parallel_mode='circuit')
                        t_reset_par_circuit.append(t)

            # Kraus noisy circuits
            if kraus_noise_tests is True:
                t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                      noise_model=kraus_noise,
                                                      parallel_mode='state')
                t_kraus_par_state.append(t)
                if th == 1:
                    if test_parallel_shot is True:
                        t_kraus_par_shot.append(t)
                    if test_parallel_circuit is True:
                        t_kraus_par_circuit.append(t)
                else:
                    if test_parallel_shot is True:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              noise_model=kraus_noise,
                                                              parallel_mode='shot')
                        t_kraus_par_shot.append(t)
                    if test_parallel_circuit is True:
                        t = benchmark_circuits_qasm_simulator(circuits, shots=shots, threads=th,
                                                              noise_model=kraus_noise,
                                                              parallel_mode='circuit')
                        t_kraus_par_circuit.append(t)

        # Add to dataset
        times_ideal_par_state[th] = t_ideal_par_state
        times_ideal_par_circuit[th] = t_ideal_par_circuit

        times_unitary_par_state[th] = t_unitary_par_state
        times_unitary_par_circuit[th] = t_unitary_par_circuit
        times_unitary_par_shot[th] = t_unitary_par_shot

        times_reset_par_state[th] = t_reset_par_state
        times_reset_par_circuit[th] = t_reset_par_circuit
        times_reset_par_shot[th] = t_reset_par_shot

        times_kraus_par_state[th] = t_kraus_par_state
        times_kraus_par_circuit[th] = t_kraus_par_circuit
        times_kraus_par_shot[th] = t_kraus_par_shot

    line_colors = ['navy', 'darkred', 'green', 'royalblue']

    # Figure (Ideal circuits)
    if ideal_tests is True:
        for j, num_threads in enumerate(threads):
            plt.semilogy(qubits_list, times_ideal_par_state[num_threads],
                         'o--', color=line_colors[j],
                         label='%d cores (par. default)' % num_threads)
            if test_parallel_circuit is True:
                plt.semilogy(qubits_list, times_ideal_par_circuit[num_threads],
                             's-.', color=line_colors[j],
                             label='%d cores (par. circuit)' % num_threads)
        plt.xlabel('No. of qubits')
        plt.ylabel('time (average)')
        plt.title("Quantum Volume (ideal):" +
                  " depth = {}".format(depth) +
                  ", shots = {}".format(shots) +
                  ", num_circ = {}".format(num_circ))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('qv_benchmark_ideal.pdf')
        plt.show()

    # Figure (Unitary noise circuits)
    if unitary_noise_tests is True:
        for j, num_threads in enumerate(threads):
            plt.semilogy(qubits_list, times_unitary_par_state[num_threads],
                         'o--', color=line_colors[j],
                         label='%d cores (par. state)' % num_threads)
            if test_parallel_circuit is True:
                plt.semilogy(qubits_list, times_unitary_par_circuit[num_threads],
                             's-.', color=line_colors[j],
                             label='%d cores (par. circuit)' % num_threads)
            if test_parallel_shot is True:
                plt.semilogy(qubits_list, times_unitary_par_shot[num_threads],
                             '^:', color=line_colors[j],
                             label='%d cores (par. shot)' % num_threads)
        plt.xlabel('No. of qubits')
        plt.ylabel('time (average)')
        plt.title("Quantum Volume (mixed unitary noise):" +
                  " depth = {}".format(depth) +
                  ", shots = {}".format(shots) +
                  ", num_circ = {}".format(num_circ))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('qv_benchmark_unitary.pdf')
        plt.show()

    # Figure (Reset noise circuits)
    if reset_noise_tests is True:
        for j, num_threads in enumerate(threads):
            plt.semilogy(qubits_list, times_reset_par_state[num_threads],
                         'o--', color=line_colors[j],
                         label='%d cores (par. state)' % num_threads)
            if test_parallel_circuit is True:
                plt.semilogy(qubits_list, times_reset_par_circuit[num_threads],
                             's-.', color=line_colors[j],
                             label='%d cores (par. circuit)' % num_threads)
            if test_parallel_shot is True:
                plt.semilogy(qubits_list, times_reset_par_shot[num_threads],
                             '^:', color=line_colors[j],
                             label='%d cores (par. shot)' % num_threads)
        plt.xlabel('No. of qubits')
        plt.ylabel('time (average)')
        plt.title("Quantum Volume (reset noise):" +
                  " depth = {}".format(depth) +
                  ", shots = {}".format(shots) +
                  ", num_circ = {}".format(num_circ))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('qv_benchmark_reset.pdf')
        plt.show()

    # Figure (Kraus noise circuits)
    if kraus_noise_tests is True:
        for j, num_threads in enumerate(threads):
            plt.semilogy(qubits_list, times_kraus_par_state[num_threads],
                         'o--', color=line_colors[j],
                         label='%d cores (par. state)' % num_threads)
            if test_parallel_circuit is True:
                plt.semilogy(qubits_list, times_kraus_par_circuit[num_threads],
                             's-.', color=line_colors[j],
                             label='%d cores (par. circuit)' % num_threads)
            if test_parallel_shot is True:
                plt.semilogy(qubits_list, times_kraus_par_shot[num_threads],
                             '^:', color=line_colors[j],
                             label='%d cores (par. shot)' % num_threads)
        plt.xlabel('No. of qubits')
        plt.ylabel('time (average)')
        plt.title("Quantum Volume (Kraus noise):" +
                  " depth = {}".format(depth) +
                  ", shots = {}".format(shots) +
                  ", num_circ = {}".format(num_circ))
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('qv_benchmark_kraus.pdf')
        plt.show()

    # Format return timing data
    output_data = {}
    if ideal_tests is True:
        output_data['t_ideal_par_state'] = times_ideal_par_state,
        if test_parallel_circuit is True:
            output_data['t_ideal_par_circuit'] = times_ideal_par_circuit,
    if unitary_noise_tests is True:
        output_data['times_unitary_unitary_par_state'] = times_unitary_par_state,
        if test_parallel_circuit is True:
            output_data['times_unitary_par_shot'] = times_unitary_par_shot,
        if test_parallel_circuit is True:
            output_data['times_unitary_par_circuit'] = times_unitary_par_circuit,
    if reset_noise_tests is True:
        output_data['times_reset_par_state'] = times_reset_par_state,
        if test_parallel_circuit is True:
            output_data['times_reset_par_shot'] = times_reset_par_shot,
        if test_parallel_circuit is True:
            output_data['times_reset_par_circuit'] = times_reset_par_circuit,
    if kraus_noise_tests is True:
        output_data['times_kraus_par_state'] = times_kraus_par_state,
        if test_parallel_circuit is True:
            output_data['times_kraus_par_shot'] = times_kraus_par_shot,
        if test_parallel_circuit is True:
            output_data['times_kraus_par_circuit'] = times_kraus_par_circuit
    return output_data
