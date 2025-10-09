#!/usr/bin/env python3
"""
ROCm GPU Benchmark for Qiskit Aer

This script demonstrates the performance difference between CPU and GPU
execution using AMD ROCm acceleration.

Requirements:
- qiskit-aer-gpu-rocm installed
- AMD GPU with ROCm support

Usage:
    python3 rocm_gpu_benchmark.py
"""

import time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def create_benchmark_circuit(num_qubits):
    """
    Create a quantum circuit for benchmarking.
    Uses a combination of gates that benefit from GPU acceleration.
    
    Args:
        num_qubits (int): Number of qubits in the circuit
        
    Returns:
        QuantumCircuit: The benchmark circuit
    """
    qc = QuantumCircuit(num_qubits)
    
    # Create entanglement
    for i in range(num_qubits):
        qc.h(i)
    
    # Add layers of CNOT gates
    for layer in range(5):
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        
        # Add rotation gates
        for i in range(num_qubits):
            qc.rz(np.pi / 4, i)
            qc.rx(np.pi / 3, i)
    
    # Final layer of CNOTs
    for i in range(0, num_qubits - 1, 2):
        qc.cx(i, i + 1)
    
    # Measurements
    qc.measure_all()
    
    return qc


def run_benchmark(device, num_qubits, shots=1024, method='statevector'):
    """
    Run a benchmark on specified device.
    
    Args:
        device (str): 'CPU' or 'GPU'
        num_qubits (int): Number of qubits
        shots (int): Number of measurement shots
        method (str): Simulation method
        
    Returns:
        tuple: (execution_time, result)
    """
    # Create simulator
    sim = AerSimulator(device=device, method=method)
    
    # Create and transpile circuit
    circuit = create_benchmark_circuit(num_qubits)
    transpiled = transpile(circuit, sim)
    
    # Run simulation and measure time
    start_time = time.time()
    result = sim.run(transpiled, shots=shots).result()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return execution_time, result


def print_device_info():
    """Print information about available devices."""
    print("=" * 70)
    print("Qiskit Aer - AMD ROCm GPU Benchmark")
    print("=" * 70)
    print()
    
    # Check available devices
    sim = AerSimulator()
    devices = sim.available_devices()
    
    print(f"Available devices: {devices}")
    print()
    
    # Get GPU info if available
    if 'GPU' in devices:
        try:
            gpu_sim = AerSimulator(device='GPU')
            print("✓ GPU acceleration is available!")
        except Exception as e:
            print(f"✗ GPU initialization failed: {e}")
            return False
    else:
        print("✗ No GPU detected. Make sure ROCm is properly installed.")
        return False
    
    print()
    return True


def run_comparison(num_qubits_list=[10, 15, 20, 25], shots=1024):
    """
    Run comparison between CPU and GPU for different qubit counts.
    
    Args:
        num_qubits_list (list): List of qubit counts to test
        shots (int): Number of shots per simulation
    """
    print(f"Running benchmarks with {shots} shots per circuit...")
    print()
    print("-" * 70)
    print(f"{'Qubits':<10} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<15}")
    print("-" * 70)
    
    results = []
    
    for num_qubits in num_qubits_list:
        try:
            # CPU benchmark
            cpu_time, cpu_result = run_benchmark('CPU', num_qubits, shots)
            
            # GPU benchmark
            gpu_time, gpu_result = run_benchmark('GPU', num_qubits, shots)
            
            # Calculate speedup
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            
            print(f"{num_qubits:<10} {cpu_time:<15.4f} {gpu_time:<15.4f} {speedup:<15.2f}x")
            
            results.append({
                'qubits': num_qubits,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            })
            
        except Exception as e:
            print(f"{num_qubits:<10} Error: {str(e)}")
    
    print("-" * 70)
    print()
    
    return results


def detailed_comparison_example():
    """
    Run a detailed comparison showing circuit details and results.
    """
    print("\n" + "=" * 70)
    print("Detailed Example: 20-Qubit Circuit")
    print("=" * 70)
    print()
    
    num_qubits = 20
    shots = 2048
    
    # Create circuit
    circuit = create_benchmark_circuit(num_qubits)
    print(f"Circuit Properties:")
    print(f"  - Number of qubits: {circuit.num_qubits}")
    print(f"  - Circuit depth: {circuit.depth()}")
    print(f"  - Number of gates: {len(circuit.data)}")
    print(f"  - Shots: {shots}")
    print()
    
    # CPU execution
    print("Running on CPU...")
    cpu_time, cpu_result = run_benchmark('CPU', num_qubits, shots)
    cpu_counts = cpu_result.get_counts()
    print(f"  ✓ Completed in {cpu_time:.4f} seconds")
    print(f"  - Total outcomes: {len(cpu_counts)}")
    print(f"  - Top 3 outcomes:")
    for i, (state, count) in enumerate(sorted(cpu_counts.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:3]):
        print(f"    {i+1}. |{state}⟩: {count} ({100*count/shots:.1f}%)")
    print()
    
    # GPU execution
    print("Running on GPU...")
    gpu_time, gpu_result = run_benchmark('GPU', num_qubits, shots)
    gpu_counts = gpu_result.get_counts()
    print(f"  ✓ Completed in {gpu_time:.4f} seconds")
    print(f"  - Total outcomes: {len(gpu_counts)}")
    print(f"  - Top 3 outcomes:")
    for i, (state, count) in enumerate(sorted(gpu_counts.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:3]):
        print(f"    {i+1}. |{state}⟩: {count} ({100*count/shots:.1f}%)")
    print()
    
    # Performance summary
    speedup = cpu_time / gpu_time
    print(f"Performance Summary:")
    print(f"  - CPU Time: {cpu_time:.4f} seconds")
    print(f"  - GPU Time: {gpu_time:.4f} seconds")
    print(f"  - Speedup: {speedup:.2f}x")
    print(f"  - Time saved: {cpu_time - gpu_time:.4f} seconds ({(1-gpu_time/cpu_time)*100:.1f}%)")
    print()


def memory_usage_comparison():
    """
    Compare memory requirements for different qubit counts.
    """
    print("=" * 70)
    print("Memory Requirements")
    print("=" * 70)
    print()
    print(f"{'Qubits':<10} {'Statevector Size':<20} {'Memory (Complex128)':<20}")
    print("-" * 70)
    
    for num_qubits in [10, 15, 20, 25, 28, 30]:
        statevector_size = 2 ** num_qubits
        memory_bytes = statevector_size * 16  # complex128 = 16 bytes
        memory_mb = memory_bytes / (1024 ** 2)
        memory_gb = memory_bytes / (1024 ** 3)
        
        if memory_gb >= 1:
            memory_str = f"{memory_gb:.2f} GB"
        else:
            memory_str = f"{memory_mb:.2f} MB"
        
        print(f"{num_qubits:<10} {statevector_size:<20,} {memory_str:<20}")
    
    print("-" * 70)
    print()
    print("Note: GPU memory requirements also include intermediate storage")
    print("      for gate operations and measurement sampling.")
    print()


def main():
    """Main execution function."""
    # Print device information
    if not print_device_info():
        print("Exiting due to GPU unavailability.")
        return
    
    # Run memory comparison
    memory_usage_comparison()
    
    # Run benchmark comparison
    print("=" * 70)
    print("Performance Benchmarks")
    print("=" * 70)
    print()
    
    # Start with smaller circuits and increase
    num_qubits_list = [10, 15, 20, 25]
    results = run_comparison(num_qubits_list, shots=1024)
    
    # Show detailed example
    detailed_comparison_example()
    
    # Summary
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Average GPU speedup: {avg_speedup:.2f}x")
        print(f"Best speedup: {max(r['speedup'] for r in results):.2f}x "
              f"(at {[r['qubits'] for r in results if r['speedup'] == max(r['speedup'] for r in results)][0]} qubits)")
        print()
        print("✓ Benchmark complete!")
        print()
        print("Tips for optimal GPU performance:")
        print("  - Use larger circuits (>20 qubits) for best GPU utilization")
        print("  - Enable blocking for very large circuits:")
        print("    sim.run(circuit, blocking_enable=True, blocking_qubits=27)")
        print("  - For MI300 GPUs with 192GB HBM, can simulate up to ~28 qubits")
        print()


if __name__ == "__main__":
    main()
