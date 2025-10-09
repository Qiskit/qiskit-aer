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
import subprocess
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
    
    # For very large circuits (>35 qubits) on GPU, enable blocking
    run_options = {'shots': shots}
    if device == 'GPU' and num_qubits > 35:
        run_options['blocking_enable'] = True
        run_options['blocking_qubits'] = 27  # Optimal for most GPUs
    
    # Run simulation and measure time
    start_time = time.time()
    result = sim.run(transpiled, **run_options).result()
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    return execution_time, result


def get_gpu_memory():
    """
    Get GPU memory in GB using rocm-smi.
    
    Returns:
        float: GPU memory in GB, or None if detection fails
    """
    try:
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse output to find VRAM size
            lines = result.stdout.split('\n')
            for line in lines:
                if 'VRAM Total Memory' in line or 'Total' in line:
                    # Extract number (in MB typically)
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').isdigit():
                            mem_mb = float(part)
                            # Check if next part indicates units
                            if i + 1 < len(parts):
                                unit = parts[i + 1].lower()
                                if 'gb' in unit:
                                    return mem_mb
                                elif 'mb' in unit:
                                    return mem_mb / 1024
                            # Assume MB if > 1000, otherwise GB
                            return mem_mb / 1024 if mem_mb > 1000 else mem_mb
    except Exception as e:
        print(f"Warning: Could not detect GPU memory: {e}")
    
    return None


def calculate_max_qubits(gpu_memory_gb):
    """
    Calculate maximum qubits based on GPU memory.
    Formula: memory_needed = 2^n * 16 bytes (complex128)
    Use 80% of GPU memory as safe limit.
    
    Args:
        gpu_memory_gb (float): GPU memory in GB
        
    Returns:
        int: Maximum number of qubits
    """
    if gpu_memory_gb is None:
        return 28  # Conservative default
    
    # Use 80% of GPU memory for safety
    usable_memory_bytes = gpu_memory_gb * 0.8 * (1024 ** 3)
    
    # 2^n * 16 = usable_memory_bytes
    # n = log2(usable_memory_bytes / 16)
    max_qubits = int(np.log2(usable_memory_bytes / 16))
    
    # Cap at 45 as requested, minimum 25
    return min(max(max_qubits, 25), 45)


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
            
            # Try to get GPU memory
            gpu_memory = get_gpu_memory()
            if gpu_memory:
                print(f"✓ GPU memory detected: {gpu_memory:.1f} GB")
                max_qubits = calculate_max_qubits(gpu_memory)
                print(f"✓ Estimated max qubits: {max_qubits}")
            else:
                print("⚠ Could not detect GPU memory, using conservative limits")
                max_qubits = 28
            
            return True, max_qubits
        except Exception as e:
            print(f"✗ GPU initialization failed: {e}")
            return False, None
    else:
        print("✗ No GPU detected. Make sure ROCm is properly installed.")
        return False, None
    
    print()
    return True, None


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
            # For circuits > 30 qubits, skip CPU benchmark (would take too long)
            if num_qubits > 30:
                print(f"{num_qubits:<10} {'Skipped':<15} ", end='', flush=True)
                
                # GPU benchmark only
                gpu_time, gpu_result = run_benchmark('GPU', num_qubits, shots)
                
                print(f"{gpu_time:<15.4f} {'GPU only':<15}")
                
                results.append({
                    'qubits': num_qubits,
                    'cpu_time': None,
                    'gpu_time': gpu_time,
                    'speedup': None
                })
            else:
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
            
        except MemoryError as e:
            print(f"{num_qubits:<10} Memory Error - Too large for available memory")
            print(f"             Stopping benchmark at {num_qubits} qubits")
            break
        except Exception as e:
            print(f"{num_qubits:<10} Error: {str(e)[:50]}")
    
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


def memory_usage_comparison(max_qubits=45):
    """
    Compare memory requirements for different qubit counts.
    
    Args:
        max_qubits (int): Maximum number of qubits to show
    """
    print("=" * 70)
    print("Memory Requirements")
    print("=" * 70)
    print()
    print(f"{'Qubits':<10} {'Statevector Size':<25} {'Memory (Complex128)':<20}")
    print("-" * 70)
    
    # Show progression from 10 to max_qubits
    qubit_counts = [10, 15, 20, 25, 28, 30, 32, 35, 38, 40, 42, 45]
    qubit_counts = [q for q in qubit_counts if q <= max_qubits]
    
    for num_qubits in qubit_counts:
        statevector_size = 2 ** num_qubits
        memory_bytes = statevector_size * 16  # complex128 = 16 bytes
        memory_mb = memory_bytes / (1024 ** 2)
        memory_gb = memory_bytes / (1024 ** 3)
        memory_tb = memory_bytes / (1024 ** 4)
        
        if memory_tb >= 1:
            memory_str = f"{memory_tb:.2f} TB"
        elif memory_gb >= 1:
            memory_str = f"{memory_gb:.2f} GB"
        else:
            memory_str = f"{memory_mb:.2f} MB"
        
        print(f"{num_qubits:<10} {statevector_size:<25,} {memory_str:<20}")
    
    print("-" * 70)
    print()
    print("Note: GPU memory requirements also include intermediate storage")
    print("      for gate operations and measurement sampling.")
    print("      Actual memory usage may be 1.2-1.5x the statevector size.")
    print()


def main():
    """Main execution function."""
    # Print device information
    gpu_available, max_qubits = print_device_info()
    if not gpu_available:
        print("Exiting due to GPU unavailability.")
        return
    
    print()
    
    # Run memory comparison
    memory_usage_comparison(max_qubits if max_qubits else 45)
    
    # Run benchmark comparison
    print("=" * 70)
    print("Performance Benchmarks")
    print("=" * 70)
    print()
    
    # Generate qubit list from 10 to max_qubits
    if max_qubits and max_qubits >= 30:
        # For large memory GPUs, test more points
        num_qubits_list = [10, 15, 20, 25, 28, 30, 32, 35]
        # Add remaining points up to max_qubits
        current = 38
        while current <= max_qubits:
            num_qubits_list.append(current)
            current += 2
    elif max_qubits:
        # For smaller memory, test conservatively
        num_qubits_list = [10, 15, 20, 25]
        if max_qubits >= 28:
            num_qubits_list.append(28)
        if max_qubits >= 30:
            num_qubits_list.append(30)
    else:
        # Default conservative list
        num_qubits_list = [10, 15, 20, 25]
    
    print(f"Testing qubit counts: {num_qubits_list}")
    print(f"Maximum qubits to test: {max(num_qubits_list)}")
    print()
    
    results = run_comparison(num_qubits_list, shots=1024)
    
    # Show detailed example for medium-sized circuit
    detailed_comparison_example()
    
    # Summary
    if results:
        # Filter results that have speedup data
        speedup_results = [r for r in results if r['speedup'] is not None]
        
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Qubits tested: {min(r['qubits'] for r in results)} to {max(r['qubits'] for r in results)}")
        
        if speedup_results:
            avg_speedup = np.mean([r['speedup'] for r in speedup_results])
            max_speedup = max(r['speedup'] for r in speedup_results)
            best_qubit = [r['qubits'] for r in speedup_results if r['speedup'] == max_speedup][0]
            
            print(f"Average GPU speedup: {avg_speedup:.2f}x (for {len(speedup_results)} circuits with CPU comparison)")
            print(f"Best speedup: {max_speedup:.2f}x (at {best_qubit} qubits)")
        
        # Show GPU-only results
        gpu_only = [r for r in results if r['speedup'] is None]
        if gpu_only:
            print(f"\nGPU-only benchmarks (CPU too slow): {len(gpu_only)} circuits")
            print(f"  Largest circuit: {max(r['qubits'] for r in gpu_only)} qubits")
            fastest_gpu = min(r['gpu_time'] for r in gpu_only)
            print(f"  Fastest GPU time: {fastest_gpu:.4f} seconds")
        
        print()
        print("✓ Benchmark complete!")
        print()
        print("Tips for optimal GPU performance:")
        print("  - Use larger circuits (>20 qubits) for best GPU utilization")
        print("  - Enable blocking for very large circuits (>35 qubits):")
        print("    sim.run(circuit, blocking_enable=True, blocking_qubits=27)")
        
        if max_qubits and max_qubits >= 40:
            print(f"  - Your GPU can handle up to ~{max_qubits} qubits!")
        elif max_qubits and max_qubits >= 30:
            print(f"  - Your GPU supports up to ~{max_qubits} qubits")
        
        print()


if __name__ == "__main__":
    main()
