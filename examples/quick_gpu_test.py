#!/usr/bin/env python3
"""
Quick GPU Test for Qiskit Aer

Tests GPU acceleration with both AMD ROCm and Nvidia CUDA.
Demonstrates the crossover point where GPU becomes faster than CPU.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time
import numpy as np


def create_gpu_friendly_circuit(num_qubits):
    """
    Create a circuit that benefits from GPU acceleration.
    Uses multiple layers of rotations and entanglement.
    """
    qc = QuantumCircuit(num_qubits)
    
    # Initial superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Multiple layers of operations (deeper circuit = better GPU utilization)
    num_layers = 8
    for layer in range(num_layers):
        # Rotation gates
        for i in range(num_qubits):
            qc.rx(np.pi / (layer + 2), i)
            qc.rz(np.pi / (layer + 3), i)
        
        # Entanglement layer
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i + 1)
    
    # Measurements
    qc.measure_all()
    
    return qc


def run_comparison(num_qubits, shots=1024):
    """Run CPU vs GPU comparison for given qubit count."""
    # Create circuit
    circuit = create_gpu_friendly_circuit(num_qubits)
    
    # CPU benchmark
    cpu_sim = AerSimulator(device='CPU', method='statevector')
    cpu_circuit = transpile(circuit, cpu_sim)
    
    start = time.time()
    cpu_result = cpu_sim.run(cpu_circuit, shots=shots).result()
    cpu_time = time.time() - start
    
    # GPU benchmark
    gpu_sim = AerSimulator(device='GPU', method='statevector')
    gpu_circuit = transpile(circuit, gpu_sim)
    
    start = time.time()
    gpu_result = gpu_sim.run(gpu_circuit, shots=shots).result()
    gpu_time = time.time() - start
    
    speedup = cpu_time / gpu_time
    
    return {
        'qubits': num_qubits,
        'depth': circuit.depth(),
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'cpu_counts': cpu_result.get_counts(),
        'gpu_counts': gpu_result.get_counts()
    }


def main():
    print("=" * 70)
    print("Qiskit Aer - GPU Quick Test")
    print("Compatible with AMD ROCm and Nvidia CUDA GPUs")
    print("=" * 70)
    print()
    
    # Check available devices
    sim = AerSimulator()
    devices = sim.available_devices()
    print(f"Available devices: {devices}")
    print()
    
    if 'GPU' not in devices:
        print("✗ GPU not available. Please check GPU drivers:")
        print("  - AMD: ROCm installation")
        print("  - Nvidia: CUDA toolkit")
        exit(1)
    
    print("✓ GPU acceleration is available!")
    print()
    
    # Detect GPU type
    try:
        import subprocess
        # Check for AMD
        result = subprocess.run(['rocm-smi'], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("GPU Type: AMD ROCm")
    except:
        try:
            # Check for Nvidia
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                print("GPU Type: Nvidia CUDA")
        except:
            print("GPU Type: Unknown (but GPU device detected)")
    
    print()
    print("=" * 70)
    print("Testing GPU Speedup Across Multiple Qubit Counts")
    print("=" * 70)
    print()
    
    # Test multiple qubit counts to show crossover
    test_qubits = [15, 20, 25, 28]
    
    print(f"{'Qubits':<10} {'Depth':<10} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
    print("-" * 70)
    
    results = []
    for num_qubits in test_qubits:
        try:
            result = run_comparison(num_qubits, shots=1024)
            results.append(result)
            
            status = "✓" if result['speedup'] > 1 else "⚠"
            print(f"{result['qubits']:<10} {result['depth']:<10} "
                  f"{result['cpu_time']:<12.4f} {result['gpu_time']:<12.4f} "
                  f"{result['speedup']:<12.2f}x {status}")
        except Exception as e:
            print(f"{num_qubits:<10} {'Error':<10} {str(e)[:40]}")
    
    print("-" * 70)
    print()
    
    # Summary
    if results:
        gpu_faster = [r for r in results if r['speedup'] > 1]
        
        if gpu_faster:
            best = max(gpu_faster, key=lambda x: x['speedup'])
            print("=" * 70)
            print("Summary")
            print("=" * 70)
            print(f"✓ GPU shows speedup for circuits with {gpu_faster[0]['qubits']}+ qubits")
            print(f"✓ Best speedup: {best['speedup']:.2f}x at {best['qubits']} qubits")
            
            if len(results) > len(gpu_faster):
                slower = [r for r in results if r['speedup'] <= 1]
                print(f"⚠ GPU slower for <{gpu_faster[0]['qubits']} qubits due to overhead")
            
            print()
            print("Recommendation:")
            print(f"  - Use GPU for circuits with ≥{gpu_faster[0]['qubits']} qubits")
            print(f"  - Expected speedup: {min(r['speedup'] for r in gpu_faster):.1f}x to {max(r['speedup'] for r in gpu_faster):.1f}x")
        else:
            print("⚠ GPU not showing speedup for tested qubit counts")
            print("  Try larger circuits (30+ qubits) for better GPU utilization")
        
        print()
        print("✓ Test complete!")


if __name__ == "__main__":
    main()
