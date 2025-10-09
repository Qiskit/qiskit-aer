#!/usr/bin/env python3
"""
Quick ROCm GPU Test

Simple script to verify GPU functionality and compare basic performance.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import time

print("=" * 60)
print("Qiskit Aer - ROCm GPU Quick Test")
print("=" * 60)
print()

# Check available devices
sim = AerSimulator()
devices = sim.available_devices()
print(f"Available devices: {devices}")
print()

if 'GPU' not in devices:
    print("✗ GPU not available. Please check ROCm installation.")
    exit(1)

print("✓ GPU is available!")
print()

# Create a simple quantum circuit
qc = QuantumCircuit(25)
qc.h(range(25))
for i in range(24):
    qc.cx(i, i + 1)
qc.measure_all()

print(f"Test circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
print()

# Test CPU
print("Testing CPU...")
cpu_sim = AerSimulator(device='CPU')
start = time.time()
cpu_result = cpu_sim.run(qc, shots=1024).result()
cpu_time = time.time() - start
print(f"  CPU time: {cpu_time:.4f} seconds")

# Test GPU
print("Testing GPU...")
gpu_sim = AerSimulator(device='GPU')
start = time.time()
gpu_result = gpu_sim.run(qc, shots=1024).result()
gpu_time = time.time() - start
print(f"  GPU time: {gpu_time:.4f} seconds")

print()
speedup = cpu_time / gpu_time
print(f"Speedup: {speedup:.2f}x")
print()

if speedup > 1:
    print(f"✓ GPU is {speedup:.1f}x faster than CPU!")
else:
    print("⚠ GPU not faster (try larger circuits with 25+ qubits)")

print()
print("Results match:", cpu_result.get_counts() == gpu_result.get_counts())
