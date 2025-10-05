#!/usr/bin/env python3
"""
ROCm GPU Benchmarking Script for Qiskit Aer

This script benchmarks the performance of Qiskit Aer on AMD ROCm GPUs
and compares it against CPU performance.

Usage:
    python rocm_benchmark.py [--output results.json] [--gpu] [--cpu]
"""

import argparse
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QuantumVolume, EfficientSU2
    from qiskit_aer import AerSimulator
except ImportError as e:
    print(f"Error: {e}")
    print("Please install qiskit and qiskit-aer")
    sys.exit(1)


def get_system_info() -> Dict[str, Any]:
    """Gather system information including ROCm details."""
    import platform
    import subprocess
    
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try to get ROCm version
    try:
        result = subprocess.run(
            ["cat", "/opt/rocm/.info/version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            info["rocm_version"] = result.stdout.strip()
    except Exception:
        info["rocm_version"] = "unknown"
    
    # Try to get GPU info
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            info["gpu_info"] = result.stdout.strip()
    except Exception:
        info["gpu_info"] = "unknown"
    
    # Try to get GPU architecture
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Name:' in line and 'gfx' in line:
                    info["gpu_arch"] = line.split()[1]
                    break
    except Exception:
        info["gpu_arch"] = "unknown"
    
    return info


def benchmark_quantum_volume(
    n_qubits: int,
    depth: int,
    shots: int,
    device: str,
    blocking_qubits: int = None
) -> Dict[str, Any]:
    """
    Benchmark Quantum Volume circuit.
    
    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        shots: Number of shots
        device: 'CPU' or 'GPU'
        blocking_qubits: Chunk size for GPU (optional)
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"  Benchmarking QV({n_qubits}, depth={depth}) on {device}...", end=" ", flush=True)
    
    # Create circuit
    qv = QuantumVolume(n_qubits, depth, seed=42)
    qv.measure_all()
    
    # Create simulator
    sim = AerSimulator(method='statevector', device=device)
    
    # Transpile
    qv_transpiled = transpile(qv, sim)
    
    # Run with timing
    run_options = {'shots': shots}
    if device == 'GPU' and blocking_qubits:
        run_options['blocking_enable'] = True
        run_options['blocking_qubits'] = blocking_qubits
    
    start_time = time.time()
    try:
        result = sim.run(qv_transpiled, **run_options).result()
        elapsed = time.time() - start_time
        
        if result.success:
            print(f"✓ {elapsed:.2f}s")
            return {
                "success": True,
                "time": elapsed,
                "shots": shots,
                "qubits": n_qubits,
                "depth": depth,
                "device": device,
                "blocking_qubits": blocking_qubits,
                "metadata": result.to_dict().get('metadata', {})
            }
        else:
            print(f"✗ Failed")
            return {
                "success": False,
                "error": "Simulation failed",
                "qubits": n_qubits,
                "device": device
            }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "qubits": n_qubits,
            "device": device
        }


def benchmark_parametric_circuit(
    n_qubits: int,
    reps: int,
    shots: int,
    device: str
) -> Dict[str, Any]:
    """Benchmark parametric circuit (EfficientSU2)."""
    print(f"  Benchmarking EfficientSU2({n_qubits}, reps={reps}) on {device}...", end=" ", flush=True)
    
    # Create circuit
    circuit = EfficientSU2(n_qubits, reps=reps)
    circuit.measure_all()
    
    # Bind parameters
    import numpy as np
    params = np.random.random(circuit.num_parameters)
    bound_circuit = circuit.bind_parameters(params)
    
    # Create simulator
    sim = AerSimulator(method='statevector', device=device)
    
    # Transpile
    transpiled = transpile(bound_circuit, sim)
    
    # Run with timing
    start_time = time.time()
    try:
        result = sim.run(transpiled, shots=shots).result()
        elapsed = time.time() - start_time
        
        if result.success:
            print(f"✓ {elapsed:.2f}s")
            return {
                "success": True,
                "time": elapsed,
                "shots": shots,
                "qubits": n_qubits,
                "reps": reps,
                "device": device
            }
        else:
            print(f"✗ Failed")
            return {"success": False, "error": "Simulation failed"}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


def run_benchmark_suite(test_gpu: bool = True, test_cpu: bool = True) -> Dict[str, Any]:
    """Run complete benchmark suite."""
    results = {
        "system_info": get_system_info(),
        "benchmarks": []
    }
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║    Qiskit Aer ROCm GPU Benchmark Suite                ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    print("System Information:")
    for key, value in results["system_info"].items():
        print(f"  {key}: {value}")
    print()
    
    # Test configurations
    test_configs = [
        # (qubits, depth/reps, shots, test_type)
        (10, 10, 100, 'qv'),
        (15, 10, 100, 'qv'),
        (20, 10, 100, 'qv'),
        (25, 10, 50, 'qv'),
        (30, 10, 10, 'qv'),
        (10, 3, 100, 'parametric'),
        (15, 3, 100, 'parametric'),
    ]
    
    # Check GPU availability
    if test_gpu:
        try:
            sim = AerSimulator(device='GPU')
            devices = sim.available_devices()
            if 'GPU' not in devices:
                print("Warning: GPU not available, skipping GPU tests\n")
                test_gpu = False
        except Exception as e:
            print(f"Warning: Cannot access GPU ({e}), skipping GPU tests\n")
            test_gpu = False
    
    # Run benchmarks
    for qubits, depth_reps, shots, test_type in test_configs:
        print(f"\nTest: {test_type.upper()} - {qubits} qubits")
        
        if test_cpu:
            if test_type == 'qv':
                result = benchmark_quantum_volume(qubits, depth_reps, shots, 'CPU')
            else:
                result = benchmark_parametric_circuit(qubits, depth_reps, shots, 'CPU')
            results["benchmarks"].append(result)
        
        if test_gpu:
            # Determine optimal blocking_qubits
            blocking = None
            gpu_arch = results["system_info"].get("gpu_arch", "")
            if "gfx940" in gpu_arch or "gfx941" in gpu_arch or "gfx942" in gpu_arch:
                blocking = 28  # MI300
            elif "gfx90a" in gpu_arch:
                blocking = 27  # MI250
            elif "gfx908" in gpu_arch:
                blocking = 25  # MI100
            elif "gfx1100" in gpu_arch:
                blocking = 25  # RX 7000
            elif "gfx1030" in gpu_arch:
                blocking = 24  # RX 6000
            else:
                blocking = 23  # Conservative default
            
            if test_type == 'qv':
                result = benchmark_quantum_volume(qubits, depth_reps, shots, 'GPU', blocking)
            else:
                result = benchmark_parametric_circuit(qubits, depth_reps, shots, 'GPU')
            results["benchmarks"].append(result)
    
    # Calculate speedups
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    cpu_times = {b['qubits']: b['time'] for b in results["benchmarks"] 
                 if b.get('success') and b.get('device') == 'CPU'}
    gpu_times = {b['qubits']: b['time'] for b in results["benchmarks"] 
                 if b.get('success') and b.get('device') == 'GPU'}
    
    if cpu_times and gpu_times:
        print("\nSpeedup (CPU time / GPU time):")
        for qubits in sorted(set(cpu_times.keys()) & set(gpu_times.keys())):
            speedup = cpu_times[qubits] / gpu_times[qubits]
            print(f"  {qubits} qubits: {speedup:.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qiskit Aer on ROCm GPUs")
    parser.add_argument('--output', '-o', default='benchmark_results.json',
                        help='Output JSON file (default: benchmark_results.json)')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Run GPU benchmarks (default: True)')
    parser.add_argument('--cpu', action='store_true', default=True,
                        help='Run CPU benchmarks (default: True)')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu',
                        help='Skip GPU benchmarks')
    parser.add_argument('--no-cpu', action='store_false', dest='cpu',
                        help='Skip CPU benchmarks')
    
    args = parser.parse_args()
    
    # Run benchmarks
    results = run_benchmark_suite(test_gpu=args.gpu, test_cpu=args.cpu)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    
    # Print summary
    successful = sum(1 for b in results["benchmarks"] if b.get('success'))
    total = len(results["benchmarks"])
    print(f"✓ Completed {successful}/{total} benchmarks successfully")


if __name__ == '__main__':
    main()
