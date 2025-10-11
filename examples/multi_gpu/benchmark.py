#!/usr/bin/env python3
"""
Multi-GPU Benchmark for Qiskit Aer with ROCm
============================================

Comprehensive benchmark of multi-GPU performance for quantum circuit simulation.
Tests various circuit sizes and GPU configurations.

Usage:
    python3 benchmark.py

Features:
    - Scaling analysis (1, 2, 4, 8 GPUs)
    - Circuit size comparison (28-34 qubits)
    - Performance metrics and speedup calculations
    - Memory usage analysis

Time: ~5-10 minutes (depending on available GPUs)
"""

import subprocess
import sys
import time
from qiskit_aer import AerSimulator
from qiskit.circuit.library import quantum_volume


def get_gpu_count():
    """Get number of available GPUs"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.count('GPU[')
    except Exception:
        pass
    return 1


def show_system_info():
    """Display system information"""
    print("="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    try:
        # GPU info
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.split('\n') if 'GPU[0]' in l]
            if lines:
                print(f"GPU Model: {lines[0].split(':')[-1].strip()}")
        
        # Count GPUs
        gpu_count = get_gpu_count()
        print(f"Total GPUs: {gpu_count}")
        
        # Memory info
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'GPU[0]' in line and 'Total VRAM' in line:
                    mem = line.split(':')[-1].strip()
                    print(f"GPU Memory: {mem} per GPU")
                    break
        
        return gpu_count
    except Exception as e:
        print(f"Could not gather system info: {e}")
        return 1


def benchmark_configuration(qubits, num_gpus, shots=100):
    """Benchmark a specific configuration"""
    
    backend = AerSimulator(method='statevector', device='GPU')
    circuit = quantum_volume(qubits, depth=5, seed=42)
    circuit.measure_all()
    
    # Determine if blocking is needed
    if qubits <= 31:
        # Single GPU, no blocking
        start = time.time()
        result = backend.run(
            circuit,
            shots=shots,
            seed_simulator=42
        ).result()
        elapsed = time.time() - start
        
        metadata = result.results[0].metadata
        gpus_used = 1
        chunks = 1
        
    else:
        # Multi-GPU with blocking
        blocking = 27  # Hard limit
        target_gpus = list(range(num_gpus))
        
        start = time.time()
        result = backend.run(
            circuit,
            shots=shots,
            seed_simulator=42,
            blocking_enable=True,
            blocking_qubits=blocking,
            target_gpus=target_gpus,
            batched_shots_gpu=True,
            batched_shots_gpu_max_qubits=qubits
        ).result()
        elapsed = time.time() - start
        
        metadata = result.results[0].metadata
        cacheblocking = metadata.get('cacheblocking', {})
        gpus_used = cacheblocking.get('chunk_parallel_gpus', 1)
        chunks = 2 ** (qubits - cacheblocking.get('block_bits', 27))
    
    return {
        'qubits': qubits,
        'gpus_requested': num_gpus,
        'gpus_used': gpus_used,
        'chunks': chunks,
        'time': elapsed,
        'shots': shots,
        'memory_mb': metadata.get('required_memory_mb', 0)
    }


def run_scaling_benchmark():
    """Test GPU scaling for 32-qubit circuit"""
    print("\n" + "="*70)
    print("GPU SCALING BENCHMARK (32 qubits)")
    print("="*70)
    print("Testing how performance scales with GPU count...\n")
    
    max_gpus = get_gpu_count()
    results = []
    
    # Test with 1, 2, 4, 8 GPUs (up to available)
    gpu_configs = [1, 2, 4, 8]
    gpu_configs = [g for g in gpu_configs if g <= max_gpus]
    
    for num_gpus in gpu_configs:
        print(f"Testing with {num_gpus} GPU{'s' if num_gpus > 1 else ''}...", end=' ')
        sys.stdout.flush()
        
        try:
            result = benchmark_configuration(32, num_gpus, shots=100)
            results.append(result)
            print(f"✅ {result['time']:.2f}s (GPUs used: {result['gpus_used']})")
        except Exception as e:
            print(f"❌ Failed: {str(e)[:50]}")
    
    # Display results
    if results:
        print("\n" + "-"*70)
        print(f"{'GPUs':<8} {'Time':<10} {'Speedup':<10} {'Efficiency':<12} {'Status'}")
        print("-"*70)
        
        baseline = results[0]['time']
        for r in results:
            speedup = baseline / r['time']
            efficiency = (speedup / r['gpus_requested']) * 100
            status = "✅" if r['gpus_used'] == r['gpus_requested'] else "⚠️"
            print(f"{r['gpus_requested']:<8} {r['time']:<10.2f} {speedup:<10.2f} {efficiency:<11.1f}% {status}")
        
        print("-"*70)
    
    return results


def run_circuit_size_benchmark():
    """Test different circuit sizes with optimal GPU count"""
    print("\n" + "="*70)
    print("CIRCUIT SIZE BENCHMARK")
    print("="*70)
    print("Testing different circuit sizes with optimal GPU configuration...\n")
    
    max_gpus = get_gpu_count()
    results = []
    
    # Test configurations: (qubits, recommended_gpus)
    configs = [
        (28, 1, "Single GPU - fits easily"),
        (30, 1, "Single GPU - near limit"),
        (31, 1, "Single GPU - maximum"),
        (32, 2, "Multi-GPU - 32 chunks"),
        (33, 4, "Multi-GPU - 64 chunks"),
    ]
    
    # Add 34q if we have 8 GPUs
    if max_gpus >= 8:
        configs.append((34, 8, "Multi-GPU - 128 chunks"))
    
    print(f"{'Qubits':<8} {'GPUs':<6} {'Time':<10} {'State Size':<12} {'Status'}")
    print("-"*70)
    
    for qubits, gpus, desc in configs:
        if gpus > max_gpus:
            print(f"{qubits:<8} {gpus:<6} {'SKIP':<10} {'-':<12} ⚠️  Need {gpus} GPUs")
            continue
        
        try:
            result = benchmark_configuration(qubits, gpus, shots=50)
            results.append(result)
            
            state_size = 2 ** (qubits - 30)  # in GB
            status = "✅" if result['gpus_used'] == gpus else "⚠️"
            
            print(f"{qubits:<8} {gpus:<6} {result['time']:<10.2f} {state_size:<11.1f}GB {status} {desc}")
        except Exception as e:
            print(f"{qubits:<8} {gpus:<6} {'FAIL':<10} {'-':<12} ❌ {str(e)[:30]}")
    
    print("-"*70)
    
    return results


def run_memory_analysis():
    """Analyze memory usage patterns"""
    print("\n" + "="*70)
    print("MEMORY USAGE ANALYSIS")
    print("="*70)
    
    configs = [
        (28, 1, "4 GB state"),
        (30, 1, "16 GB state"),
        (31, 1, "32 GB state"),
        (32, 2, "64 GB state (2 GPUs)"),
    ]
    
    print(f"{'Qubits':<8} {'GPUs':<6} {'State Size':<12} {'Memory Used':<15} {'Chunks':<10}")
    print("-"*70)
    
    for qubits, gpus, desc in configs:
        try:
            result = benchmark_configuration(qubits, gpus, shots=10)
            state_gb = 2 ** (qubits - 30)
            mem_gb = result['memory_mb'] / 1024
            
            print(f"{qubits:<8} {gpus:<6} {state_gb:<11.1f}GB {mem_gb:<14.1f}GB {result['chunks']:<10} {desc}")
        except Exception:
            print(f"{qubits:<8} {gpus:<6} {'-':<12} {'-':<15} {'-':<10} Failed")
    
    print("-"*70)


def main():
    """Run comprehensive multi-GPU benchmark"""
    
    print("\n" + "="*70)
    print("QISKIT AER - MULTI-GPU BENCHMARK")
    print("="*70)
    print("\nComprehensive performance analysis of multi-GPU execution.")
    print("This will take 5-10 minutes...\n")
    
    # System info
    max_gpus = show_system_info()
    
    if max_gpus < 2:
        print("\n⚠️  Warning: Only 1 GPU detected. Multi-GPU tests will be limited.")
        input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        # Run benchmarks
        scaling_results = run_scaling_benchmark()
        size_results = run_circuit_size_benchmark()
        run_memory_analysis()
        
        # Final summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if scaling_results and len(scaling_results) > 1:
            best_scaling = scaling_results[-1]
            baseline = scaling_results[0]['time']
            speedup = baseline / best_scaling['time']
            
            print(f"\nBest Multi-GPU Performance:")
            print(f"  Configuration: {best_scaling['gpus_used']} GPUs")
            print(f"  Circuit: 32 qubits")
            print(f"  Speedup: {speedup:.2f}x vs single GPU")
            print(f"  Time: {best_scaling['time']:.2f}s")
        
        if size_results:
            max_qubits = max(r['qubits'] for r in size_results)
            print(f"\nLargest Circuit Simulated:")
            print(f"  Qubits: {max_qubits}")
            print(f"  State size: {2 ** (max_qubits - 30):.1f} GB")
        
        print("\n" + "="*70)
        print("✅ Benchmark completed successfully!")
        print("="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
