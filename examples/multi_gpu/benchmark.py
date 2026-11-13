#!/usr/bin/env python3
"""
Multi-GPU Benchmark for Qiskit Aer with ROCm
NVIDIA-Aligned, Production Ready

Comprehensive multi-GPU performance analysis using industry-standard
Quantum Volume circuits. Supports precision selection and hardware-agnostic design.

Features:
    - Quantum Volume circuits (NVIDIA cuQuantum aligned)
    - Precision selection: complex128 (default) or complex64
    - Multi-GPU scaling analysis (1, 2, 4, 8 GPUs)
    - Hardware-agnostic GPU detection (AMD/NVIDIA)
    - Circuit specifications and memory analysis
    - Publication-quality output

Requirements:
    - qiskit-aer-gpu-rocm (AMD) or qiskit-aer-gpu (NVIDIA)
    - Multiple GPUs with ROCm or CUDA support

Usage:
    # Default: complex128, 32-34 qubits, auto GPU detection
    python3 benchmark.py
    
    # Use complex64 for performance
    python3 benchmark.py --precision complex64
    
    # Custom qubit range
    python3 benchmark.py --qubits 32,33,34,35
    
    # Specific GPU count
    python3 benchmark.py --gpus 4
    
    # Custom configuration
    python3 benchmark.py --precision complex64 --qubits 32,34,36 --gpus 2,4,8 --shots 200

Time: ~10-20 minutes (depending on configurations)
"""

import argparse
import subprocess
import sys
import time
from qiskit_aer import AerSimulator
from qiskit.circuit.library import quantum_volume
import numpy as np


def detect_gpu_vendor():
    """
    Detect GPU vendor (AMD or NVIDIA).
    
    Returns:
        str: 'AMD', 'NVIDIA', or 'Unknown'
    """
    # Try AMD first
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and 'GPU[' in result.stdout:
            return 'AMD'
    except Exception:
        pass
    
    # Try NVIDIA
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and 'GPU' in result.stdout:
            return 'NVIDIA'
    except Exception:
        pass
    
    return 'Unknown'


def get_gpu_count(vendor='AMD'):
    """Get number of available GPUs"""
    try:
        if vendor == 'AMD':
            result = subprocess.run(
                ['rocm-smi', '--showid'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.count('GPU[')
        elif vendor == 'NVIDIA':
            result = subprocess.run(
                ['nvidia-smi', '-L'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.count('GPU ')
    except Exception:
        pass
    return 1


def get_gpu_model(vendor='AMD'):
    """Get GPU model name"""
    try:
        if vendor == 'AMD':
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'GPU[0]' in line:
                        return line.split(':')[-1].strip()
        elif vendor == 'NVIDIA':
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '--id=0'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return "Unknown"


def get_gpu_memory(vendor='AMD'):
    """Get GPU memory in GB"""
    try:
        if vendor == 'AMD':
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VRAM Total Memory' in line and 'GPU[0]' in line:
                        parts = line.split(':')
                        if len(parts) >= 3:
                            try:
                                memory_bytes = float(parts[-1].strip())
                                return memory_bytes / (1024 ** 3)
                            except ValueError:
                                continue
        elif vendor == 'NVIDIA':
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits', '--id=0'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                memory_mb = float(result.stdout.strip())
                return memory_mb / 1024
    except Exception:
        pass
    return None


def calculate_max_qubits_multi_gpu(gpu_memory_gb, num_gpus, precision='complex128'):
    """
    Calculate theoretical maximum qubits for multi-GPU configuration.
    
    With blocking, state vector is distributed across GPUs:
    - Each GPU holds: state_size / num_gpus chunks
    - Enables circuits beyond single GPU capacity
    
    Args:
        gpu_memory_gb: Memory per GPU in GB
        num_gpus: Number of GPUs to use
        precision: 'complex128' or 'complex64'
    
    Returns:
        int: Theoretical maximum qubits
    """
    if gpu_memory_gb is None:
        return 32 if num_gpus > 1 else 28
    
    bytes_per_element = 16 if precision == 'complex128' else 8
    
    # With multi-GPU blocking (blocking_qubits=27):
    # - Chunk size: 2 GB (complex128) or 1 GB (complex64)
    # - Empirically validated limit: ~20 chunks per GPU (not theoretical limit!)
    # - Total chunks available: 20 * num_gpus
    # - Max qubits: log2(total_chunks) + 27
    
    # Use empirically validated limit
    chunks_per_gpu = 20  # Conservative, based on actual testing
    if gpu_memory_gb > 150:
        chunks_per_gpu = 24  # Slightly less conservative for large GPUs
    
    total_chunks = chunks_per_gpu * num_gpus
    
    # Max qubits = log2(total_chunks) + blocking_qubits
    if total_chunks > 0:
        max_qubits = int(np.log2(total_chunks)) + 27
    else:
        max_qubits = 30
    
    # Reasonable bounds
    return min(max(max_qubits, 30), 40)


def get_circuit_specs(qubits, precision='complex128', blocking_qubits=27):
    """Get circuit specifications including memory requirements"""
    circuit = quantum_volume(qubits, depth=5, seed=42)
    circuit.measure_all()
    
    gate_counts = circuit.count_ops()
    total_gates = sum(gate_counts.values())
    
    bytes_per_element = 16 if precision == 'complex128' else 8
    state_vector_bytes = (2 ** qubits) * bytes_per_element
    
    # Blocking info
    blocking_enabled = qubits >= 32
    if blocking_enabled:
        chunk_size_bytes = (2 ** blocking_qubits) * bytes_per_element
        num_chunks = 2 ** (qubits - blocking_qubits)
    else:
        chunk_size_bytes = state_vector_bytes
        num_chunks = 1
    
    return {
        'qubits': qubits,
        'depth': circuit.depth(),
        'total_gates': total_gates,
        'gate_counts': dict(gate_counts),
        'state_vector_gb': state_vector_bytes / (1024 ** 3),
        'blocking_enabled': blocking_enabled,
        'chunk_size_gb': chunk_size_bytes / (1024 ** 3),
        'num_chunks': num_chunks
    }


def show_system_info(precision='complex128'):
    """Display system information and capabilities"""
    print("="*70)
    print("MULTI-GPU BENCHMARK - NVIDIA-ALIGNED")
    print("="*70)
    print()
    
    # Detect GPU vendor
    vendor = detect_gpu_vendor()
    
    if vendor == 'Unknown':
        print("✗ No GPU detected. Make sure ROCm or CUDA is installed.")
        return False, None, None, None, None
    
    print(f"✓ GPU Vendor: {vendor}")
    
    # GPU count
    gpu_count = get_gpu_count(vendor)
    print(f"✓ Available GPUs: {gpu_count}")
    
    # GPU model
    gpu_model = get_gpu_model(vendor)
    print(f"✓ GPU Model: {gpu_model}")
    
    # GPU memory
    gpu_memory = get_gpu_memory(vendor)
    if gpu_memory:
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB per GPU")
        total_memory = gpu_memory * gpu_count
        print(f"✓ Total GPU Memory: {total_memory:.1f} GB ({gpu_count} GPUs)")
    else:
        print("⚠️  Could not detect GPU memory")
        gpu_memory = None
    
    # Precision info
    bytes_per_element = 16 if precision == 'complex128' else 8
    print(f"\nPrecision: {precision} ({bytes_per_element} bytes per element)")
    if precision == 'complex64':
        print("  ⚠️  Note: complex64 uses half memory but reduced accuracy")
    
    print()
    
    return True, vendor, gpu_count, gpu_memory, gpu_model


def calculate_min_gpus(qubits, precision='complex128', gpu_memory_gb=None, blocking_qubits=27):
    """
    Calculate minimum GPUs required for a circuit.
    
    Based on empirical validation (MI300X):
    - Qiskit Aer's blocking has implementation limits beyond memory
    - Safe limit: ~16-20 chunks per GPU (not theoretical 48!)
    - Validated: 34q (128 chunks) works with 8 GPUs (blocking=27)
    - Validated: 35q (256 chunks) needs 16 GPUs (blocking=27)
    
    Args:
        qubits: Number of qubits
        precision: Precision type (for future use)
        gpu_memory_gb: GPU memory in GB (for future use)
        blocking_qubits: Blocking parameter (default 27)
    
    Returns:
        int: Minimum number of GPUs required
    """
    if qubits <= 31:
        return 1
    
    # Calculate chunks based on blocking parameter
    num_chunks = 2 ** (qubits - blocking_qubits)
    
    # EMPIRICALLY VALIDATED LIMITS (MI300X testing):
    # - 34q complex128 (128 chunks, 256 GB) works with 8 GPUs ✅
    #   → 16 chunks/GPU, 32 GB/GPU
    # - 35q complex128 (256 chunks, 512 GB) FAILS with 8 GPUs ❌  
    #   → 32 chunks/GPU, 64 GB/GPU
    # - 35q complex64 (128 chunks, 256 GB) FAILS with 8 GPUs ❌
    #   → 16 chunks/GPU, 32 GB/GPU (same as 34q complex128!)
    #
    # CRITICAL DISCOVERY: It's not just chunks OR memory - there's a
    # COMBINED constraint based on QUBIT COUNT itself!
    # 35+ qubits appears to need more GPUs regardless of chunking/memory.
    
    # Calculate memory requirement
    bytes_per_element = 16 if precision == 'complex128' else 8
    total_memory_gb = (2 ** qubits) * bytes_per_element / (1024 ** 3)
    
    # Use validated baseline: 34q complex128 works on 8 GPUs
    baseline_chunks_per_gpu = 16
    baseline_memory_per_gpu = 32  # GB
    
    # Calculate minimum GPUs from both constraints
    min_gpus_chunks = max(1, (num_chunks + baseline_chunks_per_gpu - 1) // baseline_chunks_per_gpu)
    min_gpus_memory = max(1, int(np.ceil(total_memory_gb / baseline_memory_per_gpu)))
    
    # MUST satisfy BOTH
    min_gpus = max(min_gpus_chunks, min_gpus_memory)
    
    # CRITICAL: Add safety margin for 35+ qubits (empirically fails otherwise)
    if qubits >= 35:
        min_gpus = max(min_gpus, 11)  # Empirical minimum for 35+ qubits
    
    return min_gpus


def run_benchmark(qubits, num_gpus, shots=100, precision='complex128', gpu_memory_gb=None, blocking_qubits=27):
    """
    Run multi-GPU benchmark for specific configuration.
    
    Args:
        qubits: Number of qubits
        num_gpus: Number of GPUs to use
        shots: Number of measurement shots
        precision: 'complex128' or 'complex64'
        gpu_memory_gb: GPU memory in GB (for validation)
        blocking_qubits: Blocking parameter (default 27)
    
    Returns:
        tuple: (time, gpus_used, error, specs)
    """
    try:
        # Pre-flight validation: Check if enough GPUs for circuit
        min_gpus = calculate_min_gpus(qubits, precision, gpu_memory_gb, blocking_qubits)
        if num_gpus < min_gpus:
            specs = get_circuit_specs(qubits, precision, blocking_qubits)
            error_msg = f"Need >={min_gpus} GPUs ({specs['num_chunks']} chunks)"
            return None, None, error_msg, specs
        
        # Convert precision to Qiskit Aer format
        aer_precision = 'double' if precision == 'complex128' else 'single'
        
        # Create simulator
        backend = AerSimulator(method='statevector', device='GPU', precision=aer_precision)
        
        # Create quantum volume circuit
        circuit = quantum_volume(qubits, depth=5, seed=42)
        circuit.measure_all()
        
        # Get circuit specs
        specs = get_circuit_specs(qubits, precision, blocking_qubits)
        
        # Configure run options
        run_options = {'shots': shots, 'seed_simulator': 42}
        
        # Multi-GPU configuration for >31 qubits
        if qubits >= 32 and num_gpus > 1:
            run_options.update({
                'blocking_enable': True,
                'blocking_qubits': blocking_qubits,
                'target_gpus': list(range(num_gpus)),
                'batched_shots_gpu': True,
                'batched_shots_gpu_max_qubits': qubits
            })
        elif qubits >= 32 and num_gpus == 1:
            # Single GPU with blocking - only for circuits that fit
            # This should have been caught by pre-flight check
            run_options.update({
                'blocking_enable': True,
                'blocking_qubits': blocking_qubits,
                'target_gpus': [0]
            })
        
        # Execute
        start = time.time()
        result = backend.run(circuit, **run_options).result()
        elapsed = time.time() - start
        
        # Extract metadata
        metadata = result.results[0].metadata
        cacheblocking = metadata.get('cacheblocking', {})
        gpus_used = cacheblocking.get('chunk_parallel_gpus', 1)
        
        return elapsed, gpus_used, None, specs
        
    except Exception as e:
        error_msg = str(e)
        if 'bad_alloc' in error_msg or 'out of memory' in error_msg.lower():
            error_msg = "Out of memory"
        elif 'runtime_error' in error_msg.lower():
            error_msg = "Circuit too large"
        else:
            error_msg = error_msg[:40]
        return None, None, error_msg, None


def print_circuit_preview(qubits_list, precision='complex128', blocking_qubits=27):
    """Print preview of circuits to be tested"""
    print("\n" + "="*90)
    print("CIRCUIT SPECIFICATIONS PREVIEW")
    print("="*90)
    print(f"{'Qubits':<8} {'Depth':<8} {'Gates':<8} {'State Vec':<12} {'Chunks':<10} {'Chunk Size':<12}")
    print("-"*90)
    
    for qubits in qubits_list[:10]:
        specs = get_circuit_specs(qubits, precision, blocking_qubits)
        print(f"{qubits:<8} {specs['depth']:<8} {specs['total_gates']:<8} "
              f"{specs['state_vector_gb']:>10.2f} GB {specs['num_chunks']:<10} "
              f"{specs['chunk_size_gb']:>10.2f} GB")
    
    if len(qubits_list) > 10:
        print(f"... and {len(qubits_list) - 10} more circuits")
    print("-"*90)
    print()


def run_scaling_benchmark(qubits, gpu_configs, shots=100, precision='complex128', gpu_memory_gb=None, blocking_qubits=27):
    """
    Test GPU scaling for specific circuit size.
    
    Args:
        qubits: Number of qubits
        gpu_configs: List of GPU counts to test
        shots: Number of shots
        precision: Precision type
        gpu_memory_gb: GPU memory in GB (for validation)
        blocking_qubits: Blocking parameter (default 27)
    
    Returns:
        list: Benchmark results
    """
    print(f"\n{'='*90}")
    print(f"GPU SCALING ANALYSIS ({qubits} qubits, {precision})")
    print("="*90)
    
    specs = get_circuit_specs(qubits, precision, blocking_qubits)
    print(f"Circuit: Quantum Volume, Depth: {specs['depth']}, Gates: {specs['total_gates']}")
    print(f"State Vector: {specs['state_vector_gb']:.2f} GB, Chunks: {specs['num_chunks']}")
    print()
    
    print(f"{'GPUs':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency':<12} {'Status':<20}")
    print("-"*90)
    
    results = []
    
    for num_gpus in gpu_configs:
        elapsed, gpus_used, error, _ = run_benchmark(qubits, num_gpus, shots, precision, gpu_memory_gb, blocking_qubits)
        
        if error:
            print(f"{num_gpus:<8} {'Skip':<12} {'-':<12} {'-':<12} ⚠️  {error}")
            continue
        
        # Calculate metrics
        if results:
            baseline = results[0]['time']
            speedup = baseline / elapsed
            efficiency = (speedup / num_gpus) * 100
        else:
            speedup = 1.0
            efficiency = 100.0
        
        # Status
        if gpus_used == num_gpus:
            status = f"✅ Using {gpus_used} GPUs"
        else:
            status = f"⚠️  Using {gpus_used}/{num_gpus} GPUs"
        
        print(f"{num_gpus:<8} {elapsed:<12.4f} {speedup:>11.2f}x {efficiency:>10.1f}% {status:<20}")
        
        results.append({
            'qubits': qubits,
            'gpus_requested': num_gpus,
            'gpus_used': gpus_used,
            'time': elapsed,
            'speedup': speedup,
            'efficiency': efficiency,
            'specs': specs
        })
    
    print("-"*90)
    return results


def run_circuit_size_benchmark(qubits_list, num_gpus, shots=100, precision='complex128', gpu_memory_gb=None, blocking_qubits=27):
    """
    Test different circuit sizes with fixed GPU count.
    
    Args:
        qubits_list: List of qubit counts to test
        num_gpus: Number of GPUs to use
        shots: Number of shots
        precision: Precision type
        gpu_memory_gb: GPU memory in GB (for validation)
        blocking_qubits: Blocking parameter (default 27)
    
    Returns:
        list: Benchmark results
    """
    print(f"\n{'='*90}")
    print(f"CIRCUIT SIZE ANALYSIS ({num_gpus} GPUs, {precision})")
    print("="*90)
    print()
    
    print(f"{'Qubits':<8} {'Depth':<8} {'Gates':<8} {'Time (s)':<12} {'Memory':<12} {'Status':<20}")
    print("-"*90)
    
    results = []
    
    for qubits in qubits_list:
        elapsed, gpus_used, error, specs = run_benchmark(qubits, num_gpus, shots, precision, gpu_memory_gb, blocking_qubits)
        
        if error:
            mem_str = f"{specs['state_vector_gb']:.2f} GB" if specs else '-'
            print(f"{qubits:<8} {'-':<8} {'-':<8} {'Skip':<12} {mem_str:<12} ⚠️  {error}")
            continue
        
        mem_str = f"{specs['state_vector_gb']:.2f} GB"
        
        if gpus_used == num_gpus:
            status = f"✅ Using {gpus_used} GPUs"
        else:
            status = f"⚠️  Using {gpus_used}/{num_gpus} GPUs"
        
        print(f"{qubits:<8} {specs['depth']:<8} {specs['total_gates']:<8} "
              f"{elapsed:<12.4f} {mem_str:<12} {status:<20}")
        
        results.append({
            'qubits': qubits,
            'gpus_requested': num_gpus,
            'gpus_used': gpus_used,
            'time': elapsed,
            'specs': specs
        })
    
    print("-"*90)
    return results


def print_summary(all_results, vendor, gpu_model, precision='complex128', blocking_qubits=27):
    """Print comprehensive benchmark summary"""
    if not all_results:
        return
    
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print()
    
    # Hardware info
    print(f"Hardware: {vendor} {gpu_model}")
    print(f"Circuit Type: Quantum Volume (NVIDIA cuQuantum Standard)")
    print(f"Precision: {precision} ({'16 bytes' if precision == 'complex128' else '8 bytes'} per element)")
    print(f"Blocking: {blocking_qubits} qubits", end="")
    if blocking_qubits != 27:
        chunk_size_gb = (2 ** blocking_qubits) * (16 if precision == 'complex128' else 8) / (1024 ** 3)
        print(f" ⚠️  NON-STANDARD ({chunk_size_gb:.1f} GB chunks)")
    else:
        print(" (NVIDIA standard)")
    print(f"Total configurations tested: {len(all_results)}")
    
    # Circuit statistics
    if all_results and 'specs' in all_results[0]:
        total_gates = sum(r['specs']['total_gates'] for r in all_results)
        avg_depth = np.mean([r['specs']['depth'] for r in all_results])
        print(f"Total gates executed: {total_gates:,}")
        print(f"Average circuit depth: {avg_depth:.1f}")
    
    # Performance analysis
    print(f"\n{'Performance Metrics:'}")
    
    # Scaling efficiency
    scaling_results = [r for r in all_results if 'efficiency' in r]
    if scaling_results:
        best_scaling = max(scaling_results, key=lambda x: x['speedup'])
        print(f"  Best speedup: {best_scaling['speedup']:.2f}x "
              f"({best_scaling['gpus_requested']} GPUs)")
        print(f"  Best efficiency: {max(r['efficiency'] for r in scaling_results):.1f}%")
    
    # Circuit size range
    qubits_tested = [r['qubits'] for r in all_results]
    print(f"  Qubit range: {min(qubits_tested)}-{max(qubits_tested)} qubits")
    
    # Execution time range
    times = [r['time'] for r in all_results]
    print(f"  Execution time range: {min(times):.2f}s - {max(times):.2f}s")
    
    # Memory analysis
    print(f"\n{'Memory Analysis:'}")
    max_mem = max(r['specs']['state_vector_gb'] for r in all_results if 'specs' in r)
    print(f"  Peak state vector: {max_mem:.2f} GB")
    
    if any(r['specs']['blocking_enabled'] for r in all_results if 'specs' in r):
        blocking_circuits = [r for r in all_results if 'specs' in r and r['specs']['blocking_enabled']]
        max_chunks = max(r['specs']['num_chunks'] for r in blocking_circuits)
        chunk_size_gb = blocking_circuits[0]['specs']['chunk_size_gb']
        print(f"  Blocking used: {len(blocking_circuits)} circuits")
        print(f"  Max chunks: {max_chunks} ({chunk_size_gb:.1f} GB each)")
    
    # Multi-GPU utilization
    multi_gpu_results = [r for r in all_results if r['gpus_requested'] > 1]
    if multi_gpu_results:
        properly_used = [r for r in multi_gpu_results if r['gpus_used'] == r['gpus_requested']]
        print(f"\n{'Multi-GPU Utilization:'}")
        print(f"  Total multi-GPU tests: {len(multi_gpu_results)}")
        print(f"  Properly utilized: {len(properly_used)}/{len(multi_gpu_results)}")
        
        if len(properly_used) < len(multi_gpu_results):
            print(f"  ⚠️  Some configurations didn't use all requested GPUs")
            print(f"     Requires: circuits >= 32 qubits + shots >= 100")
    
    print("\n" + "="*90)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Benchmark (NVIDIA-Aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: complex128, 32-34 qubits, NVIDIA standard (blocking=27)
  python3 benchmark.py
  
  # Use complex64 for performance
  python3 benchmark.py --precision complex64
  
  # Custom qubit range
  python3 benchmark.py --qubits 32,33,34,35
  
  # Specific GPU configurations
  python3 benchmark.py --gpus 2,4,8
  
  # Scaling test (32 qubits on 1,2,4 GPUs)
  python3 benchmark.py --qubits 32 --gpus 1,2,4
  
  # Experimental: Run 35q on 8 GPUs with larger chunks (blocking=28)
  python3 benchmark.py --qubits 35 --gpus 8 --blocking-qubits 28
  
  # Experimental: Run 35q on 8 GPUs with even larger chunks (blocking=29)
  python3 benchmark.py --qubits 35 --gpus 8 --blocking-qubits 29
  
  # Full customization
  python3 benchmark.py --precision complex64 --qubits 32,34,36 --gpus 2,4,8 --shots 200
        """
    )
    
    parser.add_argument('--precision', type=str, default='complex128',
                        choices=['complex128', 'complex64'],
                        help='Precision: complex128 (default) or complex64')
    
    parser.add_argument('--qubits', type=str, default=None,
                        help='Comma-separated qubit counts (e.g., "32,33,34")')
    
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU counts (e.g., "1,2,4,8")')
    
    parser.add_argument('--shots', type=int, default=100,
                        help='Number of measurement shots (default: 100)')
    
    parser.add_argument('--blocking-qubits', type=int, default=27,
                        help='Blocking qubits for multi-GPU (default: 27, NVIDIA standard). '
                             'Larger values = fewer, bigger chunks. Try 28-29 for 35+ qubits on 8 GPUs')
    
    args = parser.parse_args()
    
    # Show system info
    has_gpu, vendor, gpu_count, gpu_memory, gpu_model = show_system_info(args.precision)
    
    if not has_gpu:
        print("\nCannot run benchmark without GPU. Exiting.")
        return 1
    
    if gpu_count < 2:
        print("⚠️  Warning: Only 1 GPU detected. Multi-GPU features limited.")
        response = input("\nContinue with single GPU tests? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            return 1
        print()
    
    # Determine qubit list
    if args.qubits:
        qubits_list = [int(q.strip()) for q in args.qubits.split(',')]
    else:
        # Auto-generate based on multi-GPU capability
        qubits_list = [32, 33, 34]
        
        # Add more if we have good GPU memory
        if gpu_memory and gpu_memory >= 150:
            qubits_list.extend([35, 36])
    
    # Determine GPU configurations
    if args.gpus:
        gpu_configs = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        # Auto-generate: powers of 2 up to available
        gpu_configs = [1]
        for g in [2, 4, 8, 16]:
            if g <= gpu_count:
                gpu_configs.append(g)
    
    print(f"Test Configuration:")
    print(f"  Qubits: {qubits_list}")
    print(f"  GPU configs: {gpu_configs}")
    print(f"  Shots: {args.shots}")
    print(f"  Blocking qubits: {args.blocking_qubits}", end="")
    
    if args.blocking_qubits != 27:
        chunk_size_gb = (2 ** args.blocking_qubits) * (16 if args.precision == 'complex128' else 8) / (1024 ** 3)
        print(f" ⚠️  NON-STANDARD (default: 27, chunk size: {chunk_size_gb:.1f} GB)")
        print(f"  Note: Larger blocking = fewer, bigger chunks (experimental!)")
    else:
        print(" (NVIDIA standard)")
    print()
    
    # Show circuit preview
    print_circuit_preview(qubits_list, args.precision, args.blocking_qubits)
    
    # Run benchmarks
    all_results = []
    
    # If single qubit value and multiple GPU configs -> scaling test
    if len(qubits_list) == 1 and len(gpu_configs) > 1:
        results = run_scaling_benchmark(qubits_list[0], gpu_configs, args.shots, args.precision, gpu_memory, args.blocking_qubits)
        all_results.extend(results)
    
    # If multiple qubits and single GPU config -> circuit size test
    elif len(qubits_list) > 1 and len(gpu_configs) == 1:
        results = run_circuit_size_benchmark(qubits_list, gpu_configs[0], args.shots, args.precision, gpu_memory, args.blocking_qubits)
        all_results.extend(results)
    
    # Otherwise, comprehensive test
    else:
        for qubits in qubits_list:
            results = run_scaling_benchmark(qubits, gpu_configs, args.shots, args.precision, gpu_memory, args.blocking_qubits)
            all_results.extend(results)
    
    # Print comprehensive summary
    print_summary(all_results, vendor, gpu_model, args.precision, args.blocking_qubits)
    
    # Hardware-specific tips
    print("\n💡 Tips:")
    print("  - Multi-GPU requires circuits >= 32 qubits + shots >= 100")
    print(f"  - Blocking automatically enabled for 32+ qubit circuits (currently: {args.blocking_qubits} qubits)")
    if gpu_memory:
        print(f"  - Your GPUs: {gpu_memory:.0f} GB VRAM each")
        theoretical_max = calculate_max_qubits_multi_gpu(gpu_memory, gpu_count, args.precision)
        print(f"  - Theoretical capacity ({gpu_count} GPUs): ~{theoretical_max} qubits")
    print(f"  - complex64 provides ~2x speedup vs complex128")
    
    if args.blocking_qubits != 27:
        print(f"\n⚠️  Experimental Mode:")
        print(f"  - Using blocking_qubits={args.blocking_qubits} (non-standard)")
        print(f"  - Larger blocking = fewer chunks, may enable more qubits on fewer GPUs")
        print(f"  - NVIDIA standard is 27 for benchmarking comparisons")
        print(f"  - Try 28-29 to potentially run 35+ qubits on 8 GPUs")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
