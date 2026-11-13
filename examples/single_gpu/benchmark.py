#!/usr/bin/env python3
"""
Single GPU Benchmark for Qiskit Aer with ROCm
NVIDIA-Aligned, Production Ready

Comprehensive single GPU performance analysis using industry-standard
Quantum Volume circuits. Supports precision selection (complex128/complex64)
for memory vs accuracy trade-offs.

Features:
    - Quantum Volume circuits (NVIDIA cuQuantum aligned)
    - Precision selection: complex128 (default) or complex64
    - Memory-aware adaptive qubit range
    - CPU baseline comparison
    - Detailed performance analysis

Requirements:
    - qiskit-aer-gpu-rocm installed
    - AMD GPU with ROCm support

Usage:
    # Default (complex128, auto qubit range)
    python3 benchmark.py
    
    # Use complex64 for larger circuits
    python3 benchmark.py --precision complex64
    
    # Custom qubit range
    python3 benchmark.py --qubits 20,25,30,32
    
    # Custom shots
    python3 benchmark.py --shots 2048
    
    # Combination
    python3 benchmark.py --precision complex64 --qubits 20,25,30,33,34 --shots 1024

Time: ~5-15 minutes (depending on qubit range and GPU)
"""

import argparse
import subprocess
import sys
import time
from qiskit_aer import AerSimulator
from qiskit.circuit.library import quantum_volume
import numpy as np


def get_gpu_memory():
    """
    Get GPU memory in GB using rocm-smi.
    
    Returns:
        float: GPU memory in GB, or None if detection fails
    """
    try:
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
    except Exception as e:
        print(f"Warning: Could not detect GPU memory: {e}")
    return None


def get_gpu_model():
    """Get GPU model name"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'GPU[0]' in line:
                    return line.split(':')[-1].strip()
    except Exception:
        pass
    return "Unknown"


def calculate_max_qubits(gpu_memory_gb, precision='complex128'):
    """
    Calculate theoretical maximum qubits based on GPU memory and precision.
    
    This is a conservative estimate. Actual limits may vary based on:
    - GPU architecture (MI300X, MI350, MI450, etc.)
    - Blocking overhead and framework limitations
    - Circuit complexity and gate count
    - System configuration and other processes
    
    For production use, always validate on your specific hardware.
    
    Args:
        gpu_memory_gb: GPU memory in GB
        precision: 'complex128' (16 bytes) or 'complex64' (8 bytes)
    
    Returns:
        int: Estimated maximum number of qubits (conservative)
    """
    if gpu_memory_gb is None:
        return 28 if precision == 'complex128' else 29
    
    # Bytes per complex number
    bytes_per_element = 16 if precision == 'complex128' else 8
    
    # Use 50% of GPU memory for state vector (rest for operations)
    # This is conservative to account for blocking overhead
    usable_memory_bytes = (gpu_memory_gb * 0.5) * (1024 ** 3)
    
    # 2^n * bytes_per_element = usable_memory_bytes
    max_qubits = int(np.log2(usable_memory_bytes / bytes_per_element))
    
    # Return conservative estimate
    # Note: Actual limits should be validated on target hardware
    return min(max(max_qubits, 25), 40)  # Cap at 40 for safety


def show_system_info(precision='complex128'):
    """Display system information and capabilities"""
    print("="*70)
    print("SINGLE GPU BENCHMARK - NVIDIA-ALIGNED")
    print("="*70)
    print()
    
    # Precision info
    bytes_per_element = 16 if precision == 'complex128' else 8
    print(f"Precision: {precision} ({bytes_per_element} bytes per complex number)")
    if precision == 'complex64':
        print("  ⚠️  Note: complex64 uses half memory but reduced accuracy")
    print()
    
    # Check GPU
    sim = AerSimulator()
    devices = sim.available_devices()
    
    if 'GPU' not in devices:
        print("✗ No GPU detected. Make sure ROCm is properly installed.")
        return False, None, None
    
    try:
        gpu_sim = AerSimulator(device='GPU')
        print("✓ GPU acceleration available")
        
        # GPU model
        gpu_model = get_gpu_model()
        print(f"✓ GPU Model: {gpu_model}")
        
        # GPU memory
        gpu_memory = get_gpu_memory()
        if gpu_memory:
            print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
            max_qubits = calculate_max_qubits(gpu_memory, precision)
            # Only show theoretical calculation, no specific recommendations
            # Actual limits are hardware and workload dependent
        else:
            print("⚠️  Could not detect GPU memory, using conservative limits")
            max_qubits = 28 if precision == 'complex128' else 29
        
        return True, max_qubits, gpu_memory
        
    except Exception as e:
        print(f"✗ GPU initialization failed: {e}")
        return False, None, None


def get_circuit_specs(qubits, precision='complex128'):
    """
    Get circuit specifications including memory requirements.
    
    Args:
        qubits: Number of qubits
        precision: 'complex128' or 'complex64'
    
    Returns:
        dict: Circuit specifications
    """
    # Create circuit to analyze
    circuit = quantum_volume(qubits, depth=5, seed=42)
    circuit.measure_all()
    
    # Gate statistics
    gate_counts = circuit.count_ops()
    total_gates = sum(gate_counts.values())
    
    # Memory calculations
    bytes_per_element = 16 if precision == 'complex128' else 8
    state_vector_bytes = (2 ** qubits) * bytes_per_element
    
    # Overhead estimates
    # - Operation buffer: ~20% of state vector
    # - Framework overhead: ~10%
    # - Total: ~1.3x multiplier
    overhead_multiplier = 1.3
    total_memory_bytes = state_vector_bytes * overhead_multiplier
    
    # Blocking info (for GPU)
    blocking_enabled = qubits >= 32
    if blocking_enabled:
        blocking_qubits = 27
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
        'total_memory_gb': total_memory_bytes / (1024 ** 3),
        'blocking_enabled': blocking_enabled,
        'chunk_size_gb': chunk_size_bytes / (1024 ** 3),
        'num_chunks': num_chunks
    }


def run_benchmark(device, qubits, shots=100, precision='complex128'):
    """
    Run benchmark on specified device.
    
    Args:
        device: 'CPU' or 'GPU'
        qubits: Number of qubits
        shots: Number of measurement shots
        precision: 'complex128' or 'complex64'
    
    Returns:
        tuple: (time, result, error, circuit_specs)
    """
    try:
        # Convert user-friendly precision to Qiskit Aer format
        # complex128 (16 bytes) -> 'double', complex64 (8 bytes) -> 'single'
        aer_precision = 'double' if precision == 'complex128' else 'single'
        
        # Create simulator
        if device == 'GPU':
            backend = AerSimulator(method='statevector', device='GPU', precision=aer_precision)
        else:
            backend = AerSimulator(method='statevector', device='CPU', precision=aer_precision)
        
        # Create quantum volume circuit
        circuit = quantum_volume(qubits, depth=5, seed=42)
        circuit.measure_all()
        
        # Get circuit specs
        specs = get_circuit_specs(qubits, precision)
        
        # Configure run options
        run_options = {'shots': shots, 'seed_simulator': 42}
        
        # Enable blocking for large circuits
        if device == 'GPU' and qubits >= 32:
            run_options['blocking_enable'] = True
            run_options['blocking_qubits'] = 27  # NVIDIA standard
        
        # Execute
        start = time.time()
        result = backend.run(circuit, **run_options).result()
        elapsed = time.time() - start
        
        return elapsed, result, None, specs
        
    except Exception as e:
        error_msg = str(e)
        if 'bad_alloc' in error_msg or 'out of memory' in error_msg.lower():
            error_msg = "Out of memory - circuit too large for GPU"
        elif 'hipError' in error_msg:
            error_msg = "GPU error"
        elif 'runtime_error' in error_msg.lower():
            error_msg = "Circuit too large for available GPU memory"
        elif 'std::bad_alloc' in error_msg:
            error_msg = "Memory allocation failed - reduce circuit size"
        else:
            error_msg = error_msg[:60]
        return None, None, error_msg, None


def print_circuit_preview(qubits_list, precision='complex128'):
    """Print preview of circuits to be tested"""
    print("\n" + "="*90)
    print("CIRCUIT SPECIFICATIONS PREVIEW")
    print("="*90)
    print(f"{'Qubits':<8} {'Depth':<8} {'Gates':<8} {'State Vec':<12} {'Total Mem':<12} {'Blocking':<15}")
    print("-"*90)
    
    for qubits in qubits_list[:10]:  # Limit to first 10 to avoid clutter
        specs = get_circuit_specs(qubits, precision)
        blocking_info = f"{specs['num_chunks']} chunks" if specs['blocking_enabled'] else "No"
        print(f"{qubits:<8} {specs['depth']:<8} {specs['total_gates']:<8} "
              f"{specs['state_vector_gb']:>10.2f} GB {specs['total_memory_gb']:>10.2f} GB {blocking_info:<15}")
    
    if len(qubits_list) > 10:
        print(f"... and {len(qubits_list) - 10} more circuits")
    print("-"*90)
    print()


def run_comparison(qubits_list, shots=100, precision='complex128', max_qubits=None):
    """Run CPU vs GPU comparison with detailed circuit info"""
    # Show circuit preview first
    print_circuit_preview(qubits_list, precision)
    
    print(f"Running benchmarks ({precision}, {shots} shots)...")
    print()
    print("-" * 105)
    print(f"{'Qubits':<8} {'Depth':<8} {'Gates':<8} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12} {'Memory':<12}")
    print("-" * 105)
    
    results = []
    
    for qubits in qubits_list:
        # Pre-check if qubits exceed reasonable theoretical limit
        # Actual safe limits are hardware-dependent
        if max_qubits and qubits > max_qubits + 2:
            print(f"{qubits:<8} {'-':<8} {'-':<8} {'Skip':<12} {'Skip':<12} {'-':<12} {'-':<12} ❌ Over theoretical limit")
            continue
        
        # Get circuit specs
        specs = get_circuit_specs(qubits, precision)
        depth = specs['depth']
        gates = specs['total_gates']
        mem_str = f"{specs['state_vector_gb']:.2f} GB"
        
        # Skip CPU for large circuits
        if qubits > 30:
            gpu_time, _, gpu_error, _ = run_benchmark('GPU', qubits, shots, precision)
            
            if gpu_error:
                print(f"{qubits:<8} {depth:<8} {gates:<8} {'Skip':<12} {'Failed':<12} {'-':<12} {mem_str:<12} ❌ {gpu_error[:20]}")
                continue
            
            print(f"{qubits:<8} {depth:<8} {gates:<8} {'Skip':<12} {gpu_time:<12.4f} {'GPU only':<12} {mem_str:<12} ✓")
            results.append({
                'qubits': qubits,
                'gpu_time': gpu_time,
                'state_gb': specs['state_vector_gb'],
                'specs': specs
            })
        else:
            # Run both CPU and GPU
            cpu_time, _, cpu_error, _ = run_benchmark('CPU', qubits, shots, precision)
            if cpu_error:
                continue
            
            gpu_time, _, gpu_error, _ = run_benchmark('GPU', qubits, shots, precision)
            if gpu_error:
                print(f"{qubits:<8} {depth:<8} {gates:<8} {cpu_time:<12.4f} {'Failed':<12} {'-':<12} {mem_str:<12} ❌")
                continue
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            status = "✓" if speedup > 1 else "⚠️"
            
            print(f"{qubits:<8} {depth:<8} {gates:<8} {cpu_time:<12.4f} {gpu_time:<12.4f} {speedup:<12.2f}x {mem_str:<12} {status}")
            
            results.append({
                'qubits': qubits,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'state_gb': specs['state_vector_gb'],
                'specs': specs
            })
    
    print("-" * 105)
    return results


def print_summary(results, precision='complex128'):
    """Print benchmark summary with circuit specifications"""
    if not results:
        return
    
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print()
    
    # Circuit specifications summary
    print("Circuit Type: Quantum Volume (NVIDIA cuQuantum Standard)")
    print(f"Precision: {precision} ({'16 bytes' if precision == 'complex128' else '8 bytes'} per element)")
    print(f"Total circuits tested: {len(results)}")
    
    if results and 'specs' in results[0]:
        total_gates = sum(r['specs']['total_gates'] for r in results)
        avg_depth = np.mean([r['specs']['depth'] for r in results])
        print(f"Total gates executed: {total_gates:,}")
        print(f"Average circuit depth: {avg_depth:.1f}")
    
    # Performance statistics
    speedup_results = [r for r in results if 'speedup' in r]
    if speedup_results:
        print(f"\n{'Performance Metrics:'}")
        avg_speedup = np.mean([r['speedup'] for r in speedup_results])
        max_speedup = max(r['speedup'] for r in speedup_results)
        best_qubits = [r['qubits'] for r in speedup_results if r['speedup'] == max_speedup][0]
        
        print(f"  CPU vs GPU comparisons: {len(speedup_results)}")
        print(f"  Average GPU speedup: {avg_speedup:.2f}x")
        print(f"  Best speedup: {max_speedup:.2f}x (at {best_qubits} qubits)")
        
        # Time saved
        total_cpu_time = sum(r['cpu_time'] for r in speedup_results)
        total_gpu_time = sum(r['gpu_time'] for r in speedup_results)
        time_saved = total_cpu_time - total_gpu_time
        print(f"  Total time saved: {time_saved:.2f}s ({(time_saved/total_cpu_time*100):.1f}%)")
    
    # GPU-only results
    gpu_only = [r for r in results if 'speedup' not in r]
    if gpu_only:
        print(f"\n{'Large Circuits (GPU only):'}")
        print(f"  Count: {len(gpu_only)}")
        largest = max(r['qubits'] for r in gpu_only)
        fastest = min(r['gpu_time'] for r in gpu_only)
        slowest = max(r['gpu_time'] for r in gpu_only)
        print(f"  Largest: {largest} qubits ({max(r['state_gb'] for r in gpu_only):.1f} GB state)")
        print(f"  Execution time range: {fastest:.2f}s - {slowest:.2f}s")
    
    # Memory analysis
    if results:
        print(f"\n{'Memory Analysis:'}")
        max_mem = max(r['state_gb'] for r in results)
        total_mem = sum(r['state_gb'] for r in results)
        print(f"  Peak state vector: {max_mem:.2f} GB")
        print(f"  Total state vectors: {total_mem:.2f} GB")
        
        # Blocking info
        if any('specs' in r and r['specs']['blocking_enabled'] for r in results):
            blocking_circuits = [r for r in results if 'specs' in r and r['specs']['blocking_enabled']]
            print(f"  Blocking used: {len(blocking_circuits)} circuits")
            max_chunks = max(r['specs']['num_chunks'] for r in blocking_circuits)
            print(f"  Max chunks: {max_chunks} (2 GB each)")
    
    # Precision benefits
    if precision == 'complex64':
        print(f"\n{'complex64 Benefits:'}")
        print(f"  ✓ 2x memory reduction vs complex128")
        print(f"  ✓ Enables +1 qubit on same hardware")
        print(f"  ✓ ~1.5-2x performance improvement")
        print(f"  ⚠️  Reduced accuracy (acceptable for algorithm development)")
    
    print("\n" + "="*90)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Single GPU Benchmark (NVIDIA-Aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: complex128, auto qubit range
  python3 benchmark.py
  
  # Use complex64 for larger circuits
  python3 benchmark.py --precision complex64
  
  # Custom qubit range
  python3 benchmark.py --qubits 20,25,30,32
  
  # High shot count for accuracy
  python3 benchmark.py --shots 2048
  
  # Combination
  python3 benchmark.py --precision complex64 --qubits 25,30,33,35 --shots 1024
        """
    )
    
    parser.add_argument('--precision', type=str, default='complex128',
                        choices=['complex128', 'complex64'],
                        help='Precision: complex128 (default, 16 bytes) or complex64 (8 bytes)')
    
    parser.add_argument('--qubits', type=str, default=None,
                        help='Comma-separated qubit counts (e.g., "20,25,30,32")')
    
    parser.add_argument('--shots', type=int, default=100,
                        help='Number of measurement shots (default: 100)')
    
    args = parser.parse_args()
    
    # Show system info
    has_gpu, max_qubits, gpu_memory = show_system_info(args.precision)
    
    if not has_gpu:
        print("\nCannot run benchmark without GPU. Exiting.")
        return 1
    
    print()
    
    # Determine qubit list
    if args.qubits:
        qubits_list = [int(q.strip()) for q in args.qubits.split(',')]
        
        # Check if any requested qubits significantly exceed theoretical capacity
        if max_qubits:
            over_limit = [q for q in qubits_list if q > max_qubits + 2]
            if over_limit:
                print(f"⚠️  WARNING: Qubits {over_limit} significantly exceed theoretical capacity")
                print(f"   GPU Memory: {gpu_memory:.1f} GB")
                print(f"   Theoretical estimate: ~{max_qubits} qubits ({args.precision})")
                print(f"   These circuits may fail due to insufficient memory.")
                print(f"   Note: Actual limits vary by GPU architecture and workload")
                print()
                
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    print("\nBenchmark cancelled. Adjust --qubits to stay within capacity.")
                    return 1
                print()
    else:
        # Auto-generate based on GPU capability
        if max_qubits and max_qubits >= 32:
            qubits_list = [20, 25, 28, 30, 32]
            if max_qubits >= 33:
                qubits_list.append(33)
            if max_qubits >= 34:
                qubits_list.append(34)
        elif max_qubits and max_qubits >= 30:
            qubits_list = [20, 25, 28, 30, 32]
        else:
            qubits_list = [20, 25, 28]
    
    print(f"Testing qubits: {qubits_list}")
    print(f"Estimated max: {max_qubits} qubits ({args.precision})")
    print()
    
    # Run benchmarks
    results = run_comparison(qubits_list, args.shots, args.precision, max_qubits)
    
    # Print summary
    print_summary(results, args.precision)
    
    # Tips - hardware agnostic
    print("\n💡 Tips:")
    print("  - GPU shows best speedup for 23+ qubit circuits")
    if gpu_memory:
        print(f"  - Your GPU: {gpu_memory:.0f} GB VRAM ({get_gpu_model()})")
        print(f"  - Theoretical capacity: ~{max_qubits} qubits ({args.precision})")
    print(f"  - complex64 provides ~2x speedup vs complex128")
    if args.precision == 'complex128':
        print(f"  - Use --precision complex64 for faster execution")
    print(f"  - Always validate limits on your specific hardware")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
