#!/usr/bin/env python3
"""
Multi-GPU Validation Test for Qiskit Aer with ROCm
===================================================

Validates multi-GPU functionality with blocking for large quantum circuits.

Key Findings:
1. Multi-GPU requires checking metadata.cacheblocking.chunk_parallel_gpus
2. Blocking required for 32+ qubits (single GPU limit at 31 qubits)
3. Keep chunks ≤256 to avoid HIP grid limits
4. Maximum blocking_qubits = 27 (2GB chunk limit for Aer)

Usage:
    # Run all default tests
    python3 validation.py
    
    # Run specific test
    python3 validation.py --qubits 32 --gpus 4
    
    # Quick test (fewer configurations)
    python3 validation.py --quick
    
    # Verbose output
    python3 validation.py --verbose

Requirements:
    - qiskit-aer-gpu-rocm
    - ROCm with rocm-smi
    - Multiple AMD GPUs
"""

import argparse
import os
import subprocess
import sys
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from qiskit import transpile, QuantumCircuit
from qiskit.circuit.library import quantum_volume
from qiskit_aer import AerSimulator


# Constants
DEFAULT_FALLBACK_GPU_COUNT = 8
DEFAULT_FALLBACK_GPU_MEMORY_GB = 192
MAX_BLOCKING_QUBITS = 27  # Hard limit: 16*2^27 = 2GB
BYTES_PER_COMPLEX128 = 16
GB_TO_BYTES = 1024 ** 3
MIN_CHUNKS_SHIFT = 8  # At least 256 chunks
TARGET_CHUNKS_PER_GPU = 16
ROCM_SMI_TIMEOUT = 10


@dataclass
class TestResult:
    """Structure for test results."""
    qubits: int
    gpus_requested: int
    gpus_used: int
    description: str
    success: bool
    execution_time: float = 0.0
    error_message: str = ""


def count_available_gpus_system() -> int:
    """
    Count available AMD GPUs in the system using rocm-smi.
    
    Returns:
        int: Number of GPUs detected, or fallback value if detection fails
    """
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True,
            text=True,
            timeout=ROCM_SMI_TIMEOUT
        )
        if result.returncode == 0:
            count = result.stdout.count('GPU[')
            return count if count > 0 else DEFAULT_FALLBACK_GPU_COUNT
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return DEFAULT_FALLBACK_GPU_COUNT


def count_available_gpus_aer() -> int:
    """
    Count GPUs visible to Qiskit Aer (respects ROCR_VISIBLE_DEVICES).
    
    This is the authoritative count that should be used for testing,
    as it reflects what Aer can actually use.
    
    Returns:
        int: Number of GPUs visible to Aer
    """
    try:
        # Create a test simulator to query GPU count
        sim = AerSimulator(method='statevector', device='GPU')
        
        # Try to get GPU count from backend properties
        # Aer stores this in the backend configuration
        config = sim.configuration()
        
        # Try different methods to get GPU count
        if hasattr(config, 'n_qubits'):
            # Some versions expose GPU count
            pass
        
        # Alternative: Check environment variable that Aer respects
        visible_devices = os.environ.get('ROCR_VISIBLE_DEVICES', '')
        if visible_devices:
            # Count comma-separated device IDs
            devices = [d.strip() for d in visible_devices.split(',') if d.strip()]
            if devices:
                return len(devices)
        
        # Fallback: Try small multi-GPU test to probe actual count
        # Try creating target_gpus lists of increasing size until error
        max_gpus = 1
        for gpu_count in [1, 2, 4, 8, 16, 32, 40]:
            try:
                qc = QuantumCircuit(2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure_all()
                
                # Try to run with this many GPUs
                result = sim.run(qc, shots=1, target_gpus=list(range(gpu_count))).result()
                max_gpus = gpu_count
            except Exception as e:
                if 'target_gpus has more GPUs' in str(e):
                    # Found the limit - return previous successful count
                    return max_gpus
                # Other error, continue
                continue
        
        # If we got here, all GPU counts worked - return the last one
        return max_gpus
        
    except Exception:
        pass
    
    # Final fallback: Assume 8 GPUs (conservative)
    return min(8, count_available_gpus_system())


def count_available_gpus() -> int:
    """
    Count available GPUs (uses Aer detection for accuracy).
    
    Returns:
        int: Number of GPUs available to Qiskit Aer
    """
    return count_available_gpus_aer()


def check_gpu_environment_variables(verbose: bool = False) -> Dict[str, str]:
    """
    Check environment variables that affect GPU visibility.
    
    Args:
        verbose: If True, print the variables
        
    Returns:
        Dictionary of relevant environment variables
    """
    env_vars = {
        'ROCR_VISIBLE_DEVICES': os.environ.get('ROCR_VISIBLE_DEVICES', 'not set'),
        'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', 'not set'),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
    }
    
    if verbose:
        print("\n  GPU Environment Variables:")
        for key, value in env_vars.items():
            if value != 'not set':
                print(f"    {key}={value}")
    
    return env_vars


def detect_gpu_memory() -> float:
    """
    Detect GPU memory in GB using rocm-smi.
    
    Returns:
        float: GPU memory in GB, or fallback value if detection fails
    """
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--csv'],
            capture_output=True,
            text=True,
            timeout=ROCM_SMI_TIMEOUT
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Total' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            return float(parts[1].strip()) / 1024
                        except ValueError:
                            continue
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return DEFAULT_FALLBACK_GPU_MEMORY_GB


def show_gpu_info(verbose: bool = False) -> Tuple[int, float]:
    """
    Display GPU information and return hardware specs.
    
    Args:
        verbose: If True, show detailed GPU information
        
    Returns:
        Tuple of (gpu_count_aer, gpu_memory_gb)
    """
    system_gpu_count = count_available_gpus_system()
    aer_gpu_count = count_available_gpus_aer()
    gpu_memory = detect_gpu_memory()
    
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True,
            text=True,
            timeout=ROCM_SMI_TIMEOUT
        )
        if result.returncode == 0 and verbose:
            lines = [
                line.strip() 
                for line in result.stdout.split('\n') 
                if 'GPU[' in line or 'Card series' in line or 'Card model' in line
            ]
            for line in lines[:5]:  # Show first 5 GPUs
                print(f"  {line}")
        
        print(f"  System GPUs (rocm-smi): {system_gpu_count}")
        print(f"  GPUs visible to Aer: {aer_gpu_count}")
        
        if aer_gpu_count < system_gpu_count:
            print(f"  ⚠️  Note: Aer sees fewer GPUs than system")
            env_vars = check_gpu_environment_variables(verbose=True)
            if env_vars.get('ROCR_VISIBLE_DEVICES', 'not set') == 'not set':
                print(f"  💡 Tip: Set ROCR_VISIBLE_DEVICES to control GPU visibility")
        elif verbose:
            check_gpu_environment_variables(verbose=True)
        
        print(f"  Memory per GPU: {gpu_memory:.1f} GB")
    except Exception as e:
        if verbose:
            print(f"  Could not get detailed GPU info: {e}")
        print(f"  Using defaults (GPUs: {aer_gpu_count}, Memory: {gpu_memory:.1f} GB)")
    
    return aer_gpu_count, gpu_memory


def calculate_optimal_blocking(num_qubits: int, num_gpus: int) -> Optional[int]:
    """
    Calculate optimal blocking_qubits for multi-GPU execution.
    
    The function ensures:
    1. Work is distributed across GPUs efficiently
    2. Aer chunk allocation stays ≤ 2GB (CRITICAL!)
    3. Memory constraints are respected
    
    Key empirical findings:
    - blocking=27 (2GB chunks): ✅ Works for 32q, 33q, 34q
    - blocking=28 (4GB chunks): ❌ Crashes for 34q
    
    Args:
        num_qubits: Number of qubits in the circuit
        num_gpus: Number of GPUs to distribute across
        
    Returns:
        Optimal blocking_qubits value, or None if blocking not needed
        
    Note:
        HARD LIMIT: blocking_qubits must be ≤27 (2GB max chunk size)
        Formula: 16 * 2^blocking_qubits ≤ 2GB
        Max blocking_qubits = log2(2GB / 16) = log2(128M) = 27
    """
    if num_qubits <= 31:
        return None  # No blocking needed for ≤31 qubits
    
    # Calculate desired chunks based on GPUs
    # Target: 16 chunks per GPU for optimal distribution
    desired_chunks = num_gpus * TARGET_CHUNKS_PER_GPU
    
    # blocking_qubits = num_qubits - log2(chunks)
    blocking_qubits = num_qubits - int(math.log2(desired_chunks))
    
    # MUST NOT EXCEED 27!
    blocking_qubits = min(blocking_qubits, MAX_BLOCKING_QUBITS)
    
    # Ensure we have at least 256 chunks (to avoid too few chunks)
    min_blocking = num_qubits - MIN_CHUNKS_SHIFT
    blocking_qubits = max(blocking_qubits, min_blocking)
    
    # Final safety check: ensure we're not violating the hard limit
    blocking_qubits = min(blocking_qubits, MAX_BLOCKING_QUBITS)
    
    return blocking_qubits


def test_multi_gpu(
    num_qubits: int,
    num_gpus: int,
    description: str,
    verbose: bool = False
) -> Tuple[Optional[bool], int, float, str]:
    """
    Test multi-GPU configuration with correct metadata checking.
    
    Args:
        num_qubits: Number of qubits in the circuit
        num_gpus: Number of GPUs to use
        description: Test description
        verbose: Enable verbose output
        
    Returns:
        Tuple of (success, gpus_used, execution_time, error_message)
    """
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    
    # Check if enough GPUs available
    available_gpus = count_available_gpus()
    if num_gpus > available_gpus:
        print(f"Configuration:")
        print(f"  Qubits: {num_qubits}")
        print(f"  GPUs requested: {num_gpus}")
        print(f"  GPUs available: {available_gpus}")
        print(f"\n⚠️  SKIPPED: Not enough GPUs")
        print(f"{'='*70}")
        return None, 0, 0.0, "Insufficient GPUs"
    
    blocking_qubits = calculate_optimal_blocking(num_qubits, num_gpus)
    
    state_size_gb = (BYTES_PER_COMPLEX128 * (2 ** num_qubits)) / GB_TO_BYTES
    
    print(f"Configuration:")
    print(f"  Qubits: {num_qubits}")
    print(f"  State size: {state_size_gb:.1f} GB")
    print(f"  GPUs requested: {num_gpus}")
    
    if blocking_qubits:
        num_chunks = 2 ** (num_qubits - blocking_qubits)
        chunk_size_gb = state_size_gb / num_chunks
        aer_alloc_gb = (BYTES_PER_COMPLEX128 * (2 ** blocking_qubits)) / GB_TO_BYTES
        
        print(f"  blocking_qubits: {blocking_qubits}")
        print(f"  Chunks: {num_chunks} ({chunk_size_gb:.2f} GB each)")
        print(f"  Aer allocation: {aer_alloc_gb:.2f} GB per chunk")
        print(f"  Chunks per GPU: {num_chunks / num_gpus:.0f}")
        
        # Warning if chunks might be too many
        if num_chunks > 256:
            print(f"  ⚠️  WARNING: {num_chunks} chunks may exceed HIP grid limits!")
            print(f"     Recommendation: Use more GPUs")
        elif verbose and num_chunks > 128:
            print(f"  Note: {num_chunks} chunks is approaching HIP limits")
    else:
        print(f"  No blocking (fits in single GPU)")
    
    try:
        # Create simulator
        sim = AerSimulator(method='statevector', device='GPU')
        
        # Create circuit
        qc = quantum_volume(num_qubits, depth=3, seed=42)
        qc.measure_all()
        transpiled = transpile(qc, sim, optimization_level=0)
        
        # Run options
        run_options: Dict = {'shots': 10, 'seed_simulator': 42}
        
        if blocking_qubits:
            run_options.update({
                'blocking_enable': True,
                'blocking_qubits': blocking_qubits,
                'target_gpus': list(range(num_gpus))
            })
        
        # Execute with timing
        print(f"\nExecuting...")
        start_time = time.time()
        result = sim.run(transpiled, **run_options).result()
        execution_time = time.time() - start_time
        
        # Get metadata
        metadata = result.results[0].metadata
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS (Execution time: {execution_time:.3f}s)")
        print(f"{'='*70}")
        
        # Check CORRECT metadata field
        cacheblocking = metadata.get('cacheblocking', {})
        gpus_used = 1
        
        if cacheblocking:
            gpus_used = cacheblocking.get('chunk_parallel_gpus', 1)
            enabled = cacheblocking.get('enabled', False)
            block_bits = cacheblocking.get('block_bits', 0)
            
            print(f"Multi-GPU Status:")
            print(f"  ✓ Cacheblocking enabled: {enabled}")
            print(f"  ✓ GPUs used: {gpus_used}")
            print(f"  ✓ Block bits: {block_bits}")
            
            if gpus_used > 1:
                print(f"\n  🎉 MULTI-GPU IS WORKING!")
            else:
                print(f"\n  ⚠️  Only 1 GPU used (Aer optimization)")
        else:
            print(f"Single GPU execution (no blocking)")
        
        # Memory info
        if verbose:
            max_gpu_mem = metadata.get('max_gpu_memory_mb', 0)
            required_mem = metadata.get('required_memory_mb', 0)
            print(f"\nMemory:")
            print(f"  Available GPU memory: {max_gpu_mem} MB")
            print(f"  Required: {required_mem} MB")
        
        print(f"{'='*70}")
        
        return True, gpus_used, execution_time, ""
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n{'='*70}")
        print(f"❌ FAILED")
        print(f"{'='*70}")
        print(f"Error: {error_msg[:200]}")
        if verbose and len(error_msg) > 200:
            print(f"Full error: {error_msg}")
        return False, 0, 0.0, error_msg[:100]


def get_test_configs(quick: bool = False) -> List[Tuple[int, int, str]]:
    """
    Get test configurations.
    
    Args:
        quick: If True, return minimal test set
        
    Returns:
        List of (qubits, gpus, description) tuples
    """
    if quick:
        return [
            (31, 1, "31q Single GPU - Maximum without Blocking"),
            (32, 2, "32q with 2 GPUs - blocking=27 (32 chunks × 2GB)"),
            (33, 4, "33q with 4 GPUs - blocking=27 (64 chunks × 2GB)"),
        ]
    
    return [
        # Single GPU - no blocking needed (up to 31 qubits)
        (30, 1, "30q Single GPU - No Blocking"),
        (31, 1, "31q Single GPU - Maximum without Blocking"),
        
        # Multi-GPU with blocking=27 (2GB chunks)
        (32, 2, "32q with 2 GPUs - blocking=27 (32 chunks × 2GB)"),
        (33, 4, "33q with 4 GPUs - blocking=27 (64 chunks × 2GB)"),
        
        # Larger circuits need more GPUs since blocking capped at 27
        (34, 8, "34q with 8 GPUs - blocking=27 (128 chunks × 2GB)"),
        (35, 16, "35q with 16 GPUs - blocking=27 (256 chunks × 2GB)"),
    ]


def print_summary(results: List[TestResult]) -> int:
    """
    Print test summary and return exit code.
    
    Args:
        results: List of test results
        
    Returns:
        Exit code (0 if all passed, 1 otherwise)
    """
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if not results:
        print("No tests were run.")
        return 1
    
    total_time = sum(r.execution_time for r in results)
    passed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success and r.success is not None)
    skipped = sum(1 for r in results if r.success is None)
    
    for r in results:
        if r.success is None:
            status = "⏭️  SKIP"
            gpu_info = f"({r.error_message})"
        elif r.success:
            status = "✅ PASS"
            gpu_info = f"(Used {r.gpus_used} GPU{'s' if r.gpus_used != 1 else ''}, {r.execution_time:.2f}s)"
        else:
            status = "❌ FAIL"
            gpu_info = f"({r.error_message[:30]}...)" if len(r.error_message) > 30 else f"({r.error_message})"
        
        print(f"{r.qubits:2d}q | {r.gpus_requested:2d} GPU{'s' if r.gpus_requested != 1 else ' '} | {status} {gpu_info}")
    
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total execution time: {total_time:.2f}s")
    print("="*70)
    
    return 0 if failed == 0 else 1


def main():
    """Run multi-GPU validation tests with CLI support."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU Validation Test for Qiskit Aer with ROCm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python3 validation.py
  
  # Quick test (minimal configurations)
  python3 validation.py --quick
  
  # Run specific test
  python3 validation.py --qubits 32 --gpus 4
  
  # Verbose output
  python3 validation.py --verbose
        """
    )
    
    parser.add_argument('--qubits', type=int,
                        help='Test specific number of qubits')
    parser.add_argument('--gpus', type=int,
                        help='Test with specific number of GPUs')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (fewer configurations)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("QISKIT AER MULTI-GPU VALIDATION TEST")
    print("="*70)
    
    print("\nROCm GPU Information:")
    print("-" * 70)
    available_gpus, gpu_memory = show_gpu_info(verbose=args.verbose)
    print("-" * 70)
    
    # Get test configurations
    if args.qubits and args.gpus:
        # Single custom test
        test_configs = [(args.qubits, args.gpus, f"{args.qubits}q with {args.gpus} GPUs")]
    else:
        test_configs = get_test_configs(quick=args.quick)
    
    # Run all tests
    results: List[TestResult] = []
    for num_qubits, num_gpus, description in test_configs:
        if num_gpus > available_gpus:
            print(f"\n⚠️  Skipping {description}: Need {num_gpus} GPUs, only {available_gpus} available")
            results.append(TestResult(
                qubits=num_qubits,
                gpus_requested=num_gpus,
                gpus_used=0,
                description=description,
                success=None,
                error_message="Insufficient GPUs"
            ))
            continue
        
        success, gpus_used, exec_time, error_msg = test_multi_gpu(
            num_qubits, num_gpus, description, verbose=args.verbose
        )
        
        results.append(TestResult(
            qubits=num_qubits,
            gpus_requested=num_gpus,
            gpus_used=gpus_used,
            description=description,
            success=success,
            execution_time=exec_time,
            error_message=error_msg
        ))
    
    # Print summary and exit
    exit_code = print_summary(results)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
