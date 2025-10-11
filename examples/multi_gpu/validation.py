#!/usr/bin/env python3
"""
FINAL WORKING Multi-GPU Test
=============================

Based on diagnostic findings:
1. Multi-GPU WAS working - we were checking wrong metadata field!
2. Must use blocking for 32+ qubits (single GPU limit)
3. Keep chunks ≤256 to avoid HIP grid limits
4. Check metadata.cacheblocking.chunk_parallel_gpus (not top-level)
"""

import subprocess
import math
from qiskit import transpile
from qiskit.circuit.library import quantum_volume
from qiskit_aer import AerSimulator


def count_available_gpus():
    """Count available GPUs"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            count = result.stdout.count('GPU[')
            return count if count > 0 else 8
    except Exception:
        pass
    return 8


def detect_gpu_memory():
    """Detect GPU memory in GB"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--csv'],
            capture_output=True, text=True, timeout=10
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
    except Exception:
        pass
    return 192


def show_gpu_info():
    """Display GPU information"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.split('\n') if 'GPU[' in line or 'Card series' in line or 'Card model' in line]
            for line in lines[:5]:  # Show first 5 GPUs
                print(f"  {line}")
        
        gpu_count = count_available_gpus()
        gpu_memory = detect_gpu_memory()
        print(f"  Total GPUs detected: {gpu_count}")
        print(f"  Memory per GPU: {gpu_memory:.1f} GB")
    except Exception as e:
        print(f"  Could not get GPU info: {e}")
        print(f"  Assuming MI300X defaults")


def calculate_optimal_blocking(num_qubits, num_gpus):
    """
    Calculate optimal blocking_qubits that:
    1. Distributes work across GPUs
    2. Keeps Aer chunk allocation ≤ 2GB (CRITICAL!)
    3. Respects memory constraints
    
    Key finding from tests:
    - blocking=27 (2GB chunks): ✅ Works for 32q, 33q
    - blocking=28 (4GB chunks): ❌ Crashes for 34q
    
    HARD LIMIT: blocking_qubits must be ≤27 (2GB max chunk size)
    """
    if num_qubits <= 31:
        return None  # No blocking needed
    
    # CRITICAL CONSTRAINT: Aer chunk allocation must be ≤ 2GB
    # Formula: 16 * 2^blocking_qubits ≤ 2GB
    # Max blocking_qubits = log2(2GB / 16) = log2(128M) = 27
    MAX_BLOCKING_QUBITS = 27  # Hard limit: 16*2^27 = 2GB
    
    # Calculate desired chunks based on GPUs
    # Target: 16-32 chunks per GPU
    desired_chunks = num_gpus * 16
    
    # blocking_qubits = num_qubits - log2(chunks)
    blocking_qubits = num_qubits - int(math.log2(desired_chunks))
    
    # MUST NOT EXCEED 27!
    blocking_qubits = min(blocking_qubits, MAX_BLOCKING_QUBITS)
    
    # Ensure we have at least some chunks
    # If blocking_qubits is too close to num_qubits, we get too few chunks
    min_blocking = num_qubits - 8  # At least 256 chunks
    blocking_qubits = max(blocking_qubits, min_blocking)
    
    # Safety check: ensure we're not violating the hard limit
    blocking_qubits = min(blocking_qubits, MAX_BLOCKING_QUBITS)
    
    return blocking_qubits


def count_available_gpus():
    """Count available GPUs"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.count('GPU[')
    except Exception:
        pass
    return 8  # Default fallback


def test_multi_gpu(num_qubits, num_gpus, description):
    """Test multi-GPU configuration with CORRECT metadata checking"""
    
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
        return None, 0
    
    blocking_qubits = calculate_optimal_blocking(num_qubits, num_gpus)
    
    state_size_gb = (16 * (2 ** num_qubits)) / (1024**3)
    
    print(f"Configuration:")
    print(f"  Qubits: {num_qubits}")
    print(f"  State size: {state_size_gb:.1f} GB")
    print(f"  GPUs requested: {num_gpus}")
    
    if blocking_qubits:
        num_chunks = 2 ** (num_qubits - blocking_qubits)
        chunk_size_gb = state_size_gb / num_chunks
        aer_alloc_gb = (16 * (2 ** blocking_qubits)) / (1024**3)
        
        print(f"  blocking_qubits: {blocking_qubits}")
        print(f"  Chunks: {num_chunks} ({chunk_size_gb:.2f} GB each)")
        print(f"  Aer allocation: {aer_alloc_gb:.2f} GB per chunk")
        print(f"  Chunks per GPU: {num_chunks / num_gpus:.0f}")
        
        # Warning if chunks might be too many
        if num_chunks > 64:
            print(f"  ⚠️  WARNING: {num_chunks} chunks may exceed HIP limits!")
            print(f"     Recommendation: Use more GPUs or larger blocking_qubits")
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
        run_options = {'shots': 10}
        
        if blocking_qubits:
            run_options['blocking_enable'] = True
            run_options['blocking_qubits'] = blocking_qubits
            run_options['target_gpus'] = list(range(num_gpus))
        
        # Execute
        print(f"\nExecuting...")
        result = sim.run(transpiled, **run_options).result()
        
        # Get metadata
        metadata = result.results[0].metadata
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS")
        print(f"{'='*70}")
        
        # Check CORRECT metadata field
        cacheblocking = metadata.get('cacheblocking', {})
        
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
        max_gpu_mem = metadata.get('max_gpu_memory_mb', 0)
        required_mem = metadata.get('required_memory_mb', 0)
        print(f"\nMemory:")
        print(f"  Available GPU memory: {max_gpu_mem} MB")
        print(f"  Required: {required_mem} MB")
        
        print(f"{'='*70}")
        
        return True, gpus_used if cacheblocking else 1
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ FAILED")
        print(f"{'='*70}")
        print(f"Error: {str(e)[:200]}")
        return False, 0


def main():
    """Run multi-GPU tests"""
    
    print("\n" + "="*70)
    print("QISKIT AER MULTI-GPU TEST - FINAL WORKING VERSION")
    print("="*70)
    
    print("\nROCm GPU Information:")
    print("-" * 70)
    show_gpu_info()
    print("-" * 70)
    
    available_gpus = count_available_gpus()
    
    # Test configurations: (qubits, gpus, description)
    test_configs = [
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
    
    # Run all tests
    results = []
    for num_qubits, num_gpus, description in test_configs:
        if num_gpus > available_gpus:
            print(f"\n⚠️  Skipping {description}: Need {num_gpus} GPUs, only {available_gpus} available")
            continue
        
        success, gpus_used = test_multi_gpu(num_qubits, num_gpus, description)
        results.append({
            'qubits': num_qubits,
            'gpus_requested': num_gpus,
            'gpus_used': gpus_used,
            'description': description,
            'success': success
        })
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        gpu_info = f"(Used {r['gpus_used']} GPU{'s' if r['gpus_used'] != 1 else ''})" if r['success'] else ""
        print(f"{r['qubits']:2d}q | {r['gpus_requested']:2d} GPU{'s' if r['gpus_requested'] != 1 else ' '} | {status} {gpu_info}")
    
    print("="*70)


if __name__ == '__main__':
    main()
