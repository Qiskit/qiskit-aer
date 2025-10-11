#!/usr/bin/env python3
"""
Quick Multi-GPU Test for Qiskit Aer with ROCm
==============================================

Quick verification that multi-GPU support is working correctly.
Tests a simple configuration to confirm GPU detection and multi-GPU execution.

Usage:
    python3 quick_test.py

Expected output:
    - GPU detection confirmation
    - Multi-GPU execution confirmation
    - Performance comparison

Time: ~10-20 seconds
"""

import subprocess
import sys
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import quantum_volume


def show_gpu_info():
    """Display available GPUs"""
    print("="*70)
    print("AMD ROCm GPU Detection")
    print("="*70)
    
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_count = result.stdout.count('GPU[')
            print(f"✅ Detected {gpu_count} AMD GPU(s)")
            
            # Show first GPU info
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.split('\n') if 'GPU[0]' in l]
                if lines:
                    print(f"   {lines[0]}")
            
            return gpu_count
        else:
            print("⚠️  Could not detect GPUs via rocm-smi")
            return 0
    except Exception as e:
        print(f"⚠️  Error detecting GPUs: {e}")
        return 0


def test_single_gpu():
    """Test with single GPU"""
    print("\n" + "="*70)
    print("Test 1: Single GPU (28 qubits)")
    print("="*70)
    
    try:
        backend = AerSimulator(method='statevector', device='GPU')
        
        # Create 28-qubit circuit
        circuit = quantum_volume(28, depth=5, seed=42)
        circuit.measure_all()
        
        print("Running 28-qubit quantum volume circuit...")
        result = backend.run(circuit, shots=100, seed_simulator=42).result()
        
        metadata = result.results[0].metadata
        print(f"✅ Single GPU execution successful")
        print(f"   Memory used: {metadata.get('required_memory_mb', 0)} MB")
        
        return True
    except Exception as e:
        print(f"❌ Single GPU test failed: {e}")
        return False


def test_multi_gpu(num_gpus=2):
    """Test with multiple GPUs"""
    print("\n" + "="*70)
    print(f"Test 2: Multi-GPU ({num_gpus} GPUs, 32 qubits)")
    print("="*70)
    
    try:
        backend = AerSimulator(method='statevector', device='GPU')
        
        # Create 32-qubit circuit (requires multi-GPU)
        circuit = quantum_volume(32, depth=5, seed=42)
        circuit.measure_all()
        
        target_gpus = list(range(num_gpus))
        print(f"Configuration:")
        print(f"  Circuit: 32 qubits")
        print(f"  Target GPUs: {target_gpus}")
        print(f"  Blocking: 27 (2GB chunks)")
        print(f"  Expected chunks: 32")
        
        print("\nRunning...")
        result = backend.run(
            circuit,
            shots=100,
            seed_simulator=42,
            blocking_enable=True,
            blocking_qubits=27,
            target_gpus=target_gpus,
            batched_shots_gpu=True,
            batched_shots_gpu_max_qubits=32
        ).result()
        
        # Check multi-GPU usage
        metadata = result.results[0].metadata
        cacheblocking = metadata.get('cacheblocking', {})
        
        if cacheblocking.get('enabled'):
            gpus_used = cacheblocking.get('chunk_parallel_gpus', 1)
            block_bits = cacheblocking.get('block_bits', 0)
            
            print(f"\n✅ Multi-GPU execution successful")
            print(f"   GPUs used: {gpus_used}")
            print(f"   Block bits: {block_bits}")
            print(f"   Chunks created: {2 ** (32 - block_bits)}")
            
            if gpus_used >= num_gpus:
                print(f"   🎉 Multi-GPU is WORKING! ({gpus_used} GPUs)")
                return True
            else:
                print(f"   ⚠️  Expected {num_gpus} GPUs, but only {gpus_used} used")
                return False
        else:
            print(f"❌ Cache blocking not enabled")
            return False
            
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        return False


def main():
    """Run quick multi-GPU tests"""
    
    print("\n" + "="*70)
    print("QISKIT AER - QUICK MULTI-GPU TEST")
    print("="*70)
    print("\nThis quick test verifies multi-GPU functionality.")
    print("Time: ~10-20 seconds\n")
    
    # Detect GPUs
    gpu_count = show_gpu_info()
    
    if gpu_count == 0:
        print("\n❌ No GPUs detected. Cannot proceed.")
        return 1
    
    # Test single GPU
    single_ok = test_single_gpu()
    
    # Test multi-GPU if we have enough GPUs
    if gpu_count >= 2:
        multi_ok = test_multi_gpu(num_gpus=min(2, gpu_count))
    else:
        print("\n⚠️  Only 1 GPU available. Skipping multi-GPU test.")
        multi_ok = None
    
    # Summary
    print("\n" + "="*70)
    print("QUICK TEST SUMMARY")
    print("="*70)
    print(f"Single GPU (28q):  {'✅ PASS' if single_ok else '❌ FAIL'}")
    if multi_ok is not None:
        print(f"Multi-GPU (32q):   {'✅ PASS' if multi_ok else '❌ FAIL'}")
    else:
        print(f"Multi-GPU (32q):   ⚠️  SKIPPED (need 2+ GPUs)")
    print("="*70)
    
    if single_ok and (multi_ok or multi_ok is None):
        print("\n✅ Quick test completed successfully!")
        print("\nNext steps:")
        print("  - Run full benchmark: python3 multi_gpu/benchmark.py")
        print("  - Run validation test: python3 multi_gpu/validation.py")
        return 0
    else:
        print("\n❌ Some tests failed. Check your GPU configuration.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
