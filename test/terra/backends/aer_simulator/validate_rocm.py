#!/usr/bin/env python3
"""
ROCm GPU Validation Script

This script validates that the ROCm GPU build is working correctly
and tests various GPU capabilities.

Usage:
    python validate_rocm.py [--verbose]
"""

import argparse
import sys
from typing import List, Tuple

# Test results
tests_passed = []
tests_failed = []


def test_import() -> bool:
    """Test that qiskit_aer can be imported."""
    print("Test 1: Import qiskit_aer...", end=" ")
    try:
        import qiskit_aer
        print(f"✓ (version {qiskit_aer.__version__})")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_aer_simulator() -> bool:
    """Test that AerSimulator can be created."""
    print("Test 2: Create AerSimulator...", end=" ")
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        print(f"✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_gpu_detection() -> bool:
    """Test GPU device detection."""
    print("Test 3: GPU device detection...", end=" ")
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(device='GPU')
        devices = sim.available_devices()
        if 'GPU' in devices:
            print(f"✓ (devices: {devices})")
            return True
        else:
            print(f"✗ GPU not in available devices: {devices}")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_simple_circuit_cpu() -> bool:
    """Test running a simple circuit on CPU."""
    print("Test 4: Simple circuit on CPU...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        sim = AerSimulator(device='CPU')
        result = sim.run(qc, shots=100).result()
        
        if result.success:
            counts = result.get_counts()
            print(f"✓ (counts: {counts})")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_simple_circuit_gpu() -> bool:
    """Test running a simple circuit on GPU."""
    print("Test 5: Simple circuit on GPU...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        sim = AerSimulator(method='statevector', device='GPU')
        result = sim.run(qc, shots=100).result()
        
        if result.success:
            counts = result.get_counts()
            print(f"✓ (counts: {counts})")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_larger_circuit_gpu() -> bool:
    """Test running a larger circuit on GPU."""
    print("Test 6: Larger circuit (20 qubits) on GPU...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(20)
        qc.h(range(20))
        qc.measure_all()
        
        sim = AerSimulator(method='statevector', device='GPU')
        result = sim.run(qc, shots=10).result()
        
        if result.success:
            print(f"✓")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_statevector_method() -> bool:
    """Test statevector simulation method."""
    print("Test 7: Statevector method on GPU...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(10)
        qc.h(range(10))
        
        sim = AerSimulator(method='statevector', device='GPU')
        result = sim.run(qc, shots=1).result()
        
        if result.success:
            statevector = result.get_statevector()
            print(f"✓ (dim: {len(statevector)})")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_density_matrix_method() -> bool:
    """Test density matrix simulation method."""
    print("Test 8: Density matrix method on GPU...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(8)
        qc.h(range(8))
        
        sim = AerSimulator(method='density_matrix', device='GPU')
        result = sim.run(qc, shots=1).result()
        
        if result.success:
            print(f"✓")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_blocking_enable() -> bool:
    """Test GPU memory blocking feature."""
    print("Test 9: GPU blocking (chunking) feature...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(25)
        qc.h(range(25))
        qc.measure_all()
        
        sim = AerSimulator(method='statevector', device='GPU')
        result = sim.run(qc, 
                        shots=10,
                        blocking_enable=True,
                        blocking_qubits=23).result()
        
        if result.success:
            print(f"✓")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_metadata() -> bool:
    """Test that GPU metadata is present in results."""
    print("Test 10: GPU metadata in results...", end=" ")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        sim = AerSimulator(method='statevector', device='GPU')
        result = sim.run(qc, shots=10).result()
        
        if result.success:
            metadata = result.to_dict().get('metadata', {})
            print(f"✓ (metadata keys: {list(metadata.keys())})")
            return True
        else:
            print(f"✗ Simulation failed")
            return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    import platform
    import subprocess
    
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    
    # ROCm version
    try:
        with open('/opt/rocm/.info/version', 'r') as f:
            print(f"ROCm version: {f.read().strip()}")
    except:
        print("ROCm version: unknown")
    
    # GPU info
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"GPU: {result.stdout.strip()}")
    except:
        print("GPU: unknown")
    
    # GPU architecture
    try:
        result = subprocess.run(
            ['rocminfo'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Name:' in line and 'gfx' in line:
                    print(f"GPU architecture: {line.split()[1]}")
                    break
    except:
        pass
    
    # Qiskit Aer version
    try:
        import qiskit_aer
        print(f"Qiskit Aer: {qiskit_aer.__version__}")
    except:
        print("Qiskit Aer: not installed")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate ROCm GPU build")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║       Qiskit Aer ROCm GPU Validation Suite            ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    print_system_info()
    
    print("Running validation tests...\n")
    
    # Run all tests
    tests = [
        ("Import", test_import),
        ("AerSimulator", test_aer_simulator),
        ("GPU Detection", test_gpu_detection),
        ("CPU Simple Circuit", test_simple_circuit_cpu),
        ("GPU Simple Circuit", test_simple_circuit_gpu),
        ("GPU Large Circuit", test_larger_circuit_gpu),
        ("GPU Statevector", test_statevector_method),
        ("GPU Density Matrix", test_density_matrix_method),
        ("GPU Blocking", test_blocking_enable),
        ("GPU Metadata", test_metadata),
    ]
    
    for name, test_func in tests:
        try:
            if test_func():
                tests_passed.append(name)
            else:
                tests_failed.append(name)
        except Exception as e:
            print(f"  Exception in {name}: {e}")
            tests_failed.append(name)
    
    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"✓ Passed: {len(tests_passed)}/{len(tests)}")
    print(f"✗ Failed: {len(tests_failed)}/{len(tests)}")
    
    if tests_failed:
        print("\nFailed tests:")
        for test in tests_failed:
            print(f"  - {test}")
        print("\nPlease check:")
        print("  1. ROCm is properly installed")
        print("  2. AMD GPU is available and drivers are loaded")
        print("  3. qiskit-aer-gpu-rocm was built correctly")
        sys.exit(1)
    else:
        print("\n✓ All validation tests passed!")
        print("✓ ROCm GPU support is working correctly")
        sys.exit(0)


if __name__ == '__main__':
    main()
