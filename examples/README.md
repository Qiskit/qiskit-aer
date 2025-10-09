# Qiskit Aer ROCm Examples

This directory contains example scripts demonstrating AMD GPU acceleration with ROCm.

## Quick Start

### 1. Quick GPU Test
A simple script to verify GPU functionality:

```bash
cd /home/ysha/playground/qiskit-aer/examples
python3 quick_gpu_test.py
```

**Output:**
- Confirms GPU is detected
- Runs a small benchmark comparing CPU vs GPU
- Shows speedup factor

**Time:** ~5-10 seconds

### 2. Comprehensive Benchmark
Detailed performance comparison across multiple circuit sizes:

```bash
python3 rocm_gpu_benchmark.py
```

**Features:**
- Memory requirements table
- Performance comparison for 10, 15, 20, 25 qubit circuits
- Detailed 20-qubit example with measurement outcomes
- Average speedup calculation
- Performance optimization tips

**Time:** ~2-3 minutes

## Expected Performance

### Speedup Factors (Typical)

| Qubits | Circuit Depth | Expected Speedup | Notes |
|--------|---------------|------------------|-------|
| 10 | ~50 | 1-2x | CPU overhead dominates |
| 15 | ~70 | 2-4x | GPU starts to show benefit |
| 20 | ~100 | 5-10x | GPU clearly faster |
| 25 | ~120 | 10-20x | Excellent GPU utilization |
| 28 | ~140 | 15-30x | Near memory limits |

*Note: Actual speedup depends on GPU model, circuit structure, and number of shots.*

### AMD GPU Performance

| GPU | Memory | Max Qubits* | Expected Performance |
|-----|--------|-------------|---------------------|
| MI300X | 192 GB | 28 | Best performance |
| MI250X | 128 GB | 27 | Excellent |
| MI210 | 64 GB | 26 | Very good |
| MI100 | 32 GB | 25 | Good |
| RX 7900 XTX | 24 GB | 24-25 | Good for consumer GPU |

*With `blocking_enable=True`

## Running from Source Directory

**Important:** If running from the qiskit-aer source directory, use this approach:

```bash
# Change to a different directory first
cd /tmp

# Then run with full path
python3 /home/ysha/playground/qiskit-aer/examples/quick_gpu_test.py
```

Or use the installed package:

```bash
# Activate your virtual environment
source ~/amd/qiskit-aer/venv/bin/activate

# Change directory
cd ~

# Run the script
python3 /home/ysha/playground/qiskit-aer/examples/quick_gpu_test.py
```

## Custom Examples

### Example 1: Statevector Simulation

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create circuit
qc = QuantumCircuit(25)
qc.h(range(25))
qc.cx(0, range(1, 25))
qc.measure_all()

# Simulate on GPU
sim = AerSimulator(device='GPU', method='statevector')
result = sim.run(qc, shots=1000).result()
print(result.get_counts())
```

### Example 2: Large Circuit with Memory Blocking

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create large circuit
qc = QuantumCircuit(27)
# ... build your circuit ...

# Use blocking for memory efficiency
sim = AerSimulator(device='GPU', method='statevector')
result = sim.run(qc, 
                 blocking_enable=True,
                 blocking_qubits=27,  # Adjust based on GPU memory
                 shots=1000).result()
```

### Example 3: Density Matrix Simulation

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Density matrix method (useful for noise simulation)
qc = QuantumCircuit(15)
qc.h(range(15))
qc.measure_all()

sim = AerSimulator(device='GPU', method='density_matrix')
result = sim.run(qc, shots=1000).result()
```

## Troubleshooting

### GPU Not Detected

```python
from qiskit_aer import AerSimulator
sim = AerSimulator()
print("Devices:", sim.available_devices())
```

If GPU is missing:
1. Check ROCm installation: `rocminfo`
2. Verify GPU is visible: `/opt/rocm/bin/rocm-smi`
3. Check environment: `echo $ROCM_PATH`
4. Rebuild qiskit-aer with ROCm support

### Import Errors

If you get `ModuleNotFoundError: No module named 'qiskit_aer.backends.controller_wrappers'`:
- You're in the source directory
- Run from a different directory (see above)

### Out of Memory Errors

If simulation fails with memory errors:
- Reduce number of qubits
- Enable memory blocking:
  ```python
  result = sim.run(qc, blocking_enable=True, blocking_qubits=25)
  ```
- Use fewer shots

### Poor Performance

If GPU is slower than expected:
- Try larger circuits (25+ qubits)
- Increase number of shots (more parallelism)
- Check GPU is not being used by other processes: `rocm-smi`

## Performance Tips

1. **Circuit Size**: GPU acceleration is most beneficial for ≥20 qubits
2. **Batch Jobs**: Run multiple circuits in parallel
3. **Memory Management**: Use `blocking_enable=True` for large circuits
4. **Shot Count**: Higher shot counts (≥1000) improve GPU utilization
5. **Method Selection**: 
   - `statevector`: Best for pure state simulation
   - `density_matrix`: For noisy circuits
   - `automatic`: Let Aer choose

## Additional Resources

- [Qiskit Aer Documentation](https://qiskit.org/ecosystem/aer/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Building with ROCm](../BUILDING_ROCM.md)

## Contributing Examples

Have a great example? Submit a PR with:
1. Working code
2. Clear comments
3. Expected output
4. Performance notes
