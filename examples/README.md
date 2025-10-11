# Qiskit Aer ROCm Examples# Qiskit Aer ROCm Examples



**Last Updated:** October 10, 2025  **Last Updated:** October 10, 2025  

**Status:** ✅ Production Ready (Validated on AMD MI300X)**Status:** ✅ Production Ready (Validated on AMD MI300X)



Organized examples demonstrating AMD GPU acceleration with ROCm for quantum circuit simulation.This directory contains validated example scripts demonstrating AMD GPU acceleration with ROCm.



------



## 📁 Directory Structure## 🚀 Production Scripts



```### 1. Quick GPU Test

examples/A simple script to verify GPU functionality:

├── single_gpu/          ← Single GPU examples (up to 31 qubits)

│   ├── quick_test.py       Fast GPU verification (~10 sec)```bash

│   ├── benchmark.py        Comprehensive benchmarking (~2-3 min)cd /home/ysha/playground/qiskit-aer/examples

│   └── README.md           Single GPU documentationpython3 quick_gpu_test.py

│```

├── multi_gpu/           ← Multi-GPU examples (32-40 qubits) ⭐

│   ├── quick_test.py       Fast multi-GPU verification (~20 sec)**Output:**

│   ├── benchmark.py        Scaling & performance analysis (~5-10 min)- Confirms GPU is detected

│   ├── validation.py       Complete validation suite (~2-5 min)- Runs a small benchmark comparing CPU vs GPU

│   └── README.md           Multi-GPU documentation- Shows speedup factor

│

└── archive/             ← Historical versions and old scripts**Time:** ~5-10 seconds

    ├── test_*.py

    └── docs/### 2. ROCm GPU Benchmark

```Detailed performance comparison across multiple circuit sizes:



---```bash

python3 rocm_gpu_benchmark.py

## 🚀 Quick Start```



### Step 1: Verify Single GPU**Features:**

```bash- Memory requirements table

cd single_gpu- Performance comparison for 10, 15, 20, 25 qubit circuits

python3 quick_test.py- Detailed 20-qubit example with measurement outcomes

```- Average speedup calculation

- Performance optimization tips

**Expected:** GPU detection and basic performance test (~10 seconds)

**Time:** ~2-3 minutes

### Step 2: Test Multi-GPU (if available)

```bash### 3. Multi-GPU Final Working Test ✨ NEW

cd multi_gpu**Complete multi-GPU validation test (30-35 qubits):**

python3 quick_test.py

``````bash

python3 test_multi_gpu_final_working.py

**Expected:** Multi-GPU confirmation with 32-qubit circuit (~20 seconds)```



### Step 3: Run Full Validation (recommended)**Features:**

```bash- ✅ Validates single GPU configurations (30-31 qubits)

cd multi_gpu- ✅ Tests multi-GPU with blocking=27 (32-34 qubits)

python3 validation.py- ✅ Automatically detects available GPUs

```- ✅ Shows correct GPU usage from metadata

- ✅ Provides detailed configuration info

**Expected:** Complete test suite for 30-35 qubits (~2-5 minutes)- ✅ Comprehensive test summary



---**Validated Results:**

- 30q: 1 GPU ✅

## 📊 Script Comparison- 31q: 1 GPU ✅  

- 32q: 2 GPUs ✅

### Single GPU Scripts- 33q: 4 GPUs ✅

- 34q: 8 GPUs ✅

| Script | Time | Purpose | Use When |

|--------|------|---------|----------|**Time:** ~2-5 minutes (depending on available GPUs)

| `quick_test.py` | ~10 sec | GPU verification | Testing GPU works |

| `benchmark.py` | ~2-3 min | Performance analysis | Measuring speedup |**Requirements:**

- For 32q: 2 GPUs

**Circuit Sizes:** 10-28 qubits  - For 33q: 4 GPUs

**Max Single GPU:** 31 qubits- For 34q: 8 GPUs

- For 35q: 16 GPUs (cross-node requires MPI)

### Multi-GPU Scripts- XGMI/Infinity Fabric utilization (on MI300/MI250)

- JSON results export

| Script | Time | Purpose | Use When |

|--------|------|---------|----------|**Time:** ~5-10 minutes (full), ~1-2 minutes (quick mode)

| `quick_test.py` | ~20 sec | Multi-GPU verification | Testing multi-GPU works |

| `benchmark.py` | ~5-10 min | Scaling analysis | Measuring multi-GPU scaling |**Usage Examples:**

| `validation.py` | ~2-5 min | Complete validation | Full system validation |

```bash

**Circuit Sizes:** 30-35 qubits  # Run all benchmarks

**Validated:** Up to 34 qubits on 8 GPUspython3 rocm_multi_gpu_benchmark.py



---# Run specific benchmark

python3 rocm_multi_gpu_benchmark.py --mode state

## 🎯 Which Script Should I Use?python3 rocm_multi_gpu_benchmark.py --mode shots



### New to Qiskit Aer + ROCm?# Specify GPUs to use

1. Start with `single_gpu/quick_test.py`python3 rocm_multi_gpu_benchmark.py --gpus 0,1,2,3

2. Then try `multi_gpu/quick_test.py` (if 2+ GPUs)

3. Read the READMEs in each directory# Quick test (reduced parameters)

python3 rocm_multi_gpu_benchmark.py --quick

### Want to Benchmark Performance?

- **Single GPU:** `single_gpu/benchmark.py`# Save results to custom file

- **Multi-GPU:** `multi_gpu/benchmark.py`python3 rocm_multi_gpu_benchmark.py --output my_results.json

```

### Need Production Validation?

- **Complete test:** `multi_gpu/validation.py`**Benchmark Modes:**

- **Specific config:** Modify validation.py for your needs- `state`: State vector distribution scaling (large circuits)

- `shots`: Shot parallelization scaling (high shot counts)

### Troubleshooting Issues?- `combined`: Both strategies together

1. Run `single_gpu/quick_test.py` first- `circuits`: Different circuit types comparison

2. Check `rocm-smi --showid` for GPU detection- `all`: Run all benchmarks (default)

3. See READMEs for common issues

## Expected Performance

---

### Speedup Factors (Typical)

## 📚 Documentation

| Qubits | Circuit Depth | Expected Speedup | Notes |

### Directory READMEs|--------|---------------|------------------|-------|

- **single_gpu/README.md** - Single GPU guide| 10 | ~50 | 1-2x | CPU overhead dominates |

- **multi_gpu/README.md** - Multi-GPU guide ⭐| 15 | ~70 | 2-4x | GPU starts to show benefit |

| 20 | ~100 | 5-10x | GPU clearly faster |

### Root Documentation| 25 | ~120 | 10-20x | Excellent GPU utilization |

- **MULTI_GPU_FINAL_RESULTS.md** - Complete validation results ⭐| 28 | ~140 | 15-30x | Near memory limits |

- **ROCM_MULTI_GPU_GUIDE.md** - Comprehensive user guide

- **CORRECT_MULTI_GPU_API.md** - API reference*Note: Actual speedup depends on GPU model, circuit structure, and number of shots.*

- **PROJECT_SUMMARY.md** - Project overview

### Multi-GPU Scaling Efficiency

---

Expected scaling efficiency when distributing workloads across multiple GPUs:

## ⚙️ Quick Configuration Reference

| Configuration | 2 GPUs | 4 GPUs | 8 GPUs | Notes |

### Single GPU (≤31 qubits)|---------------|--------|--------|--------|-------|

```python| State Distribution | 70-85% | 65-80% | 60-75% | Large circuits (30+ qubits) |

from qiskit_aer import AerSimulator| Shot Parallelization | 85-95% | 80-90% | 75-85% | High shot counts (10k+) |

| Combined | 75-90% | 70-85% | 65-80% | Both strategies |

backend = AerSimulator(method='statevector', device='GPU')

result = backend.run(circuit, shots=1000).result()**With XGMI/Infinity Fabric (MI300/MI250):**

```- Higher efficiency due to fast GPU interconnect

- State distribution: +5-10% efficiency improvement

### Multi-GPU (32+ qubits)- Near-linear scaling for shot parallelization

```python

from qiskit_aer import AerSimulator### AMD GPU Performance



backend = AerSimulator(method='statevector', device='GPU')| GPU | Memory | Max Qubits* | Expected Performance (Single GPU) |

|-----|--------|-------------|-----------------------------------|

result = backend.run(| MI300X | 192 GB | 28 | Best performance |

    circuit,| MI250X | 128 GB | 27 | Excellent |

    shots=1000,| MI210 | 64 GB | 26 | Very good |

    blocking_enable=True,| MI100 | 32 GB | 25 | Good |

    blocking_qubits=27,        # ⚠️ Max 27!| RX 7900 XTX | 24 GB | 24-25 | Good for consumer GPU |

    target_gpus=[0, 1, 2, 3],  # In run()!

    batched_shots_gpu=True,*With `blocking_enable=True`

    batched_shots_gpu_max_qubits=num_qubits

).result()## Running from Source Directory

```

**Important:** If running from the qiskit-aer source directory, use this approach:

**GPU Requirements:** 32q→2 GPUs, 33q→4 GPUs, 34q→8 GPUs

```bash

---# Change to a different directory first

cd /tmp

## 📈 Performance Summary

# Then run with full path

### Single GPU (MI300X)python3 /home/ysha/playground/qiskit-aer/examples/quick_gpu_test.py

- 10q: <1s | 20q: ~3s | 28q: ~8s | 31q: ~20s```

- **Speedup:** 10-50x vs CPU

Or use the installed package:

### Multi-GPU (8× MI300X)

- 32q (2 GPUs): ~15s | 33q (4 GPUs): ~20s | 34q (8 GPUs): ~30s```bash

- **Efficiency:** ~80-85% of linear scaling# Activate your virtual environment

source ~/amd/qiskit-aer/venv/bin/activate

---

# Change directory

**Status:** Production ready for 30-34 qubits on AMD MI300X  cd ~

**Repository:** qiskit-aer (coketaste/amd-rocm-multigpu branch)

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
