# Multi-GPU Examples

Examples for multi-GPU quantum circuit simulation with Qiskit Aer and ROCm.  
**Validated on AMD MI300X** (up to 34 qubits on 8 GPUs)

## Scripts

### 1. Quick Test (`quick_test.py`) ⭐
**Purpose:** Fast verification that multi-GPU is working  
**Time:** ~10-20 seconds  
**Circuit Sizes:** 28q (single), 32q (multi-GPU)

```bash
python3 quick_test.py
```

**What it does:**
- Detects available AMD GPUs
- Tests single GPU configuration (28q)
- Tests multi-GPU configuration (32q with 2 GPUs)
- Confirms correct GPU usage via metadata

**Expected output:**
```
✅ Detected 8 AMD GPU(s)
✅ Single GPU (28q): PASS
✅ Multi-GPU (32q): PASS (2 GPUs used)
🎉 Multi-GPU is WORKING!
```

---

### 2. Benchmark (`benchmark.py`)
**Purpose:** Comprehensive multi-GPU performance analysis  
**Time:** ~5-10 minutes  
**Tests:** GPU scaling, circuit sizes, memory usage

```bash
python3 benchmark.py
```

**What it does:**
- **GPU Scaling:** Tests 1, 2, 4, 8 GPUs on 32-qubit circuit
- **Circuit Sizes:** Tests 28-34 qubits with optimal GPU counts
- **Memory Analysis:** Shows memory usage patterns
- **Speedup Metrics:** Calculates scaling efficiency

**Features:**
- Scaling efficiency analysis
- Performance vs GPU count
- Memory distribution validation
- Comprehensive performance report

---

### 3. Validation (`validation.py`) 🔬
**Purpose:** Complete multi-GPU validation (30-35 qubits)  
**Time:** ~2-5 minutes  
**Most comprehensive test**

```bash
python3 validation.py
```

**What it does:**
- Tests all validated configurations:
  - 30q: 1 GPU (no blocking)
  - 31q: 1 GPU (maximum single GPU)
  - 32q: 2 GPUs, blocking=27
  - 33q: 4 GPUs, blocking=27
  - 34q: 8 GPUs, blocking=27
  - 35q: 16 GPUs, blocking=27
- Verifies correct metadata location
- Shows detailed configuration info
- Provides comprehensive test summary

**Use this for:**
- Full system validation
- Verifying all GPU counts work correctly
- Testing before production use
- Generating validation reports

---

## Validated Configurations

| Qubits | GPUs | Blocking | Chunks | Chunk Size | Status |
|--------|------|----------|--------|------------|--------|
| 30     | 1    | None     | 1      | 16 GB      | ✅ PASS |
| 31     | 1    | None     | 1      | 32 GB      | ✅ PASS |
| 32     | 2    | 27       | 32     | 2 GB       | ✅ PASS |
| 33     | 4    | 27       | 64     | 2 GB       | ✅ PASS |
| 34     | 8    | 27       | 128    | 2 GB       | ✅ PASS |

---

## Critical Constraints

### 🚨 MUST FOLLOW

1. **`blocking_qubits ≤ 27`** - Hard limit (2GB chunks maximum)
2. **Single GPU limit: 31 qubits** - 32+ requires multi-GPU
3. **Parameter location:** `target_gpus` in `run()`, NOT constructor
4. **Metadata location:** `metadata['cacheblocking']['chunk_parallel_gpus']`

### GPU Requirements Formula

```python
def get_gpu_requirements(num_qubits):
    if num_qubits <= 31:
        return 1  # Single GPU
    
    # Multi-GPU: blocking capped at 27
    chunks = 2 ** (num_qubits - 27)
    gpus = max(2, (chunks + 15) // 16)  # ~16 chunks per GPU
    return gpus

# Examples:
# 32q → 32 chunks  → 2 GPUs
# 33q → 64 chunks  → 4 GPUs
# 34q → 128 chunks → 8 GPUs
# 35q → 256 chunks → 16 GPUs (requires MPI)
```

---

## Quick Start

### First Time Setup
```bash
# 1. Quick verification
python3 quick_test.py

# 2. If successful, run full validation
python3 validation.py

# 3. Optional: Run benchmark for performance analysis
python3 benchmark.py
```

### Example Code

```python
from qiskit_aer import AerSimulator
from qiskit.circuit.library import quantum_volume

backend = AerSimulator(method='statevector', device='GPU')

# 33-qubit circuit with 4 GPUs
circuit = quantum_volume(33, depth=10, seed=42)
circuit.measure_all()

result = backend.run(
    circuit,
    shots=100,
    blocking_enable=True,
    blocking_qubits=27,           # ⚠️ MAX 27!
    target_gpus=[0, 1, 2, 3],    # In run(), not constructor!
    batched_shots_gpu=True,
    batched_shots_gpu_max_qubits=33
).result()

# Verify multi-GPU usage
cacheblocking = result.results[0].metadata['cacheblocking']
print(f"GPUs used: {cacheblocking['chunk_parallel_gpus']}")  # Should show: 4
```

---

## Troubleshooting

### Multi-GPU Not Working (Shows 1 GPU)

**Check correct metadata location:**
```python
# ❌ WRONG: Top-level (shows incorrect count)
gpus = metadata.get('parallel_state_update', 1)

# ✅ CORRECT: Nested in cacheblocking
gpus = metadata['cacheblocking']['chunk_parallel_gpus']
```

### hipErrorInvalidConfiguration

**Causes:**
1. `blocking_qubits > 27` (chunk too large)
2. Too many qubits for single GPU (>31)
3. Not enough GPUs for circuit size

**Solution:**
- Always use `blocking_qubits ≤ 27`
- Use formula to calculate required GPUs
- For 32q+, always enable blocking

### Target GPUs Ignored

**Cause:** `target_gpus` in wrong location

```python
# ❌ WRONG: Constructor (ignored)
backend = AerSimulator(..., target_gpus=[0,1])

# ✅ CORRECT: In run() method
backend.run(circuit, target_gpus=[0,1])
```

---

## Requirements

- Qiskit Aer with ROCm support
- 2+ AMD GPUs for multi-GPU tests
- ROCm 5.0+ (ROCm 6.0+ recommended)
- Python 3.8+
- For 35+ qubits: MPI configuration (see root docs)

---

## Performance Expectations

### Multi-GPU Scaling (32 qubits)
- 1 GPU: Baseline
- 2 GPUs: ~1.7-1.9x speedup
- 4 GPUs: ~3.2-3.6x speedup
- 8 GPUs: ~6.0-7.0x speedup

**Overhead:** <20% vs theoretical linear scaling

### Execution Times (Validated on MI300X)
- 30q (1 GPU): < 10 sec
- 31q (1 GPU): < 10 sec
- 32q (2 GPUs): ~15 sec
- 33q (4 GPUs): ~20 sec
- 34q (8 GPUs): ~30 sec

---

## Documentation

**Complete guides in root directory:**
- `MULTI_GPU_FINAL_RESULTS.md` - Full validation results
- `ROCM_MULTI_GPU_GUIDE.md` - Comprehensive user guide
- `CORRECT_MULTI_GPU_API.md` - API reference
- `PROJECT_SUMMARY.md` - Project overview

---

## Next Steps

**Working with 30-34 qubits?**  
→ You're all set! Use the examples above.

**Need 35+ qubits?**  
→ See `ROCM_MULTI_GPU_GUIDE.md` for MPI configuration

**Production deployment?**  
→ Check `MULTI_GPU_FINAL_RESULTS.md` for best practices

**Contributing?**  
→ See validation results and share with Qiskit Aer team
