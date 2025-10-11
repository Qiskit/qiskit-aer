# Single GPU Examples

Examples for single GPU quantum circuit simulation with Qiskit Aer and ROCm.

## Scripts

### 1. Quick Test (`quick_test.py`)
**Purpose:** Fast verification that GPU acceleration is working  
**Time:** ~5-10 seconds  
**Circuit Size:** 20 qubits

```bash
python3 quick_test.py
```

**What it does:**
- Detects available AMD GPU
- Runs simple CPU vs GPU comparison
- Shows speedup factor
- Confirms GPU is accessible

**Expected output:**
```
✅ AMD GPU detected
✅ CPU baseline: X.XX seconds
✅ GPU execution: X.XX seconds
✅ Speedup: XXx faster
```

---

### 2. Benchmark (`benchmark.py`)
**Purpose:** Comprehensive single-GPU performance analysis  
**Time:** ~2-3 minutes  
**Circuit Sizes:** 10, 15, 20, 25, 28 qubits

```bash
python3 benchmark.py
```

**What it does:**
- Tests multiple circuit sizes
- Compares CPU vs GPU performance
- Shows memory requirements
- Calculates average speedup
- Provides optimization recommendations

**Features:**
- Memory requirements table
- Detailed performance metrics
- Sample circuit execution
- GPU utilization analysis

---

## Use Cases

### Quick Verification
Use `quick_test.py` when you want to:
- Verify GPU is working after installation
- Quick health check before larger jobs
- Test after ROCm/driver updates

### Performance Analysis
Use `benchmark.py` when you want to:
- Understand GPU performance characteristics
- Compare different circuit sizes
- Determine optimal circuit size for your GPU
- Generate performance reports

---

## GPU Limits (Single GPU)

| GPU Model | Memory | Max Qubits (Statevector) | Notes |
|-----------|--------|--------------------------|-------|
| MI300X    | 192 GB | 31 qubits | Validated maximum |
| MI250X    | 128 GB | 30 qubits | Theoretical |
| MI210     | 64 GB  | 29 qubits | Theoretical |
| MI100     | 32 GB  | 28 qubits | Theoretical |

**Note:** For circuits larger than these limits, see `multi_gpu/` examples.

---

## Requirements

- Qiskit Aer with ROCm support
- AMD GPU (MI100/MI200/MI300 or RX 6000/7000)
- ROCm 5.0+ (ROCm 6.0+ recommended)
- Python 3.8+

---

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU availability
rocm-smi --showid

# Verify ROCm installation
rocm-smi --showproductname
```

### Out of Memory
If circuits fail with OOM:
1. Try smaller circuit (reduce qubits by 1)
2. Check available memory: `rocm-smi --showmeminfo vram`
3. Consider multi-GPU examples for larger circuits

### Slow Performance
- Ensure no other GPU workloads running
- Check GPU utilization: `rocm-smi --showuse`
- Verify GPU clocks: `rocm-smi --showclocks`

---

## Next Steps

**Need larger circuits (32+ qubits)?**  
→ See `multi_gpu/` examples for multi-GPU support

**Want detailed validation?**  
→ Check main documentation in root directory

**Performance tuning?**  
→ See `ROCM_MULTI_GPU_GUIDE.md` in root
