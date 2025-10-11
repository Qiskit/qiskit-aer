# Qiskit Aer ROCm Examples

**Last Updated:** October 10, 2025  
**Status:** ✅ Production Ready (Validated on AMD MI300X)

Organized examples demonstrating AMD GPU acceleration with ROCm for quantum circuit simulation.

---

## 📁 Directory Structure

```
examples/
├── single_gpu/          ← Single GPU examples (up to 31 qubits)
│   ├── quick_test.py       Fast GPU verification (~10 sec)
│   ├── benchmark.py        Comprehensive benchmarking (~2-3 min)
│   └── README.md           Single GPU documentation
│
├── multi_gpu/           ← Multi-GPU examples (32-40 qubits) ⭐
│   ├── quick_test.py       Fast multi-GPU verification (~20 sec)
│   ├── benchmark.py        Scaling & performance analysis (~5-10 min)
│   ├── validation.py       Complete validation suite (30-35q, ~2-5 min)
│   └── README.md           Multi-GPU documentation
│
└── archive/             ← Historical versions and old scripts
    ├── test_*.py           Obsolete test scripts
    └── docs/               Iterative debugging documentation
```

---

## 🚀 Quick Start

### Step 1: Verify Single GPU
```bash
cd single_gpu
python3 quick_test.py
```

**Expected:** GPU detection and basic performance test (~10 seconds)

### Step 2: Test Multi-GPU (if available)
```bash
cd multi_gpu
python3 quick_test.py
```

**Expected:** Multi-GPU confirmation with 32-qubit circuit (~20 seconds)

### Step 3: Run Full Validation (recommended)
```bash
cd multi_gpu
python3 validation.py
```

**Expected:** Complete test suite (30-35 qubits) with pass/fail summary

---

## 📊 Single GPU Examples

### quick_test.py
- Quick GPU verification (~10 sec)
- CPU vs GPU comparison (20q circuit)
- Use for: First-time setup, health checks

### benchmark.py
- Comprehensive performance analysis (~2-3 min)
- Tests 10, 15, 20, 25 qubit circuits
- Memory requirements and speedup metrics
- Use for: Performance tuning, baseline measurements

See `single_gpu/README.md` for detailed documentation.

---

## 🎯 Multi-GPU Examples ⭐

### quick_test.py
- Fast multi-GPU test (~20 sec)
- 32q circuit on 2 GPUs
- Requires: 2+ GPUs with 64GB+ each

### benchmark.py
- Scaling analysis (~5-10 min)
- Tests 30-34 qubit circuits
- Performance metrics across 1-8 GPUs

### validation.py (Main Test)
- Complete validation suite (~2-5 min)
- Tests 30q-35q configurations
- Auto-detects available GPUs
- **Validated Results:**
  - ✅ 30-31q: 1 GPU
  - ✅ 32q: 2 GPUs
  - ✅ 33q: 4 GPUs
  - ✅ 34q: 8 GPUs

See `multi_gpu/README.md` for detailed documentation.

---

## ⚠️ Critical Constraints

1. **blocking_qubits ≤ 27** (2GB chunks maximum)
2. **Single GPU limit: 31 qubits**
3. **Metadata location:** `metadata['cacheblocking']['chunk_parallel_gpus']`
4. **Parameter location:** `target_gpus` in `run()`, not constructor

---

## 📚 Documentation

- **[MULTI_GPU_FINAL_RESULTS.md](../MULTI_GPU_FINAL_RESULTS.md)** - Complete validation results
- **[ROCM_MULTI_GPU_GUIDE.md](../ROCM_MULTI_GPU_GUIDE.md)** - Comprehensive guide
- **[QUICKSTART_MULTI_GPU.md](../QUICKSTART_MULTI_GPU.md)** - 5-minute start
- **Single GPU:** `single_gpu/README.md`
- **Multi-GPU:** `multi_gpu/README.md`

---

**Validation Status:** Complete (30-34 qubits on AMD MI300X)  
**Ready for:** Production use, research, further development
