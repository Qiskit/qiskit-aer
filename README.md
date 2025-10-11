# Aer - high performance quantum circuit simulation for Qiskit

[![License](https://img.shields.io/github/license/Qiskit/qiskit-aer.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)
[![Build](https://github.com/Qiskit/qiskit-aer/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-aer/actions/workflows/build.yml)
[![Tests](https://github.com/Qiskit/qiskit-aer/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-aer/actions/workflows/tests.yml)
[![](https://img.shields.io/github/release/Qiskit/qiskit-aer.svg?style=popout-square)](https://github.com/Qiskit/qiskit-aer/releases)
[![](https://img.shields.io/pypi/dm/qiskit-aer.svg?style=popout-square)](https://pypi.org/project/qiskit-aer/)

**Aer** is a high performance simulator for quantum circuits written in Qiskit, that includes realistic noise models.

## Installation

We encourage installing Aer via the pip tool (a python package manager):

```bash
pip install qiskit-aer
```

Pip will handle all dependencies automatically for us, and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [contribution guidelines](CONTRIBUTING.md).

## Installing GPU support

In order to install and run the GPU supported simulators on Linux, you need CUDA&reg; 11.2 or newer previously installed.
CUDA&reg; itself would require a set of specific GPU drivers. Please follow CUDA&reg; installation procedure in the NVIDIA&reg; [web](https://www.nvidia.com/drivers).

If you want to install our GPU supported simulators, you have to install this other package:

```bash
pip install qiskit-aer-gpu
```

The package above is for CUDA&reg 12, so if your system has CUDA&reg; 11 installed, install separate package:
```bash
pip install qiskit-aer-gpu-cu11
```

This will overwrite your current `qiskit-aer` package installation giving you
the same functionality found in the canonical `qiskit-aer` package, plus the
ability to run the GPU supported simulators: statevector, density matrix, and unitary.

**Note**: This package is only available on x86_64 Linux. For other platforms
that have CUDA support, you will have to build from source. You can refer to
the [contributing guide](CONTRIBUTING.md#building-with-gpu-support)
for instructions on doing this.

### AMD ROCm GPU Support

Qiskit Aer also supports AMD GPUs via ROCm (5.0+). To use AMD GPUs, you need to build from source:

#### Prerequisites
- ROCm 5.0 or newer (ROCm 7.x recommended for latest features)
- AMD GPU: MI100/MI200/MI300 (data center) or RX 6000/7000 series (consumer)
- Ubuntu 20.04/22.04 or compatible Linux distribution

#### Quick Start with Auto-Detection
```bash
# 1. Install ROCm (if not already installed)
# Follow: https://rocm.docs.amd.com/

# 2. Detect your GPU and generate build script
./tools/detect_rocm.sh

# 3. Run the generated build script
bash /tmp/qiskit_aer_rocm_build.sh
```

#### Manual Build
```bash
export ROCM_PATH=/opt/rocm
export AER_THRUST_BACKEND=ROCM
export AER_ROCM_ARCH=gfx90a  # Your GPU architecture (see table below)

python3 -m pip install -r requirements-dev.txt
QISKIT_AER_PACKAGE_NAME='qiskit-aer-gpu-rocm' \
    python3 setup.py bdist_wheel -- \
        -DAER_THRUST_BACKEND=ROCM \
        -DAER_ROCM_ARCH=gfx90a
python3 -m pip install --force-reinstall dist/qiskit_aer_gpu_rocm-*.whl
```

#### Supported AMD GPU Architectures

| GPU Family | Architecture | Example GPUs | Recommended blocking_qubits |
|------------|-------------|--------------|------------------------------|
| MI300 | gfx940, gfx941, gfx942 | MI300A/X | 28 (192GB HBM3) |
| MI200 | gfx90a | MI210, MI250X | 27 (64-128GB HBM2e) |
| MI100 | gfx908 | MI100 | 25 (32GB HBM2) |
| RX 7000 | gfx1100 | RX 7900 XTX | 25 (24GB GDDR6) |
| RX 6000 | gfx1030 | RX 6900 XT | 24 (16GB GDDR6) |

#### Usage Example
```python
from qiskit_aer import AerSimulator

# Create GPU simulator
sim = AerSimulator(method='statevector', device='GPU')

# Check available devices
print(sim.available_devices())  # Should show GPU

# Run with memory management
result = sim.run(circuit, 
                 blocking_enable=True,
                 blocking_qubits=27,  # Adjust for your GPU
                 shots=1000).result()
```

#### Multi-GPU Support ✨ NEW - Validated on MI300X

Leverage multiple AMD GPUs for larger circuits (32-40 qubits on single node):

```python
from qiskit_aer import AerSimulator
from qiskit.circuit.library import quantum_volume

backend = AerSimulator(method='statevector', device='GPU')

# 33-qubit circuit on 4 GPUs
circuit = quantum_volume(33, depth=10, seed=42)
circuit.measure_all()

# Run with validated configuration
result = backend.run(
    circuit,
    shots=100,
    blocking_enable=True,
    blocking_qubits=27,        # ⚠️ CRITICAL: Max 27 (2GB chunks)
    target_gpus=[0, 1, 2, 3],  # Must be in run(), not constructor
    batched_shots_gpu=True,
    batched_shots_gpu_max_qubits=33
).result()

# Verify multi-GPU usage
cacheblocking = result.results[0].metadata['cacheblocking']
print(f"GPUs used: {cacheblocking['chunk_parallel_gpus']}")  # Should show: 4
```

**Validated Configurations (AMD MI300X):**
- ✅ **30-31 qubits:** 1 GPU (no blocking needed)
- ✅ **32 qubits:** 2 GPUs with blocking=27
- ✅ **33 qubits:** 4 GPUs with blocking=27
- ✅ **34 qubits:** 8 GPUs with blocking=27

**Critical Constraints:**
1. `blocking_qubits ≤ 27` (2GB chunk maximum)
2. Single GPU limit: 31 qubits
3. GPU count: Use `ceil(2^(qubits-27) / 16)` GPUs for qubits > 31

📚 **Complete Guide:**

**Quick Verification:**
```bash
# Single GPU test
python3 examples/single_gpu/quick_test.py

# Multi-GPU test (requires 2+ GPUs)
python3 examples/multi_gpu/quick_test.py

# Full validation (30-35 qubits)
python3 examples/multi_gpu/validation.py
```

**Advanced Usage:**
```python
# Large circuit with state distribution
result = backend.run(large_circuit,
                     blocking_enable=True,      # Distribute state across GPUs
                     blocking_qubits=27,        # Chunk size per GPU (max 27)
                     target_gpus=[0,1,2,3],     # Select specific GPUs
                     shots=1000).result()

# High-shot simulation with shot parallelization
result = backend.run(circuit,
                     batched_shots_gpu=True,    # Distribute shots across GPUs
                     shots=10000).result()
```

**Resources:**
- **Build Instructions**: [BUILDING_ROCM.md](BUILDING_ROCM.md) - ROCm build guide

**Examples:**
- **Single GPU**: `examples/single_gpu/` - Quick tests and benchmarks for single GPU
  - `quick_test.py` - Verify GPU functionality (~10 seconds)
  - `benchmark.py` - Performance comparison CPU vs GPU (~2-3 minutes)
- **Multi-GPU**: `examples/multi_gpu/` - Multi-GPU examples (validated on MI300X)
  - `quick_test.py` - Quick multi-GPU verification (~30 seconds)
  - `benchmark.py` - Comprehensive multi-GPU benchmarks (~5-10 minutes)
  - `validation.py` - Complete validation (30-35 qubits, requires 1-16 GPUs)

📚 See [examples/README.md](examples/README.md) for detailed usage instructions.

## Simulating your first Qiskit circuit with Aer
Now that you have Aer installed, you can start simulating quantum circuits using primitives and noise models. Here is a basic example:

```
$ python
```

```python
from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

sim = AerSimulator()
# --------------------------
# Simulating using estimator
#---------------------------
from qiskit_aer.primitives import EstimatorV2

psi1 = transpile(RealAmplitudes(num_qubits=2, reps=2), sim, optimization_level=0)
psi2 = transpile(RealAmplitudes(num_qubits=2, reps=3), sim, optimization_level=0)

H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
H2 = SparsePauliOp.from_list([("IZ", 1)])
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
theta3 = [1, 2, 3, 4, 5, 6]

estimator = EstimatorV2()

# calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
#              <psi1(theta3)|H3|psi1(theta3)>],
#             [<psi2(theta2)|H2|psi2(theta2)>] ]
job = estimator.run(
    [
        (psi1, [H1, H3], [theta1, theta3]),
        (psi2, H2, theta2)
    ],
    precision=0.01
)
result = job.result()
print(f"expectation values : psi1 = {result[0].data.evs}, psi2 = {result[1].data.evs}")

# --------------------------
# Simulating using sampler
# --------------------------
from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit

# create a Bell circuit
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

# create two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc = transpile(pqc, sim, optimization_level=0)
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
pqc2 = transpile(pqc2, sim, optimization_level=0)

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

# initialization of the sampler
sampler = SamplerV2()

# collect 128 shots from the Bell circuit
job = sampler.run([bell], shots=128)
job_result = job.result()
print(f"counts for Bell circuit : {job_result[0].data.meas.get_counts()}")
 
# run a sampler job on the parameterized circuits
job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
job_result = job2.result()
print(f"counts for parameterized circuit : {job_result[0].data.meas.get_counts()}")

# --------------------------------------------------
# Simulating with noise model from actual hardware
# --------------------------------------------------
from qiskit_ibm_runtime import QiskitRuntimeService
provider = QiskitRuntimeService(channel='ibm_quantum', token="set your own token here")
backend = provider.get_backend("ibm_kyoto")

# create sampler from the actual backend
sampler = SamplerV2.from_backend(backend)

# run a sampler job on the parameterized circuits with noise model of the actual hardware
bell_t = transpile(bell, AerSimulator(basis_gates=["ecr", "id", "rz", "sx"]), optimization_level=0)
job3 = sampler.run([bell_t], shots=128)
job_result = job3.result()
print(f"counts for Bell circuit w/noise: {job_result[0].data.meas.get_counts()}")
```

## Contribution Guidelines

If you'd like to contribute to Aer, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aer/issues) for tracking requests and bugs. Please use our [slack](https://qiskit.slack.com) for discussion and simple questions. To join our Slack community use the [link](https://qiskit.slack.com/join/shared_invite/zt-fybmq791-hYRopcSH6YetxycNPXgv~A#/). For questions that are more suited for a forum, we use the Qiskit tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the [Aer documentation](https://qiskit.github.io/qiskit-aer/).

## Authors and Citation

Aer is the work of [many people](https://github.com/Qiskit/qiskit-aer/graphs/contributors) who contribute to the project at different levels.
If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/main/CITATION.bib).

## License

[Apache License 2.0](LICENSE.txt)
