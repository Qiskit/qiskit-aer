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
