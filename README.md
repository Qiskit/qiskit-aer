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
Now that you have Aer installed, you can start simulating quantum circuits with noise. Here is a basic example:

```
$ python
```

```python
import qiskit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

# Construct an ideal simulator
aersim = AerSimulator()

# Perform an ideal simulation
result_ideal = aersim.run(circ).result()
counts_ideal = result_ideal.get_counts(0)
print('Counts(ideal):', counts_ideal)
# Counts(ideal): {'000': 493, '111': 531}

# Construct a simulator using a noise model
# from a real backend.
provider = QiskitRuntimeService()
backend = provider.get_backend("ibm_kyoto")
aersim_backend = AerSimulator.from_backend(backend)

# Perform noisy simulation
result_noise = aersim_backend.run(circ).result()
counts_noise = result_noise.get_counts(0)

print('Counts(noise):', counts_noise)
# Counts(noise): {'101': 16, '110': 48, '100': 7, '001': 31, '010': 7, '000': 464, '011': 15, '111': 436}
```

## Contribution Guidelines

If you'd like to contribute to Aer, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aer/issues) for tracking requests and bugs. Please use our [slack](https://qiskit.slack.com) for discussion and simple questions. To join our Slack community use the [link](https://qiskit.slack.com/join/shared_invite/zt-fybmq791-hYRopcSH6YetxycNPXgv~A#/). For questions that are more suited for a forum, we use the Qiskit tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the [Aer documentation](https://qiskit.org/ecosystem/aer/).

## Authors and Citation

Aer is the work of [many people](https://github.com/Qiskit/qiskit-aer/graphs/contributors) who contribute to the project at different levels.
If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/main/CITATION.bib).

## License

[Apache License 2.0](LICENSE.txt)
