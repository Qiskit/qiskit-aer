# Qiskit Aer

[![License](https://img.shields.io/github/license/Qiskit/qiskit-aer.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/com/Qiskit/qiskit-aer/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/qiskit-aer)[![](https://img.shields.io/github/release/Qiskit/qiskit-aer.svg?style=popout-square)](https://github.com/Qiskit/qiskit-aer/releases)[![](https://img.shields.io/pypi/dm/qiskit-aer.svg?style=popout-square)](https://pypi.org/project/qiskit-aer/)

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of pulses, circuits, and algorithms.

Qiskit is made up of elements that each work together to enable quantum computing. This element is **Aer**, which provides high-performance quantum computing simulators with realistic noise models.

## Installation

We encourage installing Qiskit via the PIP tool (a python package manager), which installs all Qiskit elements, including this one.

```bash
pip install qiskit
```

PIP will handle all dependencies automatically for us and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [contribution guidelines](https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md).

## Installing GPU support

In order to install and run the GPU supported simulators on Linux, you need CUDA&reg; 10.1 or newer previously installed.
CUDA&reg; itself would require a set of specific GPU drivers. Please follow CUDA&reg; installation procedure in the NVIDIA&reg; [web](https://www.nvidia.com/drivers).

If you want to install our GPU supported simulators, you have to install this other package:

```bash
pip install qiskit-aer-gpu
```

This will overwrite your current `qiskit-aer` package installation giving you
the same functionality found in the canonical `qiskit-aer` package, plus the
ability to run the GPU supported simulators: statevector, density matrix, and unitary.

**Note**: This package is only available on x86_64 Linux. For other platforms
that have CUDA support you will have to build from source. You can refer to
the [contributing guide](https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md#building-with-gpu-support)
for instructions on doing this.

## Simulating your first quantum program with Qiskit Aer
Now that you have Qiskit Aer installed, you can start simulating quantum circuits with noise. Here is a basic example:

```
$ python
```

```python
import qiskit
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator

# Generate 3-qubit GHZ state
circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()

# Construct an ideal simulator
aersim = AerSimulator()

# Perform an ideal simulation
result_ideal = qiskit.execute(circ, aersim).result()
counts_ideal = result_ideal.get_counts(0)
print('Counts(ideal):', counts_ideal)
# Counts(ideal): {'000': 493, '111': 531}

# Construct a noisy simulator backend from an IBMQ backend
# This simulator backend will be automatically configured
# using the device configuration and noise model 
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_athens')
aersim_backend = AerSimulator.from_backend(backend)

# Perform noisy simulation
result_noise = qiskit.execute(circ, aersim_backend).result()
counts_noise = result_noise.get_counts(0)

print('Counts(noise):', counts_noise)
# Counts(noise): {'000': 492, '001': 6, '010': 8, '011': 14, '100': 3, '101': 14, '110': 18, '111': 469}
```

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](https://github.com/Qiskit/qiskit-aer/blob/master/CONTRIBUTING.md). This project adheres to Qiskit's [code of conduct](https://github.com/Qiskit/qiskit-aer/blob/master/CODE_OF_CONDUCT.md). By participating, you are expect to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aer/issues) for tracking requests and bugs. Please use our [slack](https://qiskit.slack.com) for discussion and simple questions. To join our Slack community use the [link](https://qiskit.slack.com/join/shared_invite/zt-fybmq791-hYRopcSH6YetxycNPXgv~A#/). For questions that are more suited for a forum we use the Qiskit tag in the [Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit IQX Tutorials](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/simulators) or [Qiskit Community Tutorials](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/aer) repositories.

## Authors and Citation

Qiskit Aer is the work of [many people](https://github.com/Qiskit/qiskit-aer/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

## License

[Apache License 2.0](LICENSE.txt)
