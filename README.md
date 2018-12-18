# Qiskit-Aer

This is a working draft for the Qiskit-Aer simulator framework for Qiskit-Terra. This is the development repository and has no guarantee of stability or correctness.


## Repository contents

* The **qiskit_aer** folder contains the Qiskit-Aer python module for use with Qiskit-Terra.
* The **cmake** folder contains cmake scripts for building the simulator
* The **src** folder contains the C++ source files for building the simulator.
* The **contrib** folder for external contributions.
* The **test** folder contains simulator unit tests.

## Documentation

There's a [contributing guide](https://github.ibm.com/IBMQuantum/qiskit-aer/blob/master/.github/CONTRIBUTING.rst)
with more detailed information about the project.

For tutorials on using Qiskit Aer visit the [qiskit-tutorials](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aer) Github repository.

## Installation

Follow these steps for installing the **Qiskit Aer** package:

```bash
qiskit-aer$ pip install -r requirements-dev.txt
qiskit-aer$ python ./setup.py bdist_wheel
```

Once the build finishes, we just need to install the wheel package in our
preferred python virtual environment:

```bash
qiskit-aer$ pip install dist/qiskit_aer-0.1.0-cp36-cp36m-linux_x86_64.whl
```

We are all set! Now we ready to start using the simulator in our python code:
```python
from qiskit import Aer  # Imports the Aer Provider

Aer.backends() # List of the backends
```
