# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
"""
Aer simulator backend utils
"""
import os
from math import log2

from types import SimpleNamespace

import psutil
from qiskit.circuit import QuantumCircuit
from qiskit.result import ProbDistribution
from qiskit.quantum_info import Clifford

from .compatibility import Statevector, DensityMatrix, StabilizerState, Operator, SuperOp

# pylint: disable=import-error, no-name-in-module, abstract-method
from .controller_wrappers import aer_initialize_libraries

# Available system memory
SYSTEM_MEMORY_GB = psutil.virtual_memory().total / (1024**3)

# Max number of qubits for complex double statevector
# given available system memory
MAX_QUBITS_STATEVECTOR = int(log2(SYSTEM_MEMORY_GB * (1024**3) / 16))

# Location where we put external libraries that will be
# loaded at runtime by the simulator extension
LIBRARY_DIR = os.path.dirname(__file__)
aer_initialize_libraries(LIBRARY_DIR)

LEGACY_METHOD_MAP = {
    "statevector_cpu": ("statevector", "CPU"),
    "statevector_gpu": ("statevector", "GPU"),
    "statevector_thrust": ("statevector", "Thrust"),
    "density_matrix_cpu": ("density_matrix", "CPU"),
    "density_matrix_gpu": ("density_matrix", "GPU"),
    "density_matrix_thrust": ("density_matrix", "Thrust"),
    "unitary_cpu": ("unitary", "CPU"),
    "unitary_gpu": ("unitary", "GPU"),
    "unitary_thrust": ("unitary", "Thrust"),
}

BASIS_GATES = {
    "statevector": sorted(
        [
            "u1",
            "u2",
            "u3",
            "u",
            "p",
            "r",
            "rx",
            "ry",
            "rz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "swap",
            "cx",
            "cy",
            "cz",
            "csx",
            "cp",
            "cu",
            "cu1",
            "cu2",
            "cu3",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "ccx",
            "ccz",
            "crx",
            "cry",
            "crz",
            "cswap",
            "mcx",
            "mcy",
            "mcz",
            "mcsx",
            "mcp",
            "mcphase",
            "mcu",
            "mcu1",
            "mcu2",
            "mcu3",
            "mcrx",
            "mcry",
            "mcrz",
            "mcr",
            "mcswap",
            "unitary",
            "diagonal",
            "multiplexer",
            "pauli",
            "mcx_gray",
            "ecr",
            "store",
        ]
    ),
    "density_matrix": sorted(
        [
            "u1",
            "u2",
            "u3",
            "u",
            "p",
            "r",
            "rx",
            "ry",
            "rz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "swap",
            "cx",
            "cy",
            "cz",
            "cp",
            "cu1",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "ccx",
            "unitary",
            "diagonal",
            "pauli",
            "ecr",
        ]
    ),
    "matrix_product_state": sorted(
        [
            "u1",
            "u2",
            "u3",
            "u",
            "p",
            "cp",
            "cx",
            "cy",
            "cz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "swap",
            "ccx",
            "unitary",
            "roerror",
            "pauli",
            "r",
            "rx",
            "ry",
            "rz",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "csx",
            "cswap",
            "diagonal",
            "ecr",
            "store",
        ]
    ),
    "stabilizer": sorted(
        [
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "cx",
            "cy",
            "cz",
            "swap",
            "pauli",
            "ecr",
            "rz",
            "store",
        ]
    ),
    "extended_stabilizer": sorted(
        [
            "cx",
            "cz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "swap",
            "u0",
            "t",
            "tdg",
            "u1",
            "p",
            "ccx",
            "ccz",
            "pauli",
            "ecr",
            "rz",
            "store",
        ]
    ),
    "unitary": sorted(
        [
            "u1",
            "u2",
            "u3",
            "u",
            "p",
            "r",
            "rx",
            "ry",
            "rz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "swap",
            "cx",
            "cy",
            "cz",
            "csx",
            "cp",
            "cu",
            "cu1",
            "cu2",
            "cu3",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "ccx",
            "ccz",
            "cswap",
            "crx",
            "cry",
            "crz",
            "mcx",
            "mcy",
            "mcz",
            "mcsx",
            "mcp",
            "mcphase",
            "mcu",
            "mcu1",
            "mcu2",
            "mcu3",
            "mcrx",
            "mcry",
            "mcrz",
            "mcr",
            "mcswap",
            "unitary",
            "diagonal",
            "multiplexer",
            "pauli",
            "ecr",
            "store",
        ]
    ),
    "superop": sorted(
        [
            "u1",
            "u2",
            "u3",
            "u",
            "p",
            "r",
            "rx",
            "ry",
            "rz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "swap",
            "cx",
            "cy",
            "cz",
            "cp",
            "cu1",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "ccx",
            "unitary",
            "diagonal",
            "pauli",
            "store",
        ]
    ),
    "tensor_network": sorted(
        [
            "u1",
            "u2",
            "u3",
            "u",
            "p",
            "r",
            "rx",
            "ry",
            "rz",
            "id",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "t",
            "tdg",
            "swap",
            "cx",
            "cy",
            "cz",
            "csx",
            "cp",
            "cu",
            "cu1",
            "cu2",
            "cu3",
            "rxx",
            "ryy",
            "rzz",
            "rzx",
            "ccx",
            "ccz",
            "cswap",
            "crx",
            "cry",
            "crz",
            "mcx",
            "mcy",
            "mcz",
            "mcsx",
            "mcp",
            "mcphase",
            "mcu",
            "mcu1",
            "mcu2",
            "mcu3",
            "mcrx",
            "mcry",
            "mcrz",
            "mcr",
            "mcswap",
            "unitary",
            "diagonal",
            "multiplexer",
            "pauli",
            "mcx_gray",
            "ecr",
            "store",
        ]
    ),
}

# Automatic method basis gates are the union of statevector,
# density matrix, and stabilizer methods
BASIS_GATES[None] = BASIS_GATES["automatic"] = sorted(
    set(BASIS_GATES["statevector"])
    .union(BASIS_GATES["stabilizer"])
    .union(BASIS_GATES["density_matrix"])
    .union(BASIS_GATES["matrix_product_state"])
    .union(BASIS_GATES["unitary"])
    .union(BASIS_GATES["superop"])
    .union(BASIS_GATES["tensor_network"])
)


def cpp_execute_circuits(controller, aer_circuits, noise_model, config):
    """Execute aer circuits on C++ controller wrapper"""

    # Location where we put external libraries that will be
    # loaded at runtime by the simulator extension
    config.library_dir = LIBRARY_DIR

    noise_model = noise_model.to_dict(serializable=True) if noise_model else {}

    return controller.execute(aer_circuits, noise_model, config)


def available_methods(methods, devices):
    """Check available simulation methods"""

    valid_methods = []
    for method in methods:
        if method == "tensor_network":
            if "GPU" in devices:
                valid_methods.append(method)
        else:
            valid_methods.append(method)
    return tuple(valid_methods)


def available_devices(controller):
    """return available simulation devices"""
    dev = controller.available_devices()
    return tuple(dev)


def add_final_save_op(aer_circuits, state):
    """Add final save state op to all experiments in a qobj."""

    for aer_circuit in aer_circuits:
        num_qubits = aer_circuit.num_qubits
        aer_circuit.save_state(list(range(num_qubits)), f"save_{state}", "single", state)

    return aer_circuits


def map_legacy_method_config(config):
    """Map legacy method names of qasm simulator to aer simulator options"""
    method = config.method
    if method in LEGACY_METHOD_MAP:
        config.method, config.device = LEGACY_METHOD_MAP[method]
    return config


def format_save_type(data, save_type, save_subtype):
    """Format raw simulator result data based on save type."""
    init_fns = {
        "save_statevector": Statevector,
        "save_density_matrix": DensityMatrix,
        "save_unitary": Operator,
        "save_superop": SuperOp,
        "save_stabilizer": (lambda data: StabilizerState(Clifford.from_dict(data))),
        "save_clifford": Clifford.from_dict,
        "save_probabilities_dict": ProbDistribution,
    }

    # Non-handled cases return raw data
    if save_type not in init_fns:
        return data

    if save_subtype in ["list", "c_list"]:

        def func(data):
            init_fn = init_fns[save_type]
            return [init_fn(i) for i in data]

    else:
        func = init_fns[save_type]

    # Conditional save
    if save_subtype[:2] == "c_":
        return {key: func(val) for key, val in data.items()}

    return func(data)


def circuit_optypes(circuit):
    """Return set of all operation types and parent types in a circuit."""
    if not isinstance(circuit, QuantumCircuit):
        return set()
    optypes = set()
    for instruction in circuit.data:
        optypes.update(type(instruction.operation).mro())
    optypes.discard(object)
    return optypes


class CircuitHeader(SimpleNamespace):
    """A class used to represent a dictionary header in circuit objects."""

    def __init__(self, **kwargs):
        """Instantiate a new circuit dict field object.

        Args:
            kwargs: arbitrary keyword arguments that can be accessed as
                attributes of the object.
        """
        self.__dict__.update(kwargs)

    def to_dict(self):
        """Return a dictionary format representation of the circuit.

        Returns:
            dict: The dictionary form of the CircuitHeader.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        """Create a new header object from a dictionary.

        Args:
            data (dict): A dictionary representing the header to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            CircuitHeader: The CircuitHeader from the input dictionary.
        """

        return cls(**data)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.__dict__ == other.__dict__:
                return True
        return False
