import numpy as np
from qiskit.circuit import Gate


class CustomMultiplexer(Gate):
    def validate_parameter(self, param):
        return param


def multiplexer_multi_controlled_x(num_control):
    # Multi-controlled X gate multiplexer
    identity = np.array(np.array([[1, 0], [0, 1]], dtype=complex))
    x_gate = np.array(np.array([[0, 1], [1, 0]], dtype=complex))
    num_qubits = num_control + 1
    multiplexer = CustomMultiplexer(
        "multiplexer",
        num_qubits,
        (2**num_control - 1) * [identity] + [x_gate],
    )
    return multiplexer
