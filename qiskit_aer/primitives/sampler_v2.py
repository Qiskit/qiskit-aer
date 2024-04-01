# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sampler V2 class.
"""

from __future__ import annotations

from typing import Iterable
from dataclasses import dataclass, field
import warnings

import numpy as np
from numpy.typing import NDArray

from qiskit import ClassicalRegister, QiskitError, QuantumCircuit
from qiskit.circuit import ControlFlowOp

from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.base.validation import _has_measure
from qiskit.primitives.containers import (
    BitArray,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
    make_data_bin,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.bit_array import _min_num_bytes
from qiskit.primitives.primitive_job import PrimitiveJob

from qiskit_aer import AerSimulator


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    qreg_indices: list[int]


@dataclass
class Options:
    """Options for :class:`~.SamplerV2`."""

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run."""


class SamplerV2(BaseSamplerV2):
    """
    Aer implementation of SamplerV2 class.

    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    * ``backend_options``: Options passed to AerSimulator.
      Default: {}.

    * ``run_options``: Options passed to :meth:`AerSimulator.run`.
      Default: {}.

    """

    def __init__(
        self,
        *,
        default_shots: int = 1024,
        seed: np.random.Generator | int | None = None,
        options: dict | None = None,
    ):
        """
        Args:
            default_shots: The default shots for the sampler if not specified during run.
            seed: The seed or Generator object for random number generation.
                If None, a random seeded default RNG will be used.
            options:
                the backend options (``backend_options``), and
                the runtime options (``run_options``).

        """
        self._default_shots = default_shots
        self._seed = seed

        self._options = Options(**options) if options else Options()
        self._backend = AerSimulator(**self.options.backend_options)

    def from_backend(self, backend, **options):
        """use external backend"""
        self._backend.from_backend(backend, **options)

    @property
    def default_shots(self) -> int:
        """Return the default shots"""
        return self._default_shots

    @property
    def seed(self) -> np.random.Generator | int | None:
        """Return the seed or Generator object for random number generation."""
        return self._seed

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if shots is None:
            shots = self._default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        if any(len(pub.circuit.cregs) == 0 for pub in coerced_pubs):
            warnings.warn(
                "One of your circuits has no output classical registers and so the result "
                "will be empty. Did you mean to add measurement instructions?",
                UserWarning,
            )

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[PubResult]:
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub(self, pub: SamplerPub) -> PubResult:
        circuit, qargs, meas_info = _preprocess_circuit(pub.circuit)

        # convert to parameter bindings
        parameter_values = pub.parameter_values
        parameter_binds = {}
        param_array = parameter_values.as_array(circuit.parameters)
        parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}

        arrays = {
            item.creg_name: np.zeros(
                parameter_values.shape + (pub.shots, item.num_bytes), dtype=np.uint8
            )
            for item in meas_info
        }

        if qargs:
            circuit.measure_all()
            result = self._backend.run(
                circuit,
                shots=pub.shots,
                seed_simulator=self._seed,
                parameter_binds=[parameter_binds],
            ).result()
            all_counts = result.get_counts()

            for index, counts in np.ndenumerate(all_counts):
                samples = []
                for k, v in counts.items():
                    k = k.replace(" ", "")
                    kk = ""
                    for q in qargs:
                        kk = k[circuit.num_qubits - 1 - q] + kk

                    samples.append((np.fromiter(kk, dtype=np.uint8), v))

                samples_array = np.array([sample for sample, v in samples for _ in range(0, v)])

                for item in meas_info:
                    ary = _samples_to_packed_array(samples_array, item.num_bits, item.qreg_indices)
                    arrays[item.creg_name][index] = ary
        else:
            for index in np.ndenumerate(parameter_values.shape):
                samples = [""] * pub.shots
                samples_array = np.array(
                    [np.fromiter(sample, dtype=np.uint8) for sample in samples]
                )
                for item in meas_info:
                    ary = _samples_to_packed_array(samples_array, item.num_bits, item.qreg_indices)
                    arrays[item.creg_name][index] = ary

        data_bin_cls = make_data_bin(
            [(item.creg_name, BitArray) for item in meas_info],
            shape=parameter_values.shape,
        )
        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits) for item in meas_info
        }
        data_bin = data_bin_cls(**meas)
        return PubResult(data_bin, metadata={"shots": pub.shots})


def _preprocess_circuit(circuit: QuantumCircuit):
    num_bits_dict = {creg.name: creg.size for creg in circuit.cregs}
    mapping = _final_measurement_mapping(circuit)
    qargs = sorted(set(mapping.values()))
    qargs_index = {v: k for k, v in enumerate(qargs)}
    circuit = circuit.remove_final_measurements(inplace=False)
    if _has_control_flow(circuit):
        raise QiskitError("StatevectorSampler cannot handle ControlFlowOp")
    if _has_measure(circuit):
        raise QiskitError("StatevectorSampler cannot handle mid-circuit measurements")
    # num_qubits is used as sentinel to fill 0 in _samples_to_packed_array
    sentinel = len(qargs)
    indices = {key: [sentinel] * val for key, val in num_bits_dict.items()}
    for key, qreg in mapping.items():
        creg, ind = key
        indices[creg.name][ind] = qargs_index[qreg]
    meas_info = [
        _MeasureInfo(
            creg_name=name,
            num_bits=num_bits,
            num_bytes=_min_num_bytes(num_bits),
            qreg_indices=indices[name],
        )
        for name, num_bits in num_bits_dict.items()
    ]
    return circuit, qargs, meas_info


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, indices: list[int]
) -> NDArray[np.uint8]:
    # samples of `Statevector.sample_memory` will be in the order of
    # qubit_last, ..., qubit_1, qubit_0.
    # reverse the sample order into qubit_0, qubit_1, ..., qubit_last and
    # pad 0 in the rightmost to be used for the sentinel introduced by _preprocess_circuit.
    ary = np.pad(samples[:, ::-1], ((0, 0), (0, 1)), constant_values=0)
    # place samples in the order of clbit_last, ..., clbit_1, clbit_0
    ary = ary[:, indices[::-1]]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1)
    return ary


def _final_measurement_mapping(circuit: QuantumCircuit) -> dict[tuple[ClassicalRegister, int], int]:
    """Return the final measurement mapping for the circuit.

    Parameters:
        circuit: Input quantum circuit.

    Returns:
        Mapping of classical bits to qubits for final measurements.
    """
    active_qubits = set(range(circuit.num_qubits))
    active_cbits = set(range(circuit.num_clbits))

    # Find final measurements starting in back
    mapping = {}
    for item in circuit[::-1]:
        if item.operation.name == "measure":
            loc = circuit.find_bit(item.clbits[0])
            cbit = loc.index
            qbit = circuit.find_bit(item.qubits[0]).index
            if cbit in active_cbits and qbit in active_qubits:
                for creg in loc.registers:
                    mapping[creg] = qbit
                active_cbits.remove(cbit)
        elif item.operation.name not in ["barrier", "delay"]:
            for qq in item.qubits:
                _temp_qubit = circuit.find_bit(qq).index
                if _temp_qubit in active_qubits:
                    active_qubits.remove(_temp_qubit)

        if not active_cbits or not active_qubits:
            break

    return mapping


def _has_control_flow(circuit: QuantumCircuit) -> bool:
    return any(isinstance(instruction.operation, ControlFlowOp) for instruction in circuit)
