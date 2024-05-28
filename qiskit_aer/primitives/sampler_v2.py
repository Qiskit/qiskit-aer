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

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import QuantumCircuit
from qiskit.primitives.backend_estimator import _run_circuits
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.bit_array import _min_num_bytes
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.result import Result

from qiskit_aer import AerSimulator


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int

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

    @classmethod
    def from_backend(cls, backend, **options):
        """make new sampler that uses external backend"""
        sampler = cls(**options)
        sampler._backend = AerSimulator.from_backend(backend)
        return sampler

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
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._options.default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[SamplerPub]):
        for i, pub in enumerate(pubs):
            if len(pub.circuit.cregs) == 0:
                warnings.warn(
                    f"The {i}-th pub's circuit has no output classical registers and so the result "
                    "will be empty. Did you mean to add measurement instructions?",
                    UserWarning,
                )

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            pub_dict[pub.shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results)

    def _run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[SamplerPubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        circuits = [pub.circuit for pub in pubs]
        parameter_binds = [_convert_parameter_bindings(pub) for pub in pubs]

        # run circuits
        results = self._backend.run(
            circuits,
            shots=shots,
            seed_simulator=self._seed,
            parameter_binds=parameter_binds,
        ).result()
        print(results)

        result_memory = _prepare_memory(results)

        # pack memory to an ndarray of uint8
        results = []
        start = 0
        for pub, bound in zip(pubs, bound_circuits):
            meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
            end = start + bound.size
            results.append(
                self._postprocess_pub(
                    result_memory[start:end], shots, bound.shape, meas_info, max_num_bytes
                )
            )
            start = end

        return results

    def _postprocess_pub(
        self,
        result_memory: list[list[str]],
        shots: int,
        shape: tuple[int, ...],
        meas_info: list[_MeasureInfo],
        max_num_bytes: int,
    ) -> SamplerPubResult:
        """Converts the memory data into an array of bit arrays with the shape of the pub."""
        arrays = {
            item.creg_name: np.zeros(shape + (shots, item.num_bytes), dtype=np.uint8)
            for item in meas_info
        }
        memory_array = _memory_array(result_memory, max_num_bytes)

        for samples, index in zip(memory_array, np.ndindex(*shape)):
            for item in meas_info:
                ary = _samples_to_packed_array(samples, item.num_bits, item.start)
                arrays[item.creg_name][index] = ary

        meas = {
            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits) for item in meas_info
        }
        return SamplerPubResult(DataBin(**meas, shape=shape), metadata={})

    def __run_pub(self, pub: SamplerPub) -> PubResult:
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

        metadata = {"shots": pub.shots}
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
            metadata["simulator_metadata"] = result.metadata
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
        return PubResult(data_bin, metadata=metadata)


def _convert_parameter_bindings(pub: SamplerPub) -> dict:
    circuit = pub.circuit
    parameter_values = pub.parameter_values
    parameter_binds = {}
    param_array = parameter_values.as_array(circuit.parameters)
    parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}
    return parameter_binds


def _analyze_circuit(circuit: QuantumCircuit) -> tuple[list[_MeasureInfo], int]:
    """Analyzes the information for each creg in a circuit."""
    meas_info = []
    max_num_bits = 0
    for creg in circuit.cregs:
        name = creg.name
        num_bits = creg.size
        if num_bits != 0:
            start = circuit.find_bit(creg[0]).index
        else:
            start = 0
        meas_info.append(
            _MeasureInfo(
                creg_name=name,
                num_bits=num_bits,
                num_bytes=_min_num_bytes(num_bits),
                start=start,
            )
        )
        max_num_bits = max(max_num_bits, start + num_bits)
    return meas_info, _min_num_bytes(max_num_bits)


def _prepare_memory(results: list[Result]) -> list[list[str]]:
    """Joins splitted results if exceeding max_experiments"""
    lst = []
    for res in results:
        for exp in res.results:
            if hasattr(exp.data, "memory") and exp.data.memory:
                lst.append(exp.data.memory)
            else:
                # no measure in a circuit
                lst.append(["0x0"] * exp.shots)
    return lst


def _memory_array(results: list[list[str]], num_bytes: int) -> NDArray[np.uint8]:
    """Converts the memory data into an array in an unpacked way."""
    lst = []
    for memory in results:
        if num_bytes > 0:
            data = b"".join(int(i, 16).to_bytes(num_bytes, "big") for i in memory)
            data = np.frombuffer(data, dtype=np.uint8).reshape(-1, num_bytes)
        else:
            # no measure in a circuit
            data = np.zeros((len(memory), num_bytes), dtype=np.uint8)
        lst.append(data)
    ary = np.asarray(lst)
    return np.unpackbits(ary, axis=-1, bitorder="big")


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, start: int
) -> NDArray[np.uint8]:
    """Converts an unpacked array of the memory data into a packed array."""
    # samples of `Backend.run(memory=True)` will be the order of
    # clbit_last, ..., clbit_1, clbit_0
    # place samples in the order of clbit_start+num_bits-1, ..., clbit_start+1, clbit_start
    if start == 0:
        ary = samples[:, -start - num_bits :]
    else:
        ary = samples[:, -start - num_bits : -start]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1, bitorder="big")
    return ary
