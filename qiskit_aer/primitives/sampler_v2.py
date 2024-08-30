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
from qiskit.primitives.backend_sampler_v2 import (
    _analyze_circuit,
    _MeasureInfo,
    _memory_array,
    _samples_to_packed_array,
)
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.result import Result

from qiskit_aer import AerSimulator


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
        seed: int | None = None,
        options: dict | None = None,
    ):
        """
        Args:
            default_shots: The default shots for the sampler if not specified during run.
            seed: The seed for random number generation.
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
        if isinstance(backend, AerSimulator):
            sampler._backend = backend
        else:
            sampler._backend = AerSimulator.from_backend(backend)
        return sampler

    @property
    def default_shots(self) -> int:
        """Return the default shots"""
        return self._default_shots

    @property
    def seed(self) -> int | None:
        """Return the seed for random number generation."""
        return self._seed

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._default_shots
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
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[SamplerPubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        circuits = [pub.circuit for pub in pubs]
        parameter_binds = [_convert_parameter_bindings(pub) for pub in pubs]

        # adjust run_options not to overwrite existings options
        run_options = self.options.run_options.copy()
        for key in ["shots", "parameter_binds", "memory"]:
            if key in run_options:
                del run_options[key]
        if self._seed is not None and "seed_simulator" in run_options:
            del run_options["seed_simulator"]

        # run circuits
        result = self._backend.run(
            circuits,
            shots=shots,
            seed_simulator=self._seed,
            parameter_binds=parameter_binds,
            memory=True,
            **run_options,
        ).result()

        result_memory = _prepare_memory(result)

        # pack memory to an ndarray of uint8
        results = []
        start = 0
        for pub in pubs:
            meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
            p_v = pub.parameter_values
            end = start + p_v.size
            results.append(
                self._postprocess_pub(
                    result_memory[start:end],
                    shots,
                    p_v.shape,
                    meas_info,
                    max_num_bytes,
                    pub.circuit.metadata,
                    result.metadata,
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
        circuit_metadata: dict,
        simulator_metadata: dict,
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
        return SamplerPubResult(
            DataBin(**meas, shape=shape),
            metadata={
                "shots": shots,
                "circuit_metadata": circuit_metadata,
                "simulator_metadata": simulator_metadata,
            },
        )


def _convert_parameter_bindings(pub: SamplerPub) -> dict:
    circuit = pub.circuit
    parameter_values = pub.parameter_values
    parameter_binds = {}
    param_array = parameter_values.as_array(circuit.parameters)
    parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}
    return parameter_binds


def _prepare_memory(result: Result) -> list[list[str]]:
    """There is no split of results due to max_experiments for Aer"""
    lst = []
    for exp in result.results:
        if hasattr(exp.data, "memory") and exp.data.memory:
            lst.append(exp.data.memory)
        else:
            # no measure in a circuit
            lst.append(["0x0"] * exp.shots)
    return lst
