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

import copy
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
from qiskit_aer.noise.noise_model import NoiseModel


@dataclass
class TemporalDriftOptions:
    """Options controlling temporal drift for non-stationary noise simulation."""

    sigma: float = 0.0
    """Lognormal sigma controlling temporal drift intensity."""

    window_size: int | None = None
    """Number of shots per chunk when temporal drift is enabled."""


@dataclass
class Options:
    """Options for :class:`~.SamplerV2`."""

    backend_options: dict = field(default_factory=dict)
    """backend_options: Options passed to AerSimulator."""

    run_options: dict = field(default_factory=dict)
    """run_options: Options passed to run."""

    temporal_drift: TemporalDriftOptions = field(default_factory=TemporalDriftOptions)
    """temporal_drift: Options for non-Markovian temporal noise drift."""


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
        if isinstance(self._options.temporal_drift, dict):
            self._options.temporal_drift = TemporalDriftOptions(**self._options.temporal_drift)
        self._validate_options()
        self._backend = AerSimulator(**self.options.backend_options)
        self._rng = np.random.default_rng(seed)

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

    def _validate_options(self):
        temporal_drift = self._options.temporal_drift
        if temporal_drift.sigma < 0:
            raise ValueError("temporal_drift.sigma must be non-negative")
        if temporal_drift.window_size is not None and temporal_drift.window_size <= 0:
            raise ValueError("temporal_drift.window_size must be a positive integer")

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
        if self._temporal_drift_active():
            return self._run_with_temporal_drift(pubs)

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

    def _run_with_temporal_drift(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        results = [None] * len(pubs)
        for index, pub in enumerate(pubs):
            results[index] = self._run_drifted_pub(pub)
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_drifted_pub(self, pub: SamplerPub) -> SamplerPubResult:
        total_shots = pub.shots
        chunk_results = []
        drift_trajectory = []
        original_noise_model = getattr(self._backend.options, "noise_model", None)

        for chunk_shots in self._chunk_shots(total_shots):
            perturbed_noise_model, factor = self._perturb_noise_model(original_noise_model)
            drift_trajectory.append(factor)
            try:
                self._backend.set_options(noise_model=perturbed_noise_model)
                chunk_results.extend(self._run_pubs([pub], chunk_shots))
            finally:
                self._backend.set_options(noise_model=original_noise_model)

        stitched = self._stitch_pub_results(chunk_results, total_shots)
        stitched.metadata["temporal_drift_trajectory"] = drift_trajectory
        stitched.metadata["temporal_drift"] = {
            "sigma": self.options.temporal_drift.sigma,
            "window_size": self.options.temporal_drift.window_size,
        }
        return stitched

    def _temporal_drift_active(self) -> bool:
        temporal_drift = self.options.temporal_drift
        return temporal_drift.sigma > 0 and temporal_drift.window_size is not None

    def _chunk_shots(self, total_shots: int) -> list[int]:
        window_size = self.options.temporal_drift.window_size
        if window_size is None or window_size <= 0:
            return [total_shots]
        chunk_sizes = []
        remaining = total_shots
        while remaining > 0:
            chunk = min(window_size, remaining)
            chunk_sizes.append(chunk)
            remaining -= chunk
        return chunk_sizes

    def _perturb_noise_model(
        self, baseline_noise_model: NoiseModel | None
    ) -> tuple[NoiseModel | None, float]:
        factor = float(self._rng.lognormal(mean=0.0, sigma=self.options.temporal_drift.sigma))
        if baseline_noise_model is None:
            return None, factor

        perturbed_noise_model = copy.deepcopy(baseline_noise_model)

        for name, error in perturbed_noise_model._default_quantum_errors.items():
            perturbed_noise_model._default_quantum_errors[name] = _perturb_quantum_error(error, factor)

        for name, errors in perturbed_noise_model._local_quantum_errors.items():
            for qubits, error in errors.items():
                perturbed_noise_model._local_quantum_errors[name][qubits] = _perturb_quantum_error(
                    error, factor
                )

        return perturbed_noise_model, factor

    def _stitch_pub_results(
        self, chunk_results: list[SamplerPubResult], total_shots: int
    ) -> SamplerPubResult:
        first_result = chunk_results[0]
        field_names = [name for name in vars(first_result.data) if name != "shape"]
        stitched_data = {}
        for name in field_names:
            bitarrays = [getattr(result.data, name) for result in chunk_results]
            stitched_array = np.concatenate([bitarray.array for bitarray in bitarrays], axis=-2)
            stitched_data[name] = BitArray(stitched_array, bitarrays[0].num_bits)

        metadata = dict(first_result.metadata)
        metadata["shots"] = total_shots
        metadata["simulator_metadata"] = [
            result.metadata.get("simulator_metadata", {}) for result in chunk_results
        ]
        metadata["chunk_shots"] = [result.metadata.get("shots") for result in chunk_results]
        return SamplerPubResult(
            DataBin(**stitched_data, shape=first_result.data.shape),
            metadata=metadata,
        )

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


def _perturb_cptp_probabilities(probabilities: list[float], factor: float) -> list[float]:
    """Perturb only non-identity probability mass and reconstruct the identity branch."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.size == 0:
        return []

    identity_index = int(np.argmax(probs))
    error_indices = [index for index in range(probs.size) if index != identity_index]
    error_probs = probs[error_indices]
    scaled_error_probs = np.clip(error_probs * factor, 0.0, None)
    error_mass = float(np.sum(scaled_error_probs))
    if error_mass > 0.999:
        scaled_error_probs *= 0.999 / error_mass
        error_mass = 0.999

    perturbed = np.zeros_like(probs)
    perturbed[error_indices] = scaled_error_probs
    perturbed[identity_index] = 1.0 - error_mass
    return perturbed.tolist()


def _perturb_quantum_error(error, factor: float):
    """Return a perturbed copy of a quantum error with CPTP-preserving probabilities."""
    probabilities = _perturb_cptp_probabilities(list(error.probabilities), factor)
    return error.__class__(zip(error.circuits, probabilities))
