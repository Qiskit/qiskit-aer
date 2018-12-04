# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qiskit Aer qasm simulator backend.
"""

import json
import logging
import datetime
import uuid
import numpy as np
from numbers import Number

import qiskit
from qiskit.backends import BaseBackend
from qiskit.qobj import qobj_to_dict
from qiskit.result._result import Result
from .aerjob import AerJob
from .aersimulatorerror import AerSimulatorError
from ..noise import NoiseModel

# Logger
logger = logging.getLogger(__name__)


class AerJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        return super().default(obj)


class AerJSONDecoder(json.JSONDecoder):
    """
    JSON decoder for complex expectation value snapshots.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    # pylint: disable=method-hidden
    def object_hook(self, obj):
        """Special decoding rules for simulator output."""
        # Decode statevector
        if 'statevector' in obj:
            obj['statevector'] = self._decode_complex_vector(obj['statevector'])
        # Decode unitary
        if 'unitary' in obj:
            obj['unitary'] = self._decode_complex_matrix(obj['unitary'])
        # Decode snapshots
        if 'snapshots' in obj:
            # Decode statevector snapshot
            if 'statevector' in obj['snapshots']:
                for key, val in obj['snapshots']['statevector'].items():
                    obj['snapshots']['statevector'][key] = [
                        self._decode_complex_vector(vec) for vec in val]
            # Decode unitary snapshot
            if 'unitary' in obj['snapshots']:
                for key, val in obj['snapshots']['unitary'].items():
                    obj['snapshots']['unitary'][key] = [
                        self._decode_complex_matrix(mat) for mat in val]
            # Decode expectation value snapshot
            if 'expectation_value' in obj['snapshots']:
                for key, val in obj['snapshots']['expectation_value'].items():
                    for j, expval in enumerate(val):
                        val[j]['value'] = self._decode_complex(expval['value'])
        return obj

    def _decode_complex(self, obj):
        """Deserialize JSON real or complex number"""
        if isinstance(obj, list) and np.shape(obj) == (2,) and \
           isinstance(obj[0], Number) and isinstance(obj[1], Number):
            if obj[1] == 0.:
                obj = obj[0]
            else:
                obj = obj[0] + 1j * obj[1]
        return obj

    def _decode_complex_vector(self, obj):
        """Deserialize JSON real or complex vector"""
        if isinstance(obj, list):
            obj = np.array([self._decode_complex(z) for z in obj])
        return obj

    def _decode_complex_matrix(self, obj):
        """Deserialize JSON real or complex matrix"""
        if isinstance(obj, list):
            shape = np.shape(obj)
            # Check if dimension is consistant with complex or real matrix
            if len(shape) in [3, 2]:
                obj = np.array([[self._decode_complex(z) for z in row]
                                for row in obj])
        return obj


class AerBackend(BaseBackend):
    """Qiskit Aer Backend class."""

    def __init__(self, configuration, controller_wrapper, provider=None,
                 json_decoder=AerJSONDecoder):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (dict): configuration dictionary
            controller_wrapper (class): Aer Controller cython wrapper class
            provider (BaseProvider): provider responsible for this backend
            json_decoder (JSONDecoder): JSON decoder for simulator output

        Raises:
            FileNotFoundError if backend executable is not available.
            QISKitError: if there is no name in the configuration
        """
        super().__init__(configuration, provider=provider)
        # Extract the default basis gates set from backend configuration
        self._controller = controller_wrapper
        self._json_decoder = json_decoder

    def reset(self):
        """Reset the Aer Backend.

        This clears any set noise model or config.
        """
        self.clear_config()
        self.clear_noise_model()

    def run(self, qobj, backend_options=None, noise_model=None):
        """Run a qobj on the backend."""
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj, backend_options, noise_model)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj, backend_options, noise_model):
        """Run a qobj job"""
        self._validate(qobj)
        qobj_str = json.dumps(qobj.as_dict(), cls=AerJSONEncoder)
        options_str = json.dumps(backend_options, cls=AerJSONEncoder)
        if isinstance(noise_model, NoiseModel):
            noise_model = noise_model.as_dict()
        elif not isinstance(noise_model, dict) and noise_model is not None:
            raise AerSimulatorError("Invalid Qiskit Aer noise model.")
        noise_str = json.dumps(noise_model, cls=AerJSONEncoder)
        output = json.loads(self._controller.execute(qobj_str, options_str, noise_str),
                            cls=self._json_decoder)
        self._validate_controller_output(output)
        return self._format_results(job_id, output)

    def _format_results(self, job_id, output):
        """Construct Result object from simulator output."""
        # Add result metadata
        output["job_id"] = job_id
        output["date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        output["backend_name"] = self.DEFAULT_CONFIGURATION['name']
        output["backend_version"] = "0.0.1"  # TODO: get this from somewhere else
        # Parse result dict into Result class
        exp_results = output.get("results", {})
        experiment_names = [data.get("header", {}).get("name", None)
                            for data in exp_results]
        qobj_result = qiskit.qobj.Result(**output)
        qobj_result.results = [qiskit.qobj.ExperimentResult(**res) for res in exp_results]
        return Result(qobj_result, experiment_names=experiment_names)

    def _validate_controller_output(self, output):
        """Validate output from the controller wrapper."""
        # Check results
        # TODO: Once https://github.com/Qiskit/qiskit-terra/issues/1023
        #       is merged this should be updated to deal with errors using
        #       the Result object methods
        if not output.get("success", False):
            logger.error("AerBackend: simulation failed")
            # Check for error message in the failed circuit
            for res in output.get('results'):
                if not res.get('success', False):
                    raise AerSimulatorError(res.get("status", None))
            # If no error was found check for error message at qobj level
            raise AerSimulatorError(output.get("status", None))

    def set_max_threads_shot(self, threads):
        """
        Set the maximum threads used for parallel shot execution.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available

        Note that using parallel shot evaluation disables parallel circuit
        evaluation.
        """
        self._controller.set_max_threads_shot(int(threads))

    def set_max_threads_circuit(self, threads):
        """
        Set the maximum threads used for parallel circuit execution.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available

        Note that using parallel circuit evaluation disables parallel shot
        evaluation.
        """
        self._controller.set_max_threads_circuit(int(threads))

    def set_max_threads_state(self, threads):
        """
        Set the maximum threads used for state update parallel  routines.

        Args:
            threads (int): the thread limit, set to -1 to use maximum available.

        Note that using parallel circuit or shot execution takes precidence over
        parallel state evaluation.
        """
        self._controller.set_max_threads_state(int(threads))

    def _validate(self, qobj):
        # TODO
        return

    def __repr__(self):
        """Official string representation of an AerBackend."""
        display = "{}('{}')".format(self.__class__.__name__, self.name())
        if self.provider is not None:
            display = display + " from {}()".format(self.provider)
        return "<" + display + ">"
