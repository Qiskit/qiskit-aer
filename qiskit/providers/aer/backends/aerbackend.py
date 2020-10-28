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
"""
Qiskit Aer qasm simulator backend.
"""

import copy
import json
import logging
import datetime
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from numpy import ndarray

from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendStatus
from qiskit.result import Result

from ..aerjob import AerJob
from ..aererror import AerError

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
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


class AerBackend(BaseBackend, ABC):
    """Qiskit Aer Backend class."""
    def __init__(self,
                 configuration,
                 properties=None,
                 defaults=None,
                 available_methods=None,
                 backend_options=None,
                 provider=None):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (BackendConfiguration): backend configuration.
            properties (BackendProperties or None): Optional, backend properties.
            defaults (PulseDefaults or None): Optional, backend pulse defaults.
            available_methods (list or None): Optional, the available simulation methods
                                              if backend supports multiple methods.
            provider (BaseProvider): Optional, provider responsible for this backend.
            backend_options (dict or None): Optional set custom backend options.

        Raises:
            AerError: if there is no name in the configuration
        """
        # Init configuration and provider in BaseBackend
        configuration.simulator = True
        super().__init__(configuration, provider=provider)

        # Initialize backend properties and pulse defaults.
        self._properties = properties
        self._defaults = defaults

        # Custom configuration, properties, and pulse defaults which will store
        # any configured modifications to the base simulator values.
        self._custom_configuration = None
        self._custom_properties = None
        self._custom_defaults = None

        # Set available methods
        self._available_methods = [] if available_methods is None else available_methods

        # Set custom configured options from backend_options dictionary
        self._options = {}
        if backend_options is not None:
            for key, val in backend_options.items():
                self._set_option(key, val)

    # pylint: disable=arguments-differ
    def run(self,
            qobj,
            backend_options=None,  # DEPRECATED
            validate=False,
            **run_options):
        """Run a qobj on the backend.

        Args:
            qobj (QasmQobj): The Qobj to be executed.
            backend_options (dict or None): DEPRECATED dictionary of backend options
                                            for the execution (default: None).
            validate (bool): validate the Qobj before running (default: False).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Additional Information:
            * kwarg options specified in ``run_options`` will temporarily override
              any set options of the same name for the current run.

            * The entries in the ``backend_options`` will be combined with
              the ``Qobj.config`` dictionary with the values of entries in
              ``backend_options`` taking precedence. This kwarg is deprecated
              and direct kwarg's should be used for options to pass them to
              ``run_options``.
        """
        # DEPRECATED
        if backend_options is not None:
            warnings.warn(
                'Using `backend_options` kwarg has been deprecated as of'
                ' qiskit-aer 0.7.0 and will be removed no earlier than 3'
                ' months from that release date. Runtime backend options'
                ' should now be added directly using kwargs for each option.',
                DeprecationWarning,
                stacklevel=3)

        # Add backend options to the Job qobj
        qobj = self._format_qobj(
            qobj, backend_options=backend_options, **run_options)

        # Optional validation
        if validate:
            self._validate(qobj)

        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run, qobj)
        aer_job.submit()
        return aer_job

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        if self._custom_configuration is not None:
            return self._custom_configuration
        return self._configuration

    def properties(self):
        """Return the simulator backend properties if set.

        Returns:
            BackendProperties: The backend properties or ``None`` if the
                               backend does not have properties set.
        """
        if self._custom_properties is not None:
            return self._custom_properties
        return self._properties

    def defaults(self):
        """Return the simulator backend pulse defaults.

        Returns:
            PulseDefaults: The backend pulse defaults or ``None`` if the
                           backend does not support pulse.
        """
        if self._custom_defaults is not None:
            return self._custom_defaults
        return self._defaults

    @property
    def options(self):
        """Return the current simulator options"""
        return self._options

    def set_options(self, **backend_options):
        """Set the simulator options"""
        for key, val in backend_options.items():
            self._set_option(key, val)

    def clear_options(self):
        """Reset the simulator options to default values."""
        self._custom_configuration = None
        self._custom_properties = None
        self._custom_defaults = None
        self._options = {}

    def available_methods(self):
        """Return the available simulation methods."""
        return self._available_methods

    def status(self):
        """Return backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(
            backend_name=self.name(),
            backend_version=self.configuration().backend_version,
            operational=True,
            pending_jobs=0,
            status_msg='')

    def _run_job(self, job_id, qobj, backend_options, noise_model, validate):
        """Run a qobj job"""
        warnings.warn(
            'The `_run_job` method has been deprecated. Use `_run` instead.',
            DeprecationWarning)
        if validate:
            warnings.warn(
                'The validate arg of `_run_job` has been removed. Use '
                'validate=True in the `run` method instead.',
                DeprecationWarning)

        # The new function swaps positional args qobj and job id so we do a
        # type check to swap them back
        if not isinstance(job_id, str) and isinstance(qobj, str):
            job_id, qobj = qobj, job_id
        run_qobj = self._format_qobj(qobj, backend_options=backend_options,
                                     noise_model=noise_model)
        return self._run(run_qobj, job_id)

    def _run(self, qobj, job_id=''):
        """Run a job"""
        # Start timer
        start = time.time()

        # Run simulation
        output = self._execute(qobj)

        # Validate output
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name())
            if output:
                logger.error('Output: %s', output)
            raise AerError(
                "simulation terminated without returning valid output.")

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version

        # Add execution time
        output["time_taken"] = time.time() - start
        return Result.from_dict(output)

    @abstractmethod
    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj or PulseQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        pass

    def _validate(self, qobj):
        """Validate the qobj for the backend"""
        pass

    def _set_option(self, key, value):
        """Special handling for setting backend options.

        This method should be extended by sub classes to
        update special option values.

        Args:
            key (str): key to update
            value (any): value to update.

        Raises:
            AerError: if key is 'method' and val isn't in available methods.
        """
        # Check for key in configuration, properties, and defaults
        # If the key requires modification of one of these fields a copy
        # will be generated that can store the modified values without
        # changing the original object
        if hasattr(self._configuration, key):
            self._set_configuration_option(key, value)
            return

        if hasattr(self._properties, key):
            self._set_properties_option(key, value)
            return

        if hasattr(self._defaults, key):
            self._set_defaults_option(key, value)
            return

        # If key is method, we validate it is one of the available methods
        if key == 'method' and value not in self._available_methods:
            raise AerError("Invalid simulation method {}. Available methods"
                           " are: {}".format(value, self._available_methods))

        # Add all other options to the options dict
        # TODO: in the future this could be replaced with an options class
        #       for the simulators like configuration/properties to show all
        #       available options
        if value is not None:
            # Only add an option if its value is not None
            self._options[key] = value
        elif key in self._options:
            # If setting an existing option to None remove it from options dict
            self._options.pop(key)

    def _set_configuration_option(self, key, value):
        """Special handling for setting backend configuration options."""
        if self._custom_configuration is None:
            self._custom_configuration = copy.copy(self._configuration)
        setattr(self._custom_configuration, key, value)

    def _set_properties_option(self, key, value):
        """Special handling for setting backend properties options."""
        if self._custom_properties is None:
            self._custom_properties = copy.copy(self._properties)
        setattr(self._custom_properties, key, value)

    def _set_defaults_option(self, key, value):
        """Special handling for setting backend defaults options."""
        if self._custom_defaults is None:
            self._custom_defaults = copy.copy(self._defaults)
        setattr(self._custom_defaults, key, value)

    def _format_qobj(self, qobj,
                     backend_options=None,  # DEPRECATED
                     **run_options):
        """Return execution sim config dict from backend options."""
        # Add options to qobj config overriding any existing fields
        config = qobj.config

        # Add options
        for key, val in self.options.items():
            setattr(config, key, val)

        # DEPRECATED backend options
        if backend_options is not None:
            for key, val in backend_options.items():
                setattr(config, key, val)

        # Override with run-time options
        for key, val in run_options.items():
            setattr(config, key, val)

        return qobj

    def _run_config(
            self,
            backend_options=None,  # DEPRECATED
            **run_options):
        """Return execution sim config dict from backend options."""
        # Get sim config
        run_config = self._options.copy()

        # DEPRECATED backend options
        if backend_options is not None:
            for key, val in backend_options.items():
                run_config[key] = val

        # Override with run-time options
        for key, val in run_options.items():
            run_config[key] = val
        return run_config

    def __repr__(self):
        """String representation of an AerBackend."""
        display = "backend_name='{}'".format(self.name())
        if self.provider():
            display += ', provider={}()'.format(self.provider())
        for key, val in self.options.items():
            display += ',\n    {}={}'.format(key, repr(val))
        return '{}(\n{})'.format(self.__class__.__name__, display)
