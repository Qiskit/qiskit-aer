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

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV1 as Backend
from qiskit.providers.models import BackendStatus
from qiskit.result import Result
from qiskit.utils import deprecate_arguments
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.compiler import assemble

from ..jobs import AerJob, AerJobSet, split_qobj
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


class AerBackend(Backend, ABC):
    """Qiskit Aer Backend class."""

    def __init__(self,
                 configuration,
                 properties=None,
                 defaults=None,
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
            provider (Provider): Optional, provider responsible for this backend.
            backend_options (dict or None): Optional set custom backend options.

        Raises:
            AerError: if there is no name in the configuration
        """
        # Init configuration and provider in Backend
        configuration.simulator = True
        configuration.local = True
        super().__init__(configuration, provider=provider)

        # Initialize backend properties and pulse defaults.
        self._properties = properties
        self._defaults = defaults

        # Custom option values for config, properties, and defaults
        self._options_configuration = {}
        self._options_defaults = {}
        self._options_properties = {}

        # Set options from backend_options dictionary
        if backend_options is not None:
            self.set_options(**backend_options)

    # pylint: disable=arguments-differ
    @deprecate_arguments({'qobj': 'circuits'})
    def run(self,
            circuits,
            validate=False,
            **run_options):
        """Run a qobj on the backend.

        Args:
            circuits (QuantumCircuit or list): The QuantumCircuit (or list
                of QuantumCircuit objects) to run
            validate (bool): validate the Qobj before running (default: False).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Additional Information:
            kwarg options specified in ``run_options`` will temporarily override
            any set options of the same name for the current run.

        Raises:
            ValueError: if run is not implemented
        """
        if (isinstance(circuits, list) and
                all(isinstance(circuit, QuantumCircuit) for circuit in circuits) and
                not hasattr(self._options, 'executor')):
            return self._run_circuits(circuits, validate, **run_options)
        if isinstance(circuits, (QasmQobj, PulseQobj)):
            warnings.warn('Using a qobj for run() is deprecated and will be '
                          'removed in a future release.',
                          PendingDeprecationWarning,
                          stacklevel=2)
        else:
            circuits = assemble(circuits, self)
        return self._run_qobj(circuits, validate, **run_options)

    def _run_qobj(self,
                  qobj,
                  validate=False,
                  **run_options):
        """Run a qobj on the backend.

        Args:
            qobj (QasmQobj or PulseQobj): qobj to run
            validate (bool): validate the Qobj before running (default: False).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Additional Information:
            kwarg options specified in ``run_options`` will temporarily override
            any set options of the same name for the current run.

        Raises:
            ValueError: if run is not implemented
        """
        # A work around to support both qobj options and run options until
        # qobj is deprecated is to copy all the set qobj.config fields into
        # run_options that don't override existing fields. This means set
        # run_options fields will take precidence over the value for those
        # fields that are set via assemble.
        if not run_options:
            run_options = qobj.config.__dict__
        else:
            run_options = copy.copy(run_options)
            for key, value in qobj.config.__dict__.items():
                if key not in run_options and value is not None:
                    run_options[key] = value

        # Add submit args for the job
        self._add_options_to_qobj_config(qobj, **run_options)

        # Optional validation
        if validate:
            self._validate(qobj)

        # Split circuits for sub-jobs
        experiments = split_qobj(
            qobj, max_size=getattr(qobj.config, 'max_job_size', None))

        # Avoid serialization of self._options._executor only in submit()
        executor = None
        if hasattr(self._options, 'executor'):
            executor = getattr(self._options, 'executor')
            # We need to remove the executor from the qobj config
            # since it can't be serialized though JSON/Pybind.
            delattr(self._options, 'executor')

        # Submit job
        job_id = str(uuid.uuid4())
        if isinstance(experiments, list):
            aer_job = AerJobSet(self, job_id, self._run, experiments, executor=executor)
        else:
            aer_job = AerJob(self, job_id, self._run, experiments, executor=executor)

        # Restore self._options
        if executor:
            setattr(self._options, 'executor', executor)

        aer_job.submit()

        return aer_job

    def _run_circuits(self,
                      circuits,
                      validate=False,
                      **run_options):
        """Run a qobj on the backend.

        Args:
            circuits (QuantumCircuit or list): QuantumCircuit or its List to run
            validate (bool): validate the circuits before running (default: False).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Additional Information:
            kwarg options specified in ``run_options`` will temporarily override
            any set options of the same name for the current run.

        Raises:
            ValueError: if run is not implemented
        """
        # Optional validation
        if validate:
            self._validate(assemble(circuits, self))

        # Generate configuration
        config = self.configuration()
        # Add options
        for key, val in self.options.__dict__.items():
            if val is not None and not hasattr(config, key):
                setattr(config, key, val)
        # Override with run-time options
        for key, val in run_options.items():
            setattr(config, key, val)

        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run, circuits, config=config)

        aer_job.submit()

        return aer_job

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        config = copy.copy(self._configuration)
        for key, val in self._options_configuration.items():
            setattr(config, key, val)
        # If config has custom instructions add them to
        # basis gates to include them for the terra transpiler
        if hasattr(config, 'custom_instructions'):
            config.basis_gates = config.basis_gates + config.custom_instructions
        return config

    def properties(self):
        """Return the simulator backend properties if set.

        Returns:
            BackendProperties: The backend properties or ``None`` if the
                               backend does not have properties set.
        """
        properties = copy.copy(self._properties)
        for key, val in self._options_properties.items():
            setattr(properties, key, val)
        return properties

    def defaults(self):
        """Return the simulator backend pulse defaults.

        Returns:
            PulseDefaults: The backend pulse defaults or ``None`` if the
                           backend does not support pulse.
        """
        defaults = copy.copy(self._defaults)
        for key, val in self._options_defaults.items():
            setattr(defaults, key, val)
        return defaults

    @classmethod
    def _default_options(cls):
        pass

    def clear_options(self):
        """Reset the simulator options to default values."""
        self._options = self._default_options()
        self._options_configuration = {}
        self._options_properties = {}
        self._options_defaults = {}

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

    def _run(self, circuits, job_id='', config=None):
        """Run a job"""
        # Start timer
        start = time.time()

        # Run simulation
        output = self._execute(circuits, config)

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

        # Display warning if simulation failed
        if not output.get("success", False):
            msg = "Simulation failed"
            if "status" in output:
                msg += f" and returned the following error message:\n{output['status']}"
            logger.warning(msg)

        return Result.from_dict(output)

    @abstractmethod
    def _execute(self, qobj, config):
        """Execute circuits on the backend.

        Args:
            qobj (QasmQobj or PulseQobj or list): simulator input.
            config (BackendConfiguration): simulation config.

        Returns:
            dict: return a dictionary of results.
        """
        pass

    def _validate(self, qobj):
        """Validate the qobj for the backend"""
        pass

    def set_option(self, key, value):
        """Special handling for setting backend options.

        This method should be extended by sub classes to
        update special option values.

        Args:
            key (str): key to update
            value (any): value to update.

        Raises:
            AerError: if key is 'method' and val isn't in available methods.
        """
        # Add all other options to the options dict
        # TODO: in the future this could be replaced with an options class
        #       for the simulators like configuration/properties to show all
        #       available options
        if hasattr(self._configuration, key):
            self._set_configuration_option(key, value)
        elif hasattr(self._properties, key):
            self._set_properties_option(key, value)
        elif hasattr(self._defaults, key):
            self._set_defaults_option(key, value)
        else:
            if not hasattr(self._options, key):
                raise AerError("Invalid option %s" % key)
            if value is not None:
                # Only add an option if its value is not None
                setattr(self._options, key, value)
            else:
                # If setting an existing option to None reset it to default
                # this is for backwards compatibility when setting it to None would
                # remove it from the options dict
                setattr(self._options, key, getattr(self._default_options(), key))

    def set_options(self, **fields):
        """Set the simulator options"""
        for key, value in fields.items():
            self.set_option(key, value)

    def _set_configuration_option(self, key, value):
        """Special handling for setting backend configuration options."""
        if value is not None:
            self._options_configuration[key] = value
        elif key in self._options_configuration:
            self._options_configuration.pop(key)

    def _set_properties_option(self, key, value):
        """Special handling for setting backend properties options."""
        if value is not None:
            self._options_properties[key] = value
        elif key in self._options_properties:
            self._options_properties.pop(key)

    def _set_defaults_option(self, key, value):
        """Special handling for setting backend defaults options."""
        if value is not None:
            self._options_defaults[key] = value
        elif key in self._options_defaults:
            self._options_defaults.pop(key)

    def _add_options_to_qobj_config(self, qobj, **run_options):
        """Return execution sim config dict from backend options."""
        # Add options to qobj config overriding any existing fields
        config = qobj.config

        # Add options
        for key, val in self.options.__dict__.items():
            if val is not None:
                setattr(config, key, val)

        # Override with run-time options
        for key, val in run_options.items():
            setattr(config, key, val)

    def __repr__(self):
        """String representation of an AerBackend."""
        name = self.__class__.__name__
        display = f"'{self.name()}'"
        return f'{name}({display})'
