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
Aer qasm simulator backend.
"""

import copy
import datetime
import logging
import time
import uuid
import warnings
from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit, ParameterExpression, Delay
from qiskit.compiler import assemble
from qiskit.providers import BackendV2 as Backend
from qiskit.providers.models import BackendStatus
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.result import Result
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.target import Target
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from ..aererror import AerError
from ..jobs import AerJob, AerJobSet, split_qobj
from ..noise.noise_model import NoiseModel, QuantumErrorLocation
from ..noise.errors.base_quantum_error import QuantumChannelInstruction
from .aer_compiler import compile_circuit, assemble_circuits, generate_aer_config
from .backend_utils import format_save_type, circuit_optypes
from .name_mapping import NAME_MAPPING

# pylint: disable=import-error, no-name-in-module, abstract-method
from .controller_wrappers import AerConfig

# Logger
logger = logging.getLogger(__name__)


class AerBackend(Backend, ABC):
    """Aer Backend class."""

    def __init__(self, configuration, provider=None, target=None, backend_options=None):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (dict): backend configuration.
            provider (Provider): Optional, provider responsible for this backend.
            target (Target):  initial target for backend
            backend_options (dict or None): Optional set custom backend options.

        Raises:
            AerError: if there is no name in the configuration
        """
        # Init configuration and provider in Backend
        super().__init__(
            provider=provider,
            name=configuration["backend_name"],
            description=configuration["description"],
            backend_version=configuration["backend_version"],
        )

        # Initialize backend configuration
        self._configuration = configuration

        # Custom option values for config
        self._options_configuration = {}
        self._target = target
        self._mapping = NAME_MAPPING
        if target is not None:
            self._from_backend = True
        else:
            self._from_backend = False

        # Set options from backend_options dictionary
        if backend_options is not None:
            self.set_options(**backend_options)

        # build coupling map
        if self.configuration()["coupling_map"] is not None:
            self._coupling_map = CouplingMap(self.configuration()["coupling_map"])

    def _convert_circuit_binds(self, circuit, binds, idx_map):
        parameterizations = []

        def append_param_values(index, bind_pos, param):
            if param in binds:
                parameterizations.append([(index, bind_pos), binds[param]])
            elif isinstance(param, ParameterExpression):
                # If parameter expression has no unbound parameters
                # it's already bound and should be skipped
                if not param.parameters:
                    return
                if not binds:
                    raise AerError("The element of parameter_binds is empty.")
                len_vals = len(next(iter(binds.values())))
                bind_list = [
                    {
                        parameter: binds[parameter][i]
                        for parameter in param.parameters & binds.keys()
                    }
                    for i in range(len_vals)
                ]
                bound_values = [float(param.bind(x)) for x in bind_list]
                parameterizations.append([(index, bind_pos), bound_values])

        append_param_values(AerConfig.GLOBAL_PHASE_POS, -1, circuit.global_phase)

        for index, instruction in enumerate(circuit.data):
            if instruction.operation.is_parameterized():
                for bind_pos, param in enumerate(instruction.operation.params):
                    append_param_values(idx_map[index] if idx_map else index, bind_pos, param)
        return parameterizations

    def _convert_binds(self, circuits, parameter_binds, idx_maps=None):
        if isinstance(circuits, QuantumCircuit):
            if len(parameter_binds) > 1:
                raise AerError("More than 1 parameter table provided for a single circuit")

            return [self._convert_circuit_binds(circuits, parameter_binds[0], None)]
        elif len(parameter_binds) != len(circuits):
            raise AerError(
                "Number of input circuits does not match number of input "
                "parameter bind dictionaries"
            )
        parameterizations = [
            self._convert_circuit_binds(circuit, parameter_binds[idx], idx_maps[idx])
            for idx, circuit in enumerate(circuits)
        ]
        return parameterizations

    # pylint: disable=arguments-renamed
    def run(self, circuits, parameter_binds=None, **run_options):
        """Run circuits on the backend.

        Args:
            circuits (QuantumCircuit or list): The QuantumCircuit (or list
                of QuantumCircuit objects) to run
            parameter_binds (list): A list of parameter binding dictionaries.
                                    See additional information (default: None).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Raises:
            TypeError: If ``parameter_binds`` is specified with an input or
                has a length mismatch with the number of circuits.

        Additional Information:
            * Each parameter binding dictionary is of the form::

                {
                    param_a: [val_1, val_2],
                    param_b: [val_3, val_1],
                }

              for all parameters in that circuit. The length of the value
              list must be the same for all parameters, and the number of
              parameter dictionaries in the list must match the length of
              ``circuits`` (if ``circuits`` is a single ``QuantumCircuit``
              object it should a list of length 1).
            * kwarg options specified in ``run_options`` will temporarily override
              any set options of the same name for the current run.

        Raises:
            ValueError: if run is not implemented
        """
        if isinstance(circuits, (QuantumCircuit, Schedule, ScheduleBlock)):
            circuits = [circuits]

        return self._run_circuits(circuits, parameter_binds, **run_options)

    def _run_circuits(self, circuits, parameter_binds, **run_options):
        """Run circuits by generating native circuits."""
        # Submit job
        job_id = str(uuid.uuid4())
        aer_job = AerJob(
            self,
            job_id,
            self._execute_circuits_job,
            parameter_binds=parameter_binds,
            circuits=circuits,
            run_options=run_options,
        )
        aer_job.submit()

        return aer_job

    # pylint: disable=arguments-differ
    def _run_qobj(self, circuits, validate=False, parameter_binds=None, **run_options):
        """Run circuits by assembling qobj."""
        qobj = self._assemble(circuits, parameter_binds=parameter_binds, **run_options)

        # Optional validation
        if validate:
            self._validate(qobj)

        # Get executor from qobj config and delete attribute so qobj can still be serialized
        executor = getattr(qobj.config, "executor", None)
        if hasattr(qobj.config, "executor"):
            delattr(qobj.config, "executor")

        # Optionally split the job
        experiments = split_qobj(
            qobj,
            max_size=getattr(qobj.config, "max_job_size", None),
            max_shot_size=getattr(qobj.config, "max_shot_size", None),
        )

        # Temporarily remove any executor from options so that job submission
        # can work with Dask client executors which can't be pickled
        opts_executor = getattr(self._options, "executor", None)
        if hasattr(self._options, "executor"):
            self._options.executor = None

        # Submit job
        job_id = str(uuid.uuid4())
        if isinstance(experiments, list):
            aer_job = AerJobSet(self, job_id, self._execute_qobj_job, experiments, executor)
        else:
            aer_job = AerJob(
                self, job_id, self._execute_qobj_job, qobj=experiments, executor=executor
            )
        aer_job.submit()

        # Restore removed executor after submission
        if hasattr(self._options, "executor"):
            self._options.executor = opts_executor

        return aer_job

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        config = self._configuration.copy()
        # If config has custom instructions add them to
        # basis gates to include them for the qiskit transpiler
        if "custom_instructions" in config:
            config["basis_gates"] = config["basis_gates"] + config["custom_instructions"]
        return config

    @property
    def max_circuits(self):
        if "max_experiments" in self.configuration():
            return self.configuration()["max_experiments"]
        else:
            return None

    @property
    def target(self):
        if self._from_backend:
            return self._target

        # make target for AerBackend
        required = ["measure", "delay"]

        qiskit_inst_mapping = get_standard_gate_name_mapping()
        qiskit_inst_mapping.update(NAME_MAPPING)

        num_qubits = self.configuration()["n_qubits"]
        in_data = {"num_qubits": num_qubits}
        basis_gates = set.union(set(required), set(self.configuration().get("basis_gates", [])))

        target = Target(**in_data)
        for name in basis_gates:
            if name in required:
                target.add_instruction(
                    instruction=qiskit_inst_mapping[name],
                    properties={(q,): None for q in range(num_qubits)},
                    name=name,
                )
            elif name in qiskit_inst_mapping:
                target.add_instruction(
                    instruction=qiskit_inst_mapping[name],
                    properties=None,
                    name=name,
                )

        if self._coupling_map is not None:
            target._coupling_graph = self._coupling_map.graph.copy()
        return target

    def set_max_qubits(self, max_qubits):
        """Set maximun number of qubits to be used for this backend."""
        if not self._from_backend:
            self._configuration["n_qubits"] = max_qubits
            self._set_configuration_option("n_qubits", max_qubits)

    def clear_options(self):
        """Reset the simulator options to default values."""
        self._options = self._default_options()
        self._options_configuration = {}

    def status(self):
        """Return backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(
            backend_name=self.name,
            backend_version=self.configuration()["backend_version"],
            operational=True,
            pending_jobs=0,
            status_msg="",
        )

    def _execute_qobj_job(self, qobj, job_id="", format_result=True):
        """Run a qobj job"""
        # Start timer
        start = time.time()

        # Take metadata from headers of experiments to work around JSON serialization error
        metadata_list = []
        metadata_index = 0
        for expr in qobj.experiments:
            if hasattr(expr.header, "metadata"):
                metadata_copy = expr.header.metadata.copy()
                metadata_list.append(metadata_copy)
                expr.header.metadata.clear()
                if "id" in metadata_copy:
                    expr.header.metadata["id"] = metadata_copy["id"]
                expr.header.metadata["metadata_index"] = metadata_index
                metadata_index += 1

        # Run simulation
        output = self._execute_qobj(qobj)

        # Recover metadata
        metadata_index = 0
        for expr in qobj.experiments:
            if hasattr(expr.header, "metadata"):
                expr.header.metadata.clear()
                expr.header.metadata.update(metadata_list[metadata_index])
                metadata_index += 1

        # Validate output
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name)
            if output:
                logger.error("Output: %s", output)
            raise AerError("simulation terminated without returning valid output.")

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name
        output["backend_version"] = self.configuration()["backend_version"]

        # Push metadata to experiment headers
        for result in output["results"]:
            if (
                "header" in result
                and "metadata" in result["header"]
                and "metadata_index" in result["header"]["metadata"]
            ):
                metadata_index = result["header"]["metadata"]["metadata_index"]
                result["header"]["metadata"] = metadata_list[metadata_index]

        # Add execution time
        output["time_taken"] = time.time() - start

        # Display warning if simulation failed
        if not output.get("success", False):
            msg = "Simulation failed"
            if "status" in output:
                msg += f" and returned the following error message:\n{output['status']}"
            logger.warning(msg)
        if format_result:
            return self._format_results(output)
        return output

    def _execute_circuits_job(
        self, circuits, parameter_binds, run_options, job_id="", format_result=True
    ):
        """Run a job"""
        # Start timer
        start = time.time()

        # Compile circuits
        circuits, noise_model = self._compile(circuits, **run_options)

        if self._target is not None:
            aer_circuits, idx_maps = assemble_circuits(
                circuits, self.configuration()["basis_gates"]
            )
        else:
            aer_circuits, idx_maps = assemble_circuits(circuits)
        if parameter_binds:
            run_options["parameterizations"] = self._convert_binds(
                circuits, parameter_binds, idx_maps
            )
        elif not all(len(circuit.parameters) == 0 for circuit in circuits):
            raise AerError("circuits have parameters but parameter_binds is not specified.")

        for circ_id, aer_circuit in enumerate(aer_circuits):
            aer_circuit.circ_id = circ_id

        config = generate_aer_config(circuits, self.options, **run_options)

        # Run simulation
        metadata_map = {
            aer_circuit.circ_id: circuit.metadata
            for aer_circuit, circuit in zip(aer_circuits, circuits)
        }
        output = self._execute_circuits(aer_circuits, noise_model, config)

        # Validate output
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name)
            if output:
                logger.error("Output: %s", output)
            raise AerError("simulation terminated without returning valid output.")

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name
        output["backend_version"] = self.configuration()["backend_version"]

        # Push metadata to experiment headers
        for result in output["results"]:
            if "header" not in result:
                continue
            result["header"]["metadata"] = metadata_map[result.pop("circ_id")]

        # Add execution time
        output["time_taken"] = time.time() - start

        # Display warning if simulation failed
        if not output.get("success", False):
            msg = "Simulation failed"
            if "status" in output:
                msg += f" and returned the following error message:\n{output['status']}"
            logger.warning(msg)
        if format_result:
            return self._format_results(output)
        return output

    @staticmethod
    def _format_results(output):
        """Format C++ simulator output for constructing Result"""
        for result in output["results"]:
            data = result.get("data", {})
            metadata = result.get("metadata", {})
            save_types = metadata.get("result_types", {})
            save_subtypes = metadata.get("result_subtypes", {})
            for key, val in data.items():
                if key in save_types:
                    data[key] = format_save_type(val, save_types[key], save_subtypes[key])
        return Result.from_dict(output)

    def _compile(self, circuits, **run_options):
        """Compile circuits and noise model"""
        if isinstance(circuits, (QuantumCircuit, Schedule, ScheduleBlock)):
            circuits = [circuits]
        optypes = [circuit_optypes(circ) for circ in circuits]

        # Compile Qasm3 instructions
        circuits, optypes = compile_circuit(circuits, optypes=optypes)

        # run option noise model
        circuits, noise_model, run_options = self._assemble_noise_model(
            circuits, optypes, **run_options
        )

        return circuits, noise_model

    def _assemble(self, circuits, parameter_binds=None, **run_options):
        """Assemble one or more Qobj for running on the simulator"""

        if isinstance(circuits, (QasmQobj, PulseQobj)):
            qobj = circuits
        else:
            # compile and insert noise injection points
            circuits, noise_model = self._compile(circuits, **run_options)

            # If noise model exists, add it to the run options
            if noise_model:
                run_options["noise_model"] = noise_model

            if parameter_binds:
                # Handle parameter binding
                parameterizations = self._convert_binds(circuits, parameter_binds)
                qobj = None
                for circuit in circuits:
                    assemble_bind = {param: 1 for param in circuit.parameters}
                    qobj_tmp = assemble(
                        [circuit],
                        backend=self,
                        parameter_binds=[assemble_bind],
                        parameterizations=parameterizations,
                    )
                    if qobj:
                        qobj.experiments.append(qobj_tmp.experiments[0])
                    else:
                        qobj = qobj_tmp
            else:
                qobj = assemble(circuits, backend=self)

        # Add options
        for key, val in self.options.__dict__.items():
            if val is not None:
                setattr(qobj.config, key, val)

        # Override with run-time options
        for key, val in run_options.items():
            setattr(qobj.config, key, val)

        return qobj

    def _assemble_noise_model(self, circuits, optypes, **run_options):
        """Move quantum error instructions from circuits to noise model"""
        # Make a shallow copy so we can modify list elements if required
        run_circuits = copy.copy(circuits)

        # Flag for if we need to make a deep copy of the noise model
        # This avoids unnecessarily copying the noise model for circuits
        # that do not contain a quantum error
        updated_noise = False
        noise_model = run_options.get("noise_model", getattr(self.options, "noise_model", None))

        # Add custom pass noise only to QuantumCircuit objects that contain delay
        # instructions since this is the only instruction handled by the noise pass
        # at present
        if noise_model and all(isinstance(circ, QuantumCircuit) for circ in run_circuits):
            npm = noise_model._pass_manager()
            if npm is not None:
                # Get indicies of circuits that need noise transpiling
                transpile_idxs = [idx for idx, optype in enumerate(optypes) if Delay in optype]

                # Transpile only the required circuits
                transpiled_circuits = npm.run([run_circuits[i] for i in transpile_idxs])
                if isinstance(transpiled_circuits, QuantumCircuit):
                    transpiled_circuits = [transpiled_circuits]

                # Update the circuits with transpiled ones
                for idx, circ in zip(transpile_idxs, transpiled_circuits):
                    run_circuits[idx] = circ
                    optypes[idx] = circuit_optypes(circ)

        # Check if circuits contain quantum error instructions
        for idx, circ in enumerate(run_circuits):
            if QuantumChannelInstruction in optypes[idx] and not isinstance(
                circ, (Schedule, ScheduleBlock)
            ):
                updated_circ = False
                new_data = []
                for inst, qargs, cargs in circ.data:
                    if isinstance(inst, QuantumChannelInstruction):
                        updated_circ = True
                        if not updated_noise:
                            # Deep copy noise model on first update
                            if noise_model is None:
                                noise_model = NoiseModel()
                            else:
                                noise_model = copy.deepcopy(noise_model)
                            updated_noise = True
                        # Extract error and replace with place holder
                        qerror = inst._quantum_error
                        qerror_loc = QuantumErrorLocation(qerror)
                        new_data.append((qerror_loc, qargs, cargs))
                        optypes[idx].add(QuantumErrorLocation)
                        # Add error to noise model
                        if qerror.id not in noise_model._default_quantum_errors:
                            noise_model.add_all_qubit_quantum_error(qerror, qerror.id)
                    else:
                        new_data.append((inst, qargs, cargs))
                if updated_circ:
                    new_circ = circ.copy()
                    new_circ.data = new_data
                    run_circuits[idx] = new_circ
                    optypes[idx].discard(QuantumChannelInstruction)

        # Return the possibly updated circuits and noise model
        return run_circuits, noise_model, run_options

    def _get_executor(self, **run_options):
        """Get the executor"""
        if "executor" in run_options:
            return run_options["executor"]
        else:
            return getattr(self._options, "executor", None)

    @abstractmethod
    def _execute_qobj(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj or PulseQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        pass

    @abstractmethod
    def _execute_circuits(self, aer_circuits, noise_model, config):
        """Execute aer circuits on the backend.

        Args:
            aer_circuits (List of AerCircuit): simulator input.
            noise_model (NoiseModel): noise model
            config (Dict): configuration for simulation

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
        if key in self._configuration:
            self._set_configuration_option(key, value)
        else:
            if not hasattr(self._options, key):
                raise AerError(f"Invalid option {key}")
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

    def __repr__(self):
        """String representation of an AerBackend."""
        name = self.__class__.__name__
        display = f"'{self.name}'"
        return f"{name}({display})"
