# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Aer backend configuration
"""
import copy


class GateConfig:
    """Class representing a Gate Configuration

    Attributes:
        name: the gate name as it will be referred to in OpenQASM.
        parameters: variable names for the gate parameters (if any).
        qasm_def: definition of this gate in terms of OpenQASM 2 primitives U
                  and CX.
    """

    def __init__(
        self,
        name,
        parameters,
        qasm_def,
        coupling_map=None,
        latency_map=None,
        conditional=None,
        description=None,
    ):
        """Initialize a GateConfig object

        Args:
            name (str): the gate name as it will be referred to in OpenQASM.
            parameters (list): variable names for the gate parameters (if any)
                               as a list of strings.
            qasm_def (str): definition of this gate in terms of OpenQASM 2 primitives U and CX.
            coupling_map (list): An optional coupling map for the gate. In
                the form of a list of lists of integers representing the qubit
                groupings which are coupled by this gate.
            latency_map (list): An optional map of latency for the gate. In the
                the form of a list of lists of integers of either 0 or 1
                representing an array of dimension
                len(coupling_map) X n_registers that specifies the register
                latency (1: fast, 0: slow) conditional operations on the gate
            conditional (bool): Optionally specify whether this gate supports
                conditional operations (true/false). If this is not specified,
                then the gate inherits the conditional property of the backend.
            description (str): Description of the gate operation
        """

        self.name = name
        self.parameters = parameters
        self.qasm_def = qasm_def
        # coupling_map with length 0 is invalid
        if coupling_map:
            self.coupling_map = coupling_map
        # latency_map with length 0 is invalid
        if latency_map:
            self.latency_map = latency_map
        if conditional is not None:
            self.conditional = conditional
        if description is not None:
            self.description = description

    @classmethod
    def from_dict(cls, data):
        """Create a new GateConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateConfig to create.
                         It will be in the same format as output by
                         :func:`to_dict`.

        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        return cls(**data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = {
            "name": self.name,
            "parameters": self.parameters,
            "qasm_def": self.qasm_def,
        }
        if hasattr(self, "coupling_map"):
            out_dict["coupling_map"] = self.coupling_map
        if hasattr(self, "latency_map"):
            out_dict["latency_map"] = self.latency_map
        if hasattr(self, "conditional"):
            out_dict["conditional"] = self.conditional
        if hasattr(self, "description"):
            out_dict["description"] = self.description
        return out_dict

    def __eq__(self, other):
        if isinstance(other, GateConfig):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        out_str = f"GateConfig({self.name}, {self.parameters}, {self.qasm_def}"
        for i in ["coupling_map", "latency_map", "conditional", "description"]:
            if hasattr(self, i):
                out_str += ", " + repr(getattr(self, i))
        out_str += ")"
        return out_str


class AerBackendConfiguration:
    """Class representing an Aer Backend Configuration.

    Attributes:
        backend_name: backend name.
        backend_version: backend version in the form X.Y.Z.
        n_qubits: number of qubits.
        basis_gates: list of basis gates names on the backend.
        gates: list of basis gates on the backend.
        max_shots: maximum number of shots supported.
    """

    _data = {}

    def __init__(
        self,
        backend_name,
        backend_version,
        n_qubits,
        basis_gates,
        gates,
        max_shots,
        coupling_map,
        supported_instructions=None,
        dynamic_reprate_enabled=False,
        rep_delay_range=None,
        default_rep_delay=None,
        max_experiments=None,
        sample_name=None,
        n_registers=None,
        register_map=None,
        configurable=None,
        credits_required=None,
        online_date=None,
        display_name=None,
        description=None,
        tags=None,
        dt=None,
        dtm=None,
        processor_type=None,
        **kwargs,
    ):
        """Initialize a AerBackendConfiguration Object

        Args:
            backend_name (str): The backend name
            backend_version (str): The backend version in the form X.Y.Z
            n_qubits (int): the number of qubits for the backend
            basis_gates (list): The list of strings for the basis gates of the
                backends
            gates (list): The list of GateConfig objects for the basis gates of
                the backend
            max_shots (int): The maximum number of shots allowed on the backend
            coupling_map (list): The coupling map for the device
            supported_instructions (List[str]): Instructions supported by the backend.
            dynamic_reprate_enabled (bool): whether delay between programs can be set dynamically
                (ie via ``rep_delay``). Defaults to False.
            rep_delay_range (List[float]): 2d list defining supported range of repetition
                delays for backend in μs. First entry is lower end of the range, second entry is
                higher end of the range. Optional, but will be specified when
                ``dynamic_reprate_enabled=True``.
            default_rep_delay (float): Value of ``rep_delay`` if not specified by user and
                ``dynamic_reprate_enabled=True``.
            max_experiments (int): The maximum number of experiments per job
            sample_name (str): Sample name for the backend
            n_registers (int): Number of register slots available for feedback
                (if conditional is True)
            register_map (list): An array of dimension n_qubits X
                n_registers that specifies whether a qubit can store a
                measurement in a certain register slot.
            configurable (bool): True if the backend is configurable, if the
                backend is a simulator
            credits_required (bool): True if backend requires credits to run a
                job.
            online_date (datetime.datetime): The date that the device went online
            display_name (str): Alternate name field for the backend
            description (str): A description for the backend
            tags (list): A list of string tags to describe the backend
            dt (float): Qubit drive channel timestep in nanoseconds.
            dtm (float): Measurement drive channel timestep in nanoseconds.
            processor_type (dict): Processor type for this backend. A dictionary of the
                form ``{"family": <str>, "revision": <str>, segment: <str>}`` such as
                ``{"family": "Canary", "revision": "1.0", segment: "A"}``.

                - family: Processor family of this backend.
                - revision: Revision version of this processor.
                - segment: Segment this processor belongs to within a larger chip.

            **kwargs: optional fields
        """
        self._data = {}

        self.backend_name = backend_name
        self.backend_version = backend_version
        self.n_qubits = n_qubits
        self.basis_gates = basis_gates
        self.gates = gates
        self.local = True
        self.simulator = True
        self.conditional = True
        self.open_pulse = False
        self.memory = True
        self.max_shots = max_shots
        self.coupling_map = coupling_map
        if supported_instructions:
            self.supported_instructions = supported_instructions

        self.dynamic_reprate_enabled = dynamic_reprate_enabled
        if rep_delay_range:
            self.rep_delay_range = [_rd * 1e-6 for _rd in rep_delay_range]  # convert to sec
        if default_rep_delay is not None:
            self.default_rep_delay = default_rep_delay * 1e-6  # convert to sec

        # max_experiments must be >=1
        if max_experiments:
            self.max_experiments = max_experiments
        if sample_name is not None:
            self.sample_name = sample_name
        # n_registers must be >=1
        if n_registers:
            self.n_registers = 1
        # register_map must have at least 1 entry
        if register_map:
            self.register_map = register_map
        if configurable is not None:
            self.configurable = configurable
        if credits_required is not None:
            self.credits_required = credits_required
        if online_date is not None:
            self.online_date = online_date
        if display_name is not None:
            self.display_name = display_name
        if description is not None:
            self.description = description
        if tags is not None:
            self.tags = tags
        # Add pulse properties here because some backends do not
        # fit within the Qasm / Pulse backend partitioning in Qiskit
        if dt is not None:
            self.dt = dt * 1e-9
        if dtm is not None:
            self.dtm = dtm * 1e-9
        if processor_type is not None:
            self.processor_type = processor_type

        # convert lo range from GHz to Hz
        if "qubit_lo_range" in kwargs:
            kwargs["qubit_lo_range"] = [
                [min_range * 1e9, max_range * 1e9]
                for (min_range, max_range) in kwargs["qubit_lo_range"]
            ]

        if "meas_lo_range" in kwargs:
            kwargs["meas_lo_range"] = [
                [min_range * 1e9, max_range * 1e9]
                for (min_range, max_range) in kwargs["meas_lo_range"]
            ]

        # convert rep_times from μs to sec
        if "rep_times" in kwargs:
            kwargs["rep_times"] = [_rt * 1e-6 for _rt in kwargs["rep_times"]]

        self._data.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    @classmethod
    def from_dict(cls, data):
        """Create a new GateConfig object from a dictionary.

        Args:
            data (dict): A dictionary representing the GateConfig to create.
                         It will be in the same format as output by
                         :func:`to_dict`.
        Returns:
            GateConfig: The GateConfig from the input dictionary.
        """
        in_data = copy.copy(data)
        gates = [GateConfig.from_dict(x) for x in in_data.pop("gates")]
        in_data["gates"] = gates
        return cls(**in_data)

    def to_dict(self):
        """Return a dictionary format representation of the GateConfig.

        Returns:
            dict: The dictionary form of the GateConfig.
        """
        out_dict = {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "n_qubits": self.n_qubits,
            "basis_gates": self.basis_gates,
            "gates": [x.to_dict() for x in self.gates],
            "local": self.local,
            "simulator": self.simulator,
            "conditional": self.conditional,
            "memory": self.memory,
            "max_shots": self.max_shots,
            "coupling_map": self.coupling_map,
            "dynamic_reprate_enabled": self.dynamic_reprate_enabled,
        }

        if hasattr(self, "supported_instructions"):
            out_dict["supported_instructions"] = self.supported_instructions

        if hasattr(self, "rep_delay_range"):
            out_dict["rep_delay_range"] = [_rd * 1e6 for _rd in self.rep_delay_range]
        if hasattr(self, "default_rep_delay"):
            out_dict["default_rep_delay"] = self.default_rep_delay * 1e6

        for kwarg in [
            "max_experiments",
            "sample_name",
            "n_registers",
            "register_map",
            "configurable",
            "credits_required",
            "online_date",
            "display_name",
            "description",
            "tags",
            "dt",
            "dtm",
            "processor_type",
        ]:
            if hasattr(self, kwarg):
                out_dict[kwarg] = getattr(self, kwarg)

        out_dict.update(self._data)

        if "dt" in out_dict:
            out_dict["dt"] *= 1e9
        if "dtm" in out_dict:
            out_dict["dtm"] *= 1e9

        # Use GHz in dict
        if "qubit_lo_range" in out_dict:
            out_dict["qubit_lo_range"] = [
                [min_range * 1e-9, max_range * 1e-9]
                for (min_range, max_range) in out_dict["qubit_lo_range"]
            ]

        if "meas_lo_range" in out_dict:
            out_dict["meas_lo_range"] = [
                [min_range * 1e-9, max_range * 1e-9]
                for (min_range, max_range) in out_dict["meas_lo_range"]
            ]

        return out_dict

    @property
    def num_qubits(self):
        """Returns the number of qubits.

        In future, `n_qubits` should be replaced in favor of `num_qubits` for consistent use
        throughout Qiskit. Until this is properly refactored, this property serves as intermediate
        solution.
        """
        return self.n_qubits

    def __eq__(self, other):
        if isinstance(other, AerBackendConfiguration):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __contains__(self, item):
        return item in self.__dict__
