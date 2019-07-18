# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Idle gates scheduler module

The goal of this module is to add idle gates to a circuit instead of
having blank spaces with no operations at times where no gate can be applied
to the qubit (due to qubit dependenices or barriers)
"""
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.extensions.standard.iden import IdGate


def schedule_idle_gates(circuit, op_times=None, default_op_time=1, labels=None):
    """
    This function gets a circuit and returns a new one with the idle
    gates in place; each idle gate is labeled "id_<name>" where <name> is the name of
    the slowest gate that runs in parallel

    Args:
        circuit (QuantumCircuit): The circuit to add idle gates to
        op_times (dictionary): dictionary of times taken by specific gate types
        default_op_time (float or int): default time for gates not found in op_times
        labels (string or dictionary): the label to give the id gate, either a uniform
                label for all the id gates (string) or a dictionary mapping from the
                gate type that caused the delay to the expected id gate label

    Returns:
        QuantumCircuit: The new circuit with the added idle gates

    """
    scheduler = IdleScheduler(op_times, default_op_time, labels)
    return scheduler.schedule(circuit)


class IdleScheduler():
    """Adds idle gates to given circuits"""
    def __init__(self, op_times, default_op_time, id_gate_labels=None):
        self.set_op_times(op_times)
        self.default_op_time = default_op_time
        self.id_gate_labels = id_gate_labels
        self.circuit = None
        self.idle_times = None

    def schedule(self, circuit):
        """Adds the idle gates to the circuit
            Args:
               circuit (QuantumCircuit): The circuit to modify
            Returns:
               QuantumCircuit: The modified circuit
       """
        self.circuit = circuit
        new_dag = DAGCircuit()
        self.idle_times = {qubit: 0 for qubit in self.circuit.qubits}
        dag = circuit_to_dag(circuit)
        # convert to circuit and back as hack to prevent nondeterminism in DAGCircuit
        layers = [circuit_to_dag(dag_to_circuit(l['graph'])) for l in dag.layers()]
        for layer in layers:
            new_layer_graph = self.add_identities_to_layer(layer)
            new_dag.extend_back(new_layer_graph)
        return dag_to_circuit(new_dag)

    def add_identities_to_layer(self, layer):
        """Adds the idle gates to a specific layer in the circuit
            Args:
               layer (dict): The layer to modify (contains a circuit graph)
            Returns:
               dict: The modified layer
        """
        max_op_time, max_op_name = max(
            [(self.get_op_time((node.name, node.qargs)), node.name) for node in layer.op_nodes()]
        )
        for qubit in self.idle_times.keys():
            self.idle_times[qubit] += max_op_time

        for node in layer.op_nodes():
            for qubit in node.qargs:
                if node.op.name in ['barrier', 'snapshot']:  # special cases
                    self.idle_times[qubit] = 0
                else:
                    self.idle_times[qubit] -= self.op_times.get(node.name, self.default_op_time)

        id_time = self.op_times.get('id', self.default_op_time)
        for qubit in self.circuit.qubits:
            while self.idle_times[qubit] >= id_time:
                id_gate = IdGate(label=self.id_gate_label(max_op_name))
                layer.apply_operation_back(id_gate, [qubit], [])
                self.idle_times[qubit] -= id_time

        return layer

    def id_gate_label(self, op_name):
        """Given an op name, returns the label for id gates created by that op
            Args:
               op_name (string): The name of the op causing the creation of id gates
            Returns:
               string: the expected label, based on self.id_gate_labels
        """
        label = "id_{}".format(op_name)
        if self.id_gate_labels is not None:
            if isinstance(self.id_gate_labels, str):
                label = self.id_gate_labels
            if isinstance(self.id_gate_labels, dict) and op_name in self.id_gate_labels:
                label = self.id_gate_labels[op_name]
        return label

    def set_op_times(self, op_times):
        """Sets the op_times data field according to input
            Args:
               op_times (None or dict or list): The op times specification
                None results in using self.default_op_times for all gates
       """
        self.op_times = {}
        if op_times is None:
            return

        if isinstance(op_times, dict):
            self.op_times = op_times

        if isinstance(op_times, list):
            for op_time in op_times:
                if len(op_time) == 2: # (name, time)
                    self.op_times[op_time[0]] = op_time[1]
                if len(op_time) == 3: # (name, qubits, time)
                    if op_time[1] is None:
                        self.op_times[op_time[0]] = op_time[2]
                    else:
                        self.op_times[(op_time[0],tuple(op_time[1]))] = op_time[2]

    def get_op_time(self, op_data):
        """Gets the op time for the specified operator
            Args:
               op_data (String or tuple): The op data specification
               Can either be a name string (e.g. 'x') or a pair of name and qubits list
            Returns:
               float: the time for the specified op
        """

        if len(op_data) == 2:
            op_data = (op_data[0], tuple(op_data[1]))

        if op_data in self.op_times:
            return self.op_times[op_data]

        if len(op_data) == 2 and op_data[0] in self.op_times:
            return self.op_times[op_data[0]]

        return self.default_op_time
