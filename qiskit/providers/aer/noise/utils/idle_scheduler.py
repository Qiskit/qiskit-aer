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


def schedule_idle_gates(circuit, op_times=None, default_op_time=1):
    """
    This function gets a circuit and returns a new one with the idle
    gates in place; each idle gate is labeled "id_<name>" where <name> is the name of
    the slowest gate that runs in parallel

    Args:
        circuit (QuantumCircuit): The circuit to add idle gates to
        op_times (dictionary): dictionary of times taken by specific gate types
        default_op_time (float or int): default time for gates not found in op_times

    Returns:
        QuantumCircuit: The new circuit with the added idle gates

    """
    scheduler = IdleScheduler(op_times, default_op_time)
    return scheduler.schedule(circuit)


class IdleScheduler():
    """Adds idle gates to given circuits"""
    def __init__(self, op_times, default_op_time):
        if op_times is None:
            self.op_times = {}
        else:
            self.op_times = op_times
        self.default_op_time = default_op_time
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
        layers = list(dag.layers())
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
            [(self.op_times.get(node.name, self.default_op_time), node.name)
             for node in layer['graph'].op_nodes()])
        for qubit in self.idle_times.keys():
            self.idle_times[qubit] += max_op_time

        for node in layer['graph'].op_nodes():
            for qubit in node.qargs:
                if node.op.name == 'barrier':  # special case
                    self.idle_times[qubit] = 0
                else:
                    self.idle_times[qubit] -= self.op_times.get(node.name, self.default_op_time)

        id_time = self.op_times.get('id', self.default_op_time)
        for qubit in self.circuit.qubits:
            while self.idle_times[qubit] >= id_time:
                id_gate = IdGate(label="id_{}".format(max_op_name))
                layer['graph'].apply_operation_back(id_gate, [qubit], [])
                self.idle_times[qubit] -= id_time

        return layer['graph']
