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
Aer simulator backend transpiler plug-in
"""
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.circuit.measure import Measure
from qiskit_aer.backends.name_mapping import NAME_MAPPING


class AerBackendRebuildGateSetsFromCircuit(TransformationPass):
    def __init__(self, config, opt_lvl):
        super().__init__()
        self.config = config
        self.optimization_level = opt_lvl

    def run(self, dag):
        # do nothing for higher optimization level
        if self.optimization_level > 1:
            return dag

        # clear all instructions in target
        self.config.target._gate_map.clear()
        self.config.target._gate_name_map.clear()
        self.config.target._qarg_gate_map.clear()
        self.config.target._global_operations.clear()

        # rebuild gate sets from circuit
        opnodes = dag.op_nodes()
        for node in opnodes:
            if node.name not in self.config.target:
                if node.name in NAME_MAPPING:
                    self.config.target.add_instruction(NAME_MAPPING[node.name], name=node.name)
        self.config.target.add_instruction(Measure())

        return dag


class AerBackendPlugin(PassManagerStagePlugin):
    def pass_manager(self, pass_manager_config, optimization_level):
        return PassManager(
            [
                AerBackendRebuildGateSetsFromCircuit(
                    config=pass_manager_config, opt_lvl=optimization_level
                )
            ]
        )
