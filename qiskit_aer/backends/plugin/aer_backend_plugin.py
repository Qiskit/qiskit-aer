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
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.measure import Measure
from qiskit_aer.backends.name_mapping import NAME_MAPPING


class AerBackendRebuildGateSetsFromCircuit(TransformationPass):
    """custom translation class to rebuild basis gates with gates in circuit"""

    def __init__(self, config, opt_lvl):
        super().__init__()
        self.config = config
        self.optimization_level = opt_lvl

    def run(self, dag):
        # do nothing for higher optimization level
        if self.optimization_level > 1:
            return dag

        # search ops in supported name mapping
        ops = []
        num_unsupported_ops = 0
        opnodes = dag.op_nodes()
        if opnodes is None:
            return dag
        for node in opnodes:
            if node.name in self.config.target:
                if node.name not in ops:
                    ops.append(node.name)
            else:
                num_unsupported_ops = num_unsupported_ops + 1

        # if there are some unsupported node (i.e. RealAmplitudes) do nothing
        if num_unsupported_ops > 0:
            return dag
        if len(ops) < 1:
            return dag

        # clear all instructions in target
        self.config.target._gate_map.clear()
        self.config.target._gate_name_map.clear()
        self.config.target._qarg_gate_map.clear()
        self.config.target._global_operations.clear()

        # rebuild gate sets from circuit
        for name in ops:
            if name not in self.config.target:
                if name != "measure":
                    self.config.target.add_instruction(NAME_MAPPING[name], name=name)
        self.config.target.add_instruction(Measure())

        return dag


class AerBackendPlugin(PassManagerStagePlugin):
    """custom passmanager to avoid unnecessary gate changes"""

    def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
        return PassManager(
            [
                UnitarySynthesis(
                    pass_manager_config.basis_gates,
                    approximation_degree=pass_manager_config.approximation_degree,
                    coupling_map=pass_manager_config.coupling_map,
                    backend_props=pass_manager_config.backend_properties,
                    plugin_config=pass_manager_config.unitary_synthesis_plugin_config,
                    method=pass_manager_config.unitary_synthesis_method,
                    target=pass_manager_config.target,
                ),
                HighLevelSynthesis(
                    hls_config=pass_manager_config.hls_config,
                    coupling_map=pass_manager_config.coupling_map,
                    target=pass_manager_config.target,
                    use_qubit_indices=True,
                    equivalence_library=sel,
                    basis_gates=pass_manager_config.basis_gates,
                ),
                BasisTranslator(sel, pass_manager_config.basis_gates, pass_manager_config.target),
                AerBackendRebuildGateSetsFromCircuit(
                    config=pass_manager_config, opt_lvl=optimization_level
                ),
            ]
        )
