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
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.measure import Measure
from qiskit.circuit.library import Barrier
from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit_aer.backends.name_mapping import NAME_MAPPING


class AerBackendRebuildGateSetsFromCircuit(TransformationPass):
    """custom translation class to rebuild basis gates with gates in circuit"""

    def __init__(self, config, opt_lvl):
        super().__init__()
        self.config = config
        if opt_lvl is None:
            self.optimization_level = 1
        else:
            self.optimization_level = opt_lvl
        self.qiskit_inst_name_map = get_standard_gate_name_mapping()
        self.qiskit_inst_name_map["barrier"] = Barrier

    def _add_ops(self, dag, ops: set):
        num_unsupported_ops = 0
        opnodes = dag.op_nodes()
        if opnodes is None:
            return num_unsupported_ops

        for node in opnodes:
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    num_unsupported_ops += self._add_ops(circuit_to_dag(block), ops)
            if node.name in self.qiskit_inst_name_map:
                ops.add(node.name)
            elif node.name in self.config.target:
                ops.add(node.name)
            else:
                num_unsupported_ops = num_unsupported_ops + 1
        return num_unsupported_ops

    def run(self, dag):
        # do nothing for higher optimization level
        if self.optimization_level > 1:
            return dag
        if self.config is None or self.config.target is None:
            return dag

        # search ops in supported name mapping
        ops = set()
        num_unsupported_ops = self._add_ops(dag, ops)

        # if there are some unsupported node (i.e. RealAmplitudes) do nothing
        if num_unsupported_ops > 0 or len(ops) < 1:
            return dag

        # clear all instructions in target
        self.config.target._gate_map.clear()
        self.config.target._gate_name_map.clear()
        self.config.target._qarg_gate_map.clear()
        self.config.target._global_operations.clear()

        # rebuild gate sets from circuit
        for name in ops:
            if name in self.qiskit_inst_name_map:
                self.config.target.add_instruction(self.qiskit_inst_name_map[name], name=name)
            else:
                self.config.target.add_instruction(NAME_MAPPING[name], name=name)
        if "measure" not in ops:
            self.config.target.add_instruction(Measure())
        self.config.basis_gates = list(self.config.target.operation_names)

        return dag


# This plugin should not be used outside of simulator
# TODO : this plugin should be moved to optimization stage plugin
#        if Qiskit will have custom optimizaiton stage plugin interface
#        in that case just return pass without Optimize1qGatesDecomposition
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
