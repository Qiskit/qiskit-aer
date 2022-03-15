# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from .default_simulator import Benchmark as BaseBanchmark




class UnitaryCPU:

    def simulator(self):
        if self._simulator:
            return self._simulator
        try:
            from qiskit.providers.aer import AerSimulator
            self._simulator = AerSimulator(method='unitary', device='CPU')
        except ImportError:
            from qiskit.providers.aer import UnitarySimulator
            self._simulator = UnitarySimulator(method='unitary')
        return self._simulator


class Benchmark(UnitaryCPU, BaseBanchmark):

    def __init__(self):
        super().__init__([5, 13])

    def add_measure(self, circuit):
        """append measurement"""
        if self.simulator().__class__.__name__ != 'UnitarySimulator':
            circuit.save_state()
