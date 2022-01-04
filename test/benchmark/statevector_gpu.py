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
from .default_simulator import ExpVal as BaseExpVal
from .default_simulator import Noise as BaseNoise


class StatevectorGPU:

    def simulator(self):
        if self._simulator:
            return self._simulator
        try:
            from qiskit.providers.aer import AerSimulator
            self._simulator = AerSimulator(method='statevector', device='GPU')
        except:
            from qiskit.providers.aer import QasmSimulator
            self._simulator = QasmSimulator(method='statevector_gpu')
        return self._simulator


class Benchmark(StatevectorGPU, BaseBanchmark):

    def __init__(self):
        super().__init__([5, 15, 25])


class ExpVal(StatevectorGPU, BaseExpVal):

    def __init__(self):
        super().__init__([5, 15, 25])


class Noise(StatevectorGPU, BaseNoise):

    def __init__(self):
        super().__init__([5, 15])
