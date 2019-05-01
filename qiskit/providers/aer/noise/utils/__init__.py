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
Noise utils for Qiskit Aer.
"""

from .noise_remapper import remap_noise_model
from .noise_transformation import NoiseTransformer
from .noise_transformation import approximate_quantum_error
from .noise_transformation import approximate_noise_model
