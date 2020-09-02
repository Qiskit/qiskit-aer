# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Dict

from qiskit.pulse import Schedule, SamplePulse
from signals import BaseSignal, PiecewiseConstant


class PulseConverter:
    """Class to convert Qiskit Schedules to signals"""

    def __init__(self, backend):
        """Create a new converter."""

        self.dt = backend.configuration().dt

    def convert_schedule(self, schedule: Schedule) -> Dict[BaseSignal]:
        """
        Converts a schedule to a list of signals

        #TODO This is still WIP

        Args:
            schedule: Schedule to convert to a dictionary of instructions.
        """
        signals_dict = {}

        for instruction in schedule.instructions:

            if isinstance(instruction[1].command, SamplePulse):
                channel = instruction[1].channels[0].name
                time = instruction[0]
                samples = instruction[1].command.samples

                if channel not in signals_dict:
                    signals_dict[channel] = PiecewiseConstant(self.dt, samples)

        return signals_dict
