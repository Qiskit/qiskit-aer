# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os
from pathlib import Path

ROOT_DIR = os.path.dirname(__file__)
with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r") as version_file:
    __version__ = version_file.read().strip()
