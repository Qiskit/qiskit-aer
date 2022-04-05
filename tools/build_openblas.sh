#!/bin/sh
# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

SOURCE_URL="https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20.tar.gz"

pushd /tmp
curl -L $SOURCE_URL -o openblas.tar.gz
tar xzf openblas.tar.gz
pushd OpenBLAS-0.3.20
PREFIX="/usr/local" DYNAMIC_ARCH=1 USE_OPENMP=1 NUM_THREADS=128 make libs netlib shared
PREFIX="/usr/local" make install
popd
popd
