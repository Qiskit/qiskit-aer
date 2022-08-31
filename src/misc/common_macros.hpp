/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef QASM_SIMULATOR_COMMON_MACROS_HPP
#define QASM_SIMULATOR_COMMON_MACROS_HPP

#if defined(__GNUC__) && defined(__x86_64__) && defined(__AVX2__)
 #define GNUC_AVX2
#endif

#endif //QASM_SIMULATOR_COMMON_MACROS_HPP
