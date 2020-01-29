/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _TEST_TYPES_HPP
#define _TEST_TYPES_HPP

#include <complex>

using complex_t = std::complex<double>;

struct TermExpression {
    TermExpression(const std::string& term) : term(term) {}
    std::string term;
};

#endif // _TEST_TYPES_HPP