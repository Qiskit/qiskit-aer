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

#ifndef _aer_framework_types_hpp_
#define _aer_framework_types_hpp_

#include <cstdint>
#include <complex>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "framework/matrix.hpp" // matrix class
#include "framework/stl_ostream.hpp" 


//=============================================================================
// Numeric Types for backends
//=============================================================================

namespace AER {

  // Numeric Types
  using int_t = int_fast64_t;
  using uint_t = uint_fast64_t; 
  using complex_t = std::complex<double>;
  using cvector_t = std::vector<complex_t>;
  using cmatrix_t = matrix<complex_t>;
  using rvector_t = std::vector<double>;
  using rmatrix_t = matrix<double>;
  using reg_t = std::vector<uint_t>;
  using stringset_t = std::unordered_set<std::string>;
  template <typename T>
  using stringmap_t = std::unordered_map<std::string, T>;
}


//-----------------------------------------------------------------------------
#endif
