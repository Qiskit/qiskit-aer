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



#ifndef _qv_qubit_vector_avx2_hpp_
#define _qv_qubit_vector_avx2_hpp_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "framework/json.hpp"
#include "qvintrin_avx.hpp"
#include "qubitvector.hpp"


namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;

//============================================================================
// QubitVectorAvx class
//============================================================================

template <typename data_t = double>
class QubitVectorAvx : public QubitVector<data_t, QubitVectorAvx> {

  // We need this to access the base class members
  using Base = QubitVector<data_t, QubitVectorAvx>;

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVectorAvx();
  explicit QubitVectorAvx(size_t num_qubits);
  virtual ~QubitVectorAvx();
  QubitVectorAvx(const QubitVectorAvx& obj) = delete;
  QubitVectorAvx &operator=(const QubitVectorAvx& obj) = delete;

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t<double> &mat);

  protected:

  uint_t _calculate_num_threads();
};

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------
template <typename data_t>
QubitVectorAvx<data_t>::QubitVectorAvx(size_t num_qubits) :
  Base::num_qubits_(0),
  Base::data_(nullptr),
  Base::checkpoint_(0)
  {
    Base::set_num_qubits(num_qubits);
}

template <typename data_t>
QubitVectorAvx<data_t>::QubitVectorAvx() : QubitVectorAvx(0) {}

template <typename data_t>
void QubitVectorAvx<data_t>::apply_matrix(const uint_t qubit,
                                          const cvector_t<double>& mat) {

  // Check if matrix is diagonal and if so use optimized lambda OR
  // Check if anti-diagonal matrix and if so use optimized lambda
  if((mat[1] == 0.0 && mat[2] == 0.0) || (mat[0] == 0.0 && mat[3] == 0.0))
  {
      // These cases are treated in the Base class
      Base::apply_matrix(qubit, mat);
      return;
  }
  // Convert qubit to array register for lambda functions
  areg_t<1> qubits = {{qubit}};
  apply_matrix_avx<data_t>(Base::data_, Base::data_size_, qubits,
    (void*) Base::convert(mat).data(), _calculate_num_threads());
}

template <typename data_t>
void QubitVectorAvx<data_t>::apply_matrix(const reg_t &qubits,
                                          const cvector_t<double> &mat) {
  apply_matrix_avx<data_t>(Base::data_, Base::data_size_, qubits,
    (void*) Base::convert(mat).data(), _calculate_num_threads());
}

template <typename data_t>
uint_t QubitVectorAvx<data_t>::_calculate_num_threads(){
  if(Base::num_qubits_ > Base::omp_threshold_ &&  Base::omp_threads_ > 1){
       return omp_threads_;
  }
  return 1;
}
//------------------------------------------------------------------------------
#endif // end module
