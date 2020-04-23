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

/* This is the implementation of QubitVectorAVX2 class.
 * This file needs to be compiled separately because because it will be compiled
 * to AVX2 machine code instructions. As AVX2 capabiltiies are detected at
 * runtime, only machines with AVX2 support will/could run this code */

#include "qubitvector_avx2.hpp"
#include "qubitvector.hpp"

using namespace QV;

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------
template <typename data_t>
QubitVectorAvx2<data_t>::QubitVectorAvx2(size_t num_qubits)
  {
      Base::num_qubits_ = 0;
      Base::data_ = nullptr;
      Base::checkpoint_ = 0;
      Base::set_num_qubits(num_qubits);
}

template <typename data_t>
QubitVectorAvx2<data_t>::QubitVectorAvx2() : QubitVectorAvx2(0) {}

template <typename data_t>
void QubitVectorAvx2<data_t>::apply_matrix(const uint_t qubit,
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
void QubitVectorAvx2<data_t>::apply_matrix(const reg_t &qubits,
                                          const cvector_t<double> &mat) {
  apply_matrix_avx<data_t>(Base::data_, Base::data_size_, qubits,
    (void*) Base::convert(mat).data(), _calculate_num_threads());
}

template <typename data_t>
uint_t QubitVectorAvx2<data_t>::_calculate_num_threads(){
  if(Base::num_qubits_ > Base::omp_threshold_ &&  Base::omp_threads_ > 1){
       return Base::omp_threads_;
  }
  return 1;
}

template class QV::QubitVectorAvx2<double>;
template class QV::QubitVectorAvx2<float>;



