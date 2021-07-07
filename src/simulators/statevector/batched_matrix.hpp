/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020, 2021.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */


#ifndef _qv_batched_matrix_hpp_
#define _qv_batched_matrix_hpp_

#include "framework/operations.hpp"


#include "misc/wrap_thrust.hpp"


namespace AER {
namespace QV {

class batched_matrix_params 
{
public:
  uint_t state_index_;
  uint_t num_qubits_;
  uint_t control_mask_;
  uint_t offset_qubits_;
  uint_t offset_matrix_;
  uint_t qubit_;
  bool super_op_;
  std::complex<double> matrix2x2_[4];

  batched_matrix_params(){}
  __host__ __device__ batched_matrix_params(const batched_matrix_params& prm)
  {
    state_index_ = prm.state_index_;
    num_qubits_ = prm.num_qubits_;
    control_mask_ = prm.control_mask_;
    offset_qubits_ = prm.offset_qubits_;
    offset_matrix_ = prm.offset_matrix_;
    qubit_ = prm.qubit_;
    matrix2x2_[0] = prm.matrix2x2_[0];
    matrix2x2_[1] = prm.matrix2x2_[1];
    matrix2x2_[2] = prm.matrix2x2_[2];
    matrix2x2_[3] = prm.matrix2x2_[3];
  }

  __host__ __device__ batched_matrix_params& operator=(const batched_matrix_params& prm)
  {
    state_index_ = prm.state_index_;
    num_qubits_ = prm.num_qubits_;
    control_mask_ = prm.control_mask_;
    offset_qubits_ = prm.offset_qubits_;
    offset_matrix_ = prm.offset_matrix_;
    qubit_ = prm.qubit_;
    matrix2x2_[0] = prm.matrix2x2_[0];
    matrix2x2_[1] = prm.matrix2x2_[1];
    matrix2x2_[2] = prm.matrix2x2_[2];
    matrix2x2_[3] = prm.matrix2x2_[3];
    return *this;
  }

  void set_control_mask(const reg_t& qubits)
  {
    int_t i;
    control_mask_ = 0;
    for(i=0;i<qubits.size()-num_qubits_;i++){
      control_mask_ |= (1ull << qubits[i]);
    }
  }


  void set_2x2matrix(std::vector<std::complex<double>>& mat)
  {
    matrix2x2_[0] = mat[0];
    matrix2x2_[1] = mat[1];
    matrix2x2_[2] = mat[2];
    matrix2x2_[3] = mat[3];
  }
};



//------------------------------------------------------------------------------
} // end namespace QV
} // namespace AER
//------------------------------------------------------------------------------


#endif	//_qv_batched_matrix_hpp_

