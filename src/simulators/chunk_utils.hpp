/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _chunk_utils_hpp
#define _chunk_utils_hpp

#include "framework/opset.hpp"
#include "framework/types.hpp"

namespace AER {

namespace Chunk {

void get_qubits_inout(const int chunk_qubits, const reg_t &qubits,
                      reg_t &qubits_in, reg_t &qubits_out);
void get_inout_ctrl_qubits(const Operations::Op &op, const uint_t num_qubits,
                           reg_t &qubits_in, reg_t &qubits_out);
Operations::Op correct_gate_op_in_chunk(const Operations::Op &op,
                                        reg_t &qubits_in);
void block_diagonal_matrix(const uint_t gid, const uint_t chunk_bits,
                           reg_t &qubits, cvector_t &diag);

void get_qubits_inout(const int chunk_qubits, const reg_t &qubits,
                      reg_t &qubits_in, reg_t &qubits_out) {
  uint_t i;
  qubits_in.clear();
  qubits_out.clear();
  for (i = 0; i < qubits.size(); i++) {
    if (qubits[i] < (uint_t)chunk_qubits) { // in chunk
      qubits_in.push_back(qubits[i]);
    } else {
      qubits_out.push_back(qubits[i]);
    }
  }
}

void get_inout_ctrl_qubits(const Operations::Op &op, const uint_t num_qubits,
                           reg_t &qubits_in, reg_t &qubits_out) {
  if (op.type == Operations::OpType::gate &&
      (op.name[0] == 'c' || op.name.find("mc") == 0)) {
    for (uint_t i = 0; i < op.qubits.size(); i++) {
      if (op.qubits[i] < num_qubits)
        qubits_in.push_back(op.qubits[i]);
      else
        qubits_out.push_back(op.qubits[i]);
    }
  }
}

Operations::Op correct_gate_op_in_chunk(const Operations::Op &op,
                                        reg_t &qubits_in) {
  Operations::Op new_op = op;
  new_op.qubits = qubits_in;
  // change gate name if there is no control qubits inside chunk
  if (op.name.find("swap") != std::string::npos && qubits_in.size() == 2) {
    new_op.name = "swap";
  }
  if (op.name.find("ccx") != std::string::npos) {
    if (qubits_in.size() == 1)
      new_op.name = "x";
    else
      new_op.name = "cx";
  } else if (qubits_in.size() == 1) {
    if (op.name[0] == 'c')
      new_op.name = op.name.substr(1);
    else if (op.name == "mcphase")
      new_op.name = "p";
    else
      new_op.name = op.name.substr(2); // remove "mc"
  }
  return new_op;
}

void block_diagonal_matrix(const uint_t gid, const uint_t chunk_bits,
                           reg_t &qubits, cvector_t &diag) {
  uint_t i;
  uint_t mask_out = 0;
  uint_t mask_id = 0;

  reg_t qubits_in;
  cvector_t diag_in;

  for (i = 0; i < qubits.size(); i++) {
    if (qubits[i] < chunk_bits) { // in chunk
      qubits_in.push_back(qubits[i]);
    } else {
      mask_out |= (1ull << i);
      if ((gid >> (qubits[i] - chunk_bits)) & 1)
        mask_id |= (1ull << i);
    }
  }

  if (qubits_in.size() < qubits.size()) {
    for (i = 0; i < diag.size(); i++) {
      if ((i & mask_out) == mask_id)
        diag_in.push_back(diag[i]);
    }

    if (qubits_in.size() == 0) {
      qubits_in.push_back(0);
      diag_in.resize(2);
      diag_in[1] = diag_in[0];
    }
    qubits = qubits_in;
    diag = diag_in;
  }
}

//-------------------------------------------------------------------------
} // namespace Chunk
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
