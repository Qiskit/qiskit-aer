/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

/*
This transpiler converts parameterized gates into matrix operations.
This transpiler is called after gate fusion, because the parameterized gates
may be fused and transpiled to matrix operations in gate fusion.

This transplier stores matrices in Operations::Op.params array in cvector_t
format not in Operations::Op.mats for effective data transfer to GPU memory
Also matrices in Operations::OpType::matrix will be stored in Op.params as well

GPU simulator supports matrix multiplication with control qubits
but CPU does not. So there is option to convert to matrix including
control qubits for CPU.
*/

#ifndef _aer_parameter2matrix_hpp_
#define _aer_parameter2matrix_hpp_

#include "framework/config.hpp"
#include "framework/utils.hpp"
#include "transpile/circuitopt.hpp"

namespace AER {
namespace Transpile {

enum class ParamGates {
  rxx,
  ryy,
  rzz,
  rzx,
  mcr,
  mcrx,
  mcry,
  mcrz,
  mcp,
  mcu2,
  mcu3,
  mcu,
};

class Parameter2Matrix : public CircuitOptimization {
public:
  Parameter2Matrix() {}
  ~Parameter2Matrix() {}

  void optimize_circuit(Circuit &circ, Noise::NoiseModel &noise,
                        const opset_t &allowed_opset,
                        ExperimentResult &result) const override;

  void set_config(const Config &config) override;

  void include_control_qubits(bool flg) {
    include_control_qubits_in_matrix_ = flg;
  }

protected:
  void gate_to_matrix(Operations::Op &op, uint_t num_params) const;

  bool include_control_qubits_in_matrix_ = false;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<ParamGates> gateset_;
};

const stringmap_t<ParamGates> Parameter2Matrix::gateset_(
    {{"p", ParamGates::mcp},       {"r", ParamGates::mcr},
     {"rx", ParamGates::mcrx},     {"ry", ParamGates::mcry},
     {"rz", ParamGates::mcrz},     {"u1", ParamGates::mcp},
     {"u2", ParamGates::mcu2},     {"u3", ParamGates::mcu3},
     {"u", ParamGates::mcu3},      {"U", ParamGates::mcu3},
     {"cp", ParamGates::mcp},      {"cu1", ParamGates::mcp},
     {"cu2", ParamGates::mcu2},    {"cu3", ParamGates::mcu3},
     {"cu", ParamGates::mcu},      {"cp", ParamGates::mcp},
     {"rxx", ParamGates::rxx},     {"ryy", ParamGates::ryy},
     {"rzz", ParamGates::rzz},     {"rzx", ParamGates::rzx},
     {"mcr", ParamGates::mcr},     {"mcrx", ParamGates::mcrx},
     {"mcry", ParamGates::mcry},   {"mcrz", ParamGates::mcrz},
     {"mcphase", ParamGates::mcp}, {"mcp", ParamGates::mcp},
     {"mcu1", ParamGates::mcp},    {"mcu2", ParamGates::mcu2},
     {"mcu3", ParamGates::mcu3},   {"mcu", ParamGates::mcu}});

void Parameter2Matrix::set_config(const Config &config) {
  CircuitOptimization::set_config(config);
}

void Parameter2Matrix::optimize_circuit(Circuit &circ, Noise::NoiseModel &noise,
                                        const opset_t &allowed_opset,
                                        ExperimentResult &result) const {
  if (circ.num_bind_params <= 1)
    return;

  for (int_t i = 0; i < circ.ops.size(); i++) {
    if (circ.ops[i].has_bind_params) {
      if (circ.ops[i].type == Operations::OpType::gate) {
        gate_to_matrix(circ.ops[i], circ.num_bind_params);
      } else if (circ.ops[i].type == Operations::OpType::matrix) {
        // convert matrix to cvector_t in params
        uint_t matrix_size = circ.ops[i].mats[0].size();
        circ.ops[i].params.resize(matrix_size * circ.num_bind_params);
        for (int_t j = 0; j < circ.num_bind_params; j++) {
          for (int_t k = 0; k < matrix_size; k++)
            circ.ops[i].params[j * matrix_size + k] = circ.ops[i].mats[j][k];
        }
        circ.ops[i].mats.clear();
      }
    }
  }

  circ.set_params();
}

void Parameter2Matrix::gate_to_matrix(Operations::Op &op,
                                      uint_t num_params) const {
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    return;

  uint_t matrix_size;
  if (it->second == ParamGates::mcrz || it->second == ParamGates::rzz ||
      it->second == ParamGates::mcp) {
    matrix_size = 2ull;
    op.type = Operations::OpType::diagonal_matrix;
  } else {
    matrix_size = 4ull;
    op.type = Operations::OpType::matrix;
  }
  cvector_t matrix_array(num_params * matrix_size);

  auto store_matrix = [&matrix_array, matrix_size](int_t iparam,
                                                   cvector_t mat) {
    for (int_t j = 0; j < matrix_size; j++)
      matrix_array[iparam * matrix_size + j] = mat[j];
  };

  switch (it->second) {
  case ParamGates::mcr:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i,
                   Linalg::VMatrix::r(op.params[i * 2], op.params[i * 2 + 1]));
    break;
  case ParamGates::mcrx:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::rx(std::real(op.params[i])));
    break;
  case ParamGates::mcry:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::ry(std::real(op.params[i])));
    break;
  case ParamGates::mcrz:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::rz_diag(std::real(op.params[i])));
    break;
  case ParamGates::rxx:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::rxx(std::real(op.params[i])));
    break;
  case ParamGates::ryy:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::ryy(std::real(op.params[i])));
    break;
  case ParamGates::rzz:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::rzz_diag(std::real(op.params[i])));
    break;
  case ParamGates::rzx:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::rzx(std::real(op.params[i])));
    break;
  case ParamGates::mcu3:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::u3(std::real(op.params[i * 3]),
                                          std::real(op.params[i * 3 + 1]),
                                          std::real(op.params[i * 3 + 2])));
    break;
  case ParamGates::mcu:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::u4(std::real(op.params[i * 4]),
                                          std::real(op.params[i * 4 + 1]),
                                          std::real(op.params[i * 4 + 2]),
                                          std::real(op.params[i * 4 + 3])));
    break;
  case ParamGates::mcu2:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::u2(std::real(op.params[i * 2]),
                                          std::real(op.params[i * 2 + 1])));
    break;
  case ParamGates::mcp:
    for (int_t i = 0; i < num_params; i++)
      store_matrix(i, Linalg::VMatrix::phase_diag(std::real(op.params[i])));
    break;
  default:
    break;
  }

  op.params = matrix_array;
}

//-------------------------------------------------------------------------
} // end namespace Transpile
} // end namespace AER
//-------------------------------------------------------------------------
#endif
