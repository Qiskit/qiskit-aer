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

#ifndef _aer_framework_linalg_matrix_utils_smatrix_defs_hpp_
#define _aer_framework_linalg_matrix_utils_smatrix_defs_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/linalg/matrix_utils/matrix_defs.hpp"
#include "framework/types.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace Linalg {

//------------------------------------------------------------------------------
// Static Superoperator Matrices
//------------------------------------------------------------------------------

class SMatrix {
public:
  // Single-qubit gates
  const static cmatrix_t I;    // name: "id"
  const static cmatrix_t X;    // name: "x"
  const static cmatrix_t Y;    // name: "y"
  const static cmatrix_t Z;    // name: "z"
  const static cmatrix_t H;    // name: "h"
  const static cmatrix_t S;    // name: "s"
  const static cmatrix_t SDG;  // name: "sdg"
  const static cmatrix_t T;    // name: "t"
  const static cmatrix_t TDG;  // name: "tdg"
  const static cmatrix_t SX;   // name: "sx"
  const static cmatrix_t SXDG; // name: "sxdg"
  const static cmatrix_t X90;  // name: "x90"

  // Two-qubit gates
  const static cmatrix_t CX;   // name: "cx"
  const static cmatrix_t CY;   // name: "cy"
  const static cmatrix_t CZ;   // name: "cz"
  const static cmatrix_t SWAP; // name: "swap"
  const static cmatrix_t ECR;  // name: "ecr"

  // Identity Matrix
  static cmatrix_t identity(size_t dim);

  // Single-qubit waltz gates
  static cmatrix_t u1(double lam);
  static cmatrix_t u2(double phi, double lam);
  static cmatrix_t u3(double theta, double phi, double lam);
  static cmatrix_t u4(double theta, double phi, double lam, double gamma);

  // Single-qubit rotation gates
  static cmatrix_t r(double phi, double lam);
  static cmatrix_t rx(double theta);
  static cmatrix_t ry(double theta);
  static cmatrix_t rz(double theta);

  // Two-qubit rotation gates
  static cmatrix_t rxx(double theta);
  static cmatrix_t ryy(double theta);
  static cmatrix_t rzz(double theta);
  static cmatrix_t rzx(double theta); // rotation around Tensor(X, Z)

  // Phase Gates
  static cmatrix_t phase(double theta);
  static cmatrix_t phase_diag(double theta);
  static cmatrix_t cphase_diag(double theta);
  static cmatrix_t cphase(double theta);

  // Controlled-single qubit gate
  static cmatrix_t cu(double theta, double phi, double lam, double gamma);

  // Complex arguments are implemented by taking std::real
  // of the input
  static cmatrix_t u1(complex_t lam) { return phase(std::real(lam)); }
  static cmatrix_t u2(complex_t phi, complex_t lam) {
    return u2(std::real(phi), std::real(lam));
  }
  static cmatrix_t u3(complex_t theta, complex_t phi, complex_t lam) {
    return u3(std::real(theta), std::real(phi), std::real(lam));
  };
  static cmatrix_t u4(complex_t theta, complex_t phi, complex_t lam,
                      complex_t gamma) {
    return u4(std::real(theta), std::real(phi), std::real(lam),
              std::real(gamma));
  }
  static cmatrix_t r(complex_t theta, complex_t phi) {
    return r(std::real(theta), std::real(phi));
  }
  static cmatrix_t rx(complex_t theta) { return rx(std::real(theta)); }
  static cmatrix_t ry(complex_t theta) { return ry(std::real(theta)); }
  static cmatrix_t rz(complex_t theta) { return rz(std::real(theta)); }
  static cmatrix_t rxx(complex_t theta) { return rxx(std::real(theta)); }
  static cmatrix_t ryy(complex_t theta) { return ryy(std::real(theta)); }
  static cmatrix_t rzz(complex_t theta) { return rzz(std::real(theta)); }
  static cmatrix_t rzx(complex_t theta) { return rzx(std::real(theta)); }
  static cmatrix_t phase(complex_t theta) { return phase(std::real(theta)); }
  static cmatrix_t phase_diag(complex_t theta) {
    return phase_diag(std::real(theta));
  }
  static cmatrix_t cphase(complex_t theta) { return cphase(std::real(theta)); }
  static cmatrix_t cphase_diag(complex_t theta) {
    return cphase_diag(std::real(theta));
  }
  static cmatrix_t cu(complex_t theta, complex_t phi, complex_t lam,
                      complex_t gamma) {
    return cu(std::real(theta), std::real(phi), std::real(lam),
              std::real(gamma));
  }

  // Return superoperator matrix for reset instruction
  // on specified dim statespace.
  // The returned matrix is (dim * dim, dim * dim).
  static cmatrix_t reset(size_t dim);

  // Return the matrix for a named matrix string
  // Allowed names correspond to all the const static single-qubit
  // and two-qubit gate members
  static const cmatrix_t from_name(const std::string &name) {
    return *label_map_.at(name);
  }

  // Check if the input name string is allowed
  static bool allowed_name(const std::string &name) {
    return (label_map_.find(name) != label_map_.end());
  }

private:
  // Lookup table that returns a pointer to the static data member
  const static stringmap_t<const cmatrix_t *> label_map_;
};

//==============================================================================
// Implementations
//==============================================================================

const cmatrix_t SMatrix::I = Utils::unitary_superop(Matrix::I);

const cmatrix_t SMatrix::X = Utils::unitary_superop(Matrix::X);

const cmatrix_t SMatrix::Y = Utils::unitary_superop(Matrix::Y);

const cmatrix_t SMatrix::Z = Utils::unitary_superop(Matrix::Z);

const cmatrix_t SMatrix::S = Utils::unitary_superop(Matrix::S);

const cmatrix_t SMatrix::SDG = Utils::unitary_superop(Matrix::SDG);

const cmatrix_t SMatrix::T = Utils::unitary_superop(Matrix::T);

const cmatrix_t SMatrix::TDG = Utils::unitary_superop(Matrix::TDG);

const cmatrix_t SMatrix::H = Utils::unitary_superop(Matrix::H);

const cmatrix_t SMatrix::SX = Utils::unitary_superop(Matrix::SX);

const cmatrix_t SMatrix::SXDG = Utils::unitary_superop(Matrix::SXDG);

const cmatrix_t SMatrix::X90 = Utils::unitary_superop(Matrix::X90);

const cmatrix_t SMatrix::CX = Utils::unitary_superop(Matrix::CX);

const cmatrix_t SMatrix::CY = Utils::unitary_superop(Matrix::CY);

const cmatrix_t SMatrix::CZ = Utils::unitary_superop(Matrix::CZ);

const cmatrix_t SMatrix::SWAP = Utils::unitary_superop(Matrix::SWAP);

const cmatrix_t SMatrix::ECR = Utils::unitary_superop(Matrix::ECR);

// Lookup table
const stringmap_t<const cmatrix_t *> SMatrix::label_map_ = {
    {"id", &SMatrix::I},      {"x", &SMatrix::X},       {"y", &SMatrix::Y},
    {"z", &SMatrix::Z},       {"h", &SMatrix::H},       {"s", &SMatrix::S},
    {"sdg", &SMatrix::SDG},   {"t", &SMatrix::T},       {"tdg", &SMatrix::TDG},
    {"x90", &SMatrix::X90},   {"cx", &SMatrix::CX},     {"cy", &SMatrix::CY},
    {"cz", &SMatrix::CZ},     {"swap", &SMatrix::SWAP}, {"sx", &SMatrix::SX},
    {"sxdg", &SMatrix::SXDG}, {"delay", &SMatrix::I},   {"ecr", &SMatrix::ECR}};

cmatrix_t SMatrix::identity(size_t dim) { return Matrix::identity(dim * dim); }

cmatrix_t SMatrix::u1(double lambda) { return phase(lambda); }

cmatrix_t SMatrix::u2(double phi, double lambda) {
  return Utils::tensor_product(Matrix::u2(-phi, -lambda),
                               Matrix::u2(phi, lambda));
}

cmatrix_t SMatrix::u3(double theta, double phi, double lambda) {
  return Utils::tensor_product(Matrix::u3(theta, -phi, -lambda),
                               Matrix::u3(theta, phi, lambda));
}

cmatrix_t SMatrix::u4(double theta, double phi, double lambda, double gamma) {
  return Utils::tensor_product(Matrix::u4(theta, -phi, -lambda, -gamma),
                               Matrix::u4(theta, phi, lambda, gamma));
}

cmatrix_t SMatrix::r(double theta, double phi) {
  return Utils::tensor_product(Matrix::r(-theta, -phi), Matrix::r(theta, phi));
}

cmatrix_t SMatrix::rx(double theta) {
  return Utils::tensor_product(Matrix::rx(-theta), Matrix::rx(theta));
}

cmatrix_t SMatrix::ry(double theta) {
  return Utils::tensor_product(Matrix::ry(theta), Matrix::ry(theta));
}

cmatrix_t SMatrix::rz(double theta) {
  return Utils::tensor_product(Matrix::rz(-theta), Matrix::rz(theta));
}

cmatrix_t SMatrix::rxx(double theta) {
  return Utils::tensor_product(Matrix::rxx(-theta), Matrix::rxx(theta));
}

cmatrix_t SMatrix::ryy(double theta) {
  return Utils::tensor_product(Matrix::ryy(-theta), Matrix::ryy(theta));
}

cmatrix_t SMatrix::rzz(double theta) {
  return Utils::tensor_product(Matrix::rzz(-theta), Matrix::rzz(theta));
}

cmatrix_t SMatrix::rzx(double theta) {
  return Utils::tensor_product(Matrix::rzx(-theta), Matrix::rzx(theta));
}

cmatrix_t SMatrix::cu(double theta, double phi, double lambda, double gamma) {
  return Utils::tensor_product(Matrix::cu(theta, -phi, -lambda, -gamma),
                               Matrix::cu(theta, phi, lambda, gamma));
}

cmatrix_t SMatrix::phase(double theta) {
  cmatrix_t mat(4, 4);
  mat(0, 0) = {1., 0.};
  mat(1, 1) = std::exp(complex_t(0., theta));
  mat(2, 2) = std::exp(complex_t(0., -theta));
  mat(3, 3) = {1., 0.};
  return mat;
}

cmatrix_t SMatrix::phase_diag(double theta) {
  cmatrix_t mat(1, 4);
  mat(0, 0) = {1., 0.};
  mat(0, 1) = std::exp(complex_t(0., theta));
  mat(0, 2) = std::exp(complex_t(0., -theta));
  mat(0, 3) = {1., 0.};
  return mat;
}

cmatrix_t SMatrix::cphase(double theta) {
  const auto exp_p = std::exp(complex_t(0., theta));
  const auto exp_m = std::exp(complex_t(0., -theta));
  cmatrix_t mat(16, 16);
  mat(0, 0) = {1., 0.};
  mat(1, 1) = {1., 0.};
  mat(2, 2) = {1., 0.};
  mat(3, 3) = exp_p;
  mat(4, 4) = {1., 0.};
  mat(5, 5) = {1., 0.};
  mat(6, 6) = {1., 0.};
  mat(7, 7) = exp_p;
  mat(8, 8) = {1., 0.};
  mat(9, 9) = {1., 0.};
  mat(10, 10) = {1., 0.};
  mat(11, 11) = exp_p;
  mat(12, 12) = exp_m;
  mat(13, 13) = exp_m;
  mat(14, 14) = exp_m;
  mat(15, 15) = {1., 0.};
  return mat;
}

cmatrix_t SMatrix::cphase_diag(double theta) {
  const auto exp_p = std::exp(complex_t(0., theta));
  const auto exp_m = std::exp(complex_t(0., -theta));
  cmatrix_t mat(1, 16);
  mat(0, 0) = {1., 0.};
  mat(0, 1) = {1., 0.};
  mat(0, 2) = {1., 0.};
  mat(0, 3) = exp_p;
  mat(0, 4) = {1., 0.};
  mat(0, 5) = {1., 0.};
  mat(0, 6) = {1., 0.};
  mat(0, 7) = exp_p;
  mat(0, 8) = {1., 0.};
  mat(0, 9) = {1., 0.};
  mat(0, 10) = {1., 0.};
  mat(0, 11) = exp_p;
  mat(0, 12) = exp_m;
  mat(0, 13) = exp_m;
  mat(0, 14) = exp_m;
  mat(0, 15) = {1., 0.};
  return mat;
}

cmatrix_t SMatrix::reset(size_t dim) {
  cmatrix_t mat(dim * dim, dim * dim);
  for (size_t j = 0; j < dim; j++) {
    mat(0, j * (dim + 1)) = 1.;
  }
  return mat;
}

//------------------------------------------------------------------------------
} // end namespace Linalg
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
