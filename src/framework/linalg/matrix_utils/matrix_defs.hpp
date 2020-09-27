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

#ifndef _aer_framework_linalg_matrix_utils_matrix_defs_hpp_
#define _aer_framework_linalg_matrix_utils_matrix_defs_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/types.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace Linalg {

//------------------------------------------------------------------------------
// Static matrices
//------------------------------------------------------------------------------

class Matrix {
public:
  // Single-qubit gates
  const static cmatrix_t I;   // name: "id"
  const static cmatrix_t X;   // name: "x"
  const static cmatrix_t Y;   // name: "y"
  const static cmatrix_t Z;   // name: "z"
  const static cmatrix_t H;   // name: "h"
  const static cmatrix_t S;   // name: "s"
  const static cmatrix_t SDG; // name: "sdg"
  const static cmatrix_t T;   // name: "t"
  const static cmatrix_t TDG; // name: "tdg"
  const static cmatrix_t SX;  // name: "sx"
  const static cmatrix_t X90; // name: "x90"

  // Two-qubit gates
  const static cmatrix_t CX;   // name: "cx"
  const static cmatrix_t CY;   // name: "cy"
  const static cmatrix_t CZ;   // name: "cz"
  const static cmatrix_t SWAP; // name: "swap"

  // Identity Matrix
  static cmatrix_t identity(size_t dim);

  // Single-qubit waltz gates
  static cmatrix_t u1(double lam);
  static cmatrix_t u2(double phi, double lam);
  static cmatrix_t u3(double theta, double phi, double lam);

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
  static cmatrix_t cphase(double theta);
  static cmatrix_t cphase_diag(double theta);

  // Complex arguments are implemented by taking std::real
  // of the input
  static cmatrix_t u1(complex_t lam) { return phase(std::real(lam)); }
  static cmatrix_t u2(complex_t phi, complex_t lam) {
    return u2(std::real(phi), std::real(lam));
  }
  static cmatrix_t u3(complex_t theta, complex_t phi, complex_t lam) {
    return u3(std::real(theta), std::real(phi), std::real(lam));
  };
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
  static cmatrix_t phase_diag(complex_t theta) { return phase_diag(std::real(theta)); }
  static cmatrix_t cphase(complex_t theta) { return cphase(std::real(theta)); }
  static cmatrix_t cphase_diag(complex_t theta) { return cphase_diag(std::real(theta)); }

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

const cmatrix_t Matrix::I =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}}, {{0, 0}, {1, 0}}});

const cmatrix_t Matrix::X =
    Utils::make_matrix<complex_t>({{{0, 0}, {1, 0}}, {{1, 0}, {0, 0}}});

const cmatrix_t Matrix::Y =
    Utils::make_matrix<complex_t>({{{0, 0}, {0, -1}}, {{0, 1}, {0, 0}}});

const cmatrix_t Matrix::Z =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}}, {{0, 0}, {-1, 0}}});

const cmatrix_t Matrix::S =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}}, {{0, 0}, {0, 1}}});

const cmatrix_t Matrix::SDG =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}}, {{0, 0}, {0, -1}}});
const cmatrix_t Matrix::T = Utils::make_matrix<complex_t>(
    {{{1, 0}, {0, 0}}, {{0, 0}, {1 / std::sqrt(2), 1 / std::sqrt(2)}}});

const cmatrix_t Matrix::TDG = Utils::make_matrix<complex_t>(
    {{{1, 0}, {0, 0}}, {{0, 0}, {1 / std::sqrt(2), -1 / std::sqrt(2)}}});

const cmatrix_t Matrix::H = Utils::make_matrix<complex_t>(
    {{{1 / std::sqrt(2.), 0}, {1 / std::sqrt(2.), 0}},
     {{1 / std::sqrt(2.), 0}, {-1 / std::sqrt(2.), 0}}});

const cmatrix_t Matrix::SX = Utils::make_matrix<complex_t>(
    {{{0.5, 0.5}, {0.5, -0.5}}, {{0.5, -0.5}, {0.5, 0.5}}});

const cmatrix_t Matrix::X90 = Utils::make_matrix<complex_t>(
    {{{1. / std::sqrt(2.), 0}, {0, -1. / std::sqrt(2.)}},
     {{0, -1. / std::sqrt(2.)}, {1. / std::sqrt(2.), 0}}});

const cmatrix_t Matrix::CX =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
                                   {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                   {{0, 0}, {1, 0}, {0, 0}, {0, 0}}});

const cmatrix_t Matrix::CY =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}, {0, -1}},
                                   {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                   {{0, 0}, {0, 1}, {0, 0}, {0, 0}}});
                     
const cmatrix_t Matrix::CZ =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}, {-1, 0}}});

const cmatrix_t Matrix::SWAP =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                   {{0, 0}, {1, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}, {1, 0}}});

// Lookup table
const stringmap_t<const cmatrix_t *> Matrix::label_map_ = {
    {"id", &Matrix::I},     {"x", &Matrix::X},   {"y", &Matrix::Y},
    {"z", &Matrix::Z},      {"h", &Matrix::H},   {"s", &Matrix::S},
    {"sdg", &Matrix::SDG},  {"t", &Matrix::T},   {"tdg", &Matrix::TDG},
    {"x90", &Matrix::X90},  {"cx", &Matrix::CX}, {"cy", &Matrix::CY},
    {"cz", &Matrix::CZ},    {"swap", &Matrix::SWAP}, {"sx", &Matrix::SX},
    {"delay", &Matrix::I}};

cmatrix_t Matrix::identity(size_t dim) {
  cmatrix_t mat(dim, dim);
  for (size_t j = 0; j < dim; j++)
    mat(j, j) = {1.0, 0.0};
  return mat;
}

cmatrix_t Matrix::u1(double lambda) {
  return phase(lambda);
}

cmatrix_t Matrix::u2(double phi, double lambda) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  const complex_t invsqrt2(1. / std::sqrt(2), 0.);
  mat(0, 0) = invsqrt2;
  mat(0, 1) = -std::exp(i * lambda) * invsqrt2;
  mat(1, 0) = std::exp(i * phi) * invsqrt2;
  mat(1, 1) = std::exp(i * (phi + lambda)) * invsqrt2;
  return mat;
}

cmatrix_t Matrix::u3(double theta, double phi, double lambda) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  mat(0, 0) = std::cos(0.5 * theta);
  mat(0, 1) = -std::exp(i * lambda) * std::sin(0.5 * theta);
  mat(1, 0) = std::exp(i * phi) * std::sin(0.5 * theta);
  mat(1, 1) = std::exp(i * (phi + lambda)) * std::cos(0.5 * theta);
  return mat;
}

cmatrix_t Matrix::r(double theta, double phi) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  mat(0, 0) = std::cos(0.5 * theta);
  mat(0, 1) = -i * std::exp(-i * phi) * std::sin(0.5 * theta);
  mat(1, 0) = -i * std::exp(i * phi) * std::sin(0.5 * theta);
  mat(1, 1) = std::cos(0.5 * theta);
  return mat;
}

cmatrix_t Matrix::rx(double theta) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  mat(0, 0) = std::cos(0.5 * theta);
  mat(0, 1) = -i * std::sin(0.5 * theta);
  mat(1, 0) = mat(0, 1);
  mat(1, 1) = mat(0, 0);
  return mat;
}

cmatrix_t Matrix::ry(double theta) {
  cmatrix_t mat(2, 2);
  mat(0, 0) = std::cos(0.5 * theta);
  mat(0, 1) = -1.0 * std::sin(0.5 * theta);
  mat(1, 0) = -mat(0, 1);
  mat(1, 1) = mat(0, 0);
  return mat;
}

cmatrix_t Matrix::rz(double theta) {
  cmatrix_t mat(2, 2);
  const complex_t i(0., 1.);
  mat(0, 0) = std::exp(-i * 0.5 * theta);
  mat(1, 1) = std::exp(i * 0.5 * theta);
  return mat;
}

cmatrix_t Matrix::rxx(double theta) {
  cmatrix_t mat(4, 4);
  const complex_t i(0., 1.);
  const double cost = std::cos(0.5 * theta);
  const double sint = std::sin(0.5 * theta);
  mat(0, 0) = cost;
  mat(0, 3) = -i * sint;
  mat(1, 1) = cost;
  mat(1, 2) = -i * sint;
  mat(2, 1) = -i * sint;
  mat(2, 2) = cost;
  mat(3, 0) = -i * sint;
  mat(3, 3) = cost;
  return mat;
}

cmatrix_t Matrix::ryy(double theta) {
  cmatrix_t mat(4, 4);
  const complex_t i(0., 1.);
  const double cost = std::cos(0.5 * theta);
  const double sint = std::sin(0.5 * theta);
  mat(0, 0) = cost;
  mat(0, 3) = i * sint;
  mat(1, 1) = cost;
  mat(1, 2) = -i * sint;
  mat(2, 1) = -i * sint;
  mat(2, 2) = cost;
  mat(3, 0) = i * sint;
  mat(3, 3) = cost;
  return mat;
}

cmatrix_t Matrix::rzz(double theta) {
  cmatrix_t mat(4, 4);
  const complex_t i(0., 1.);
  const complex_t exp_p = std::exp(i * 0.5 * theta);
  const complex_t exp_m = std::exp(-i * 0.5 * theta);
  mat(0, 0) = exp_m;
  mat(1, 1) = exp_p;
  mat(2, 2) = exp_p;
  mat(3, 3) = exp_m;
  return mat;
}

cmatrix_t Matrix::rzx(double theta) {
  cmatrix_t mat(4, 4);
  const complex_t i(0., 1.);
  const double cost = std::cos(0.5 * theta);
  const double sint = std::sin(0.5 * theta);
  mat(0, 0) = cost;
  mat(0, 2) = -i * sint;
  mat(1, 1) = cost;
  mat(1, 3) = i * sint;
  mat(2, 0) = -i * sint;
  mat(2, 2) = cost;
  mat(3, 1) = i * sint;
  mat(3, 3) = cost;
  return mat;
}

cmatrix_t Matrix::phase(double theta) {
  cmatrix_t mat(2, 2);
  mat(0, 0) = 1;
  mat(1, 1) = std::exp(complex_t(0.0, theta));
  return mat;
}

cmatrix_t Matrix::phase_diag(double theta) {
  cmatrix_t mat(1, 2);
  mat(0, 0) = 1;
  mat(0, 1) = std::exp(complex_t(0.0, theta));
  return mat;
}

cmatrix_t Matrix::cphase(double theta) {
  cmatrix_t mat(4, 4);
  mat(0, 0) = 1;
  mat(1, 1) = 1;
  mat(2, 2) = 1;
  mat(3, 3) = std::exp(complex_t(0.0, theta));
  return mat;
}

cmatrix_t Matrix::cphase_diag(double theta) {
  cmatrix_t mat(1, 4);
  mat(0, 0) = 1;
  mat(0, 1) = 1;
  mat(0, 2) = 1;
  mat(0, 3) = std::exp(complex_t(0.0, theta));
  return mat;
}

//------------------------------------------------------------------------------
} // end namespace Linalg
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif