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
  const static cmatrix_t X90; // name: "x90"

  // Two-qubit gates
  const static cmatrix_t CX;   // name: "cx"
  const static cmatrix_t CZ;   // name: "cz"
  const static cmatrix_t SWAP; // name: "swap"

  // Identity Matrix
  static cmatrix_t identity(size_t dim);

  // Single-qubit waltz gates
  static cmatrix_t u1(double lam);
  static cmatrix_t u2(double phi, double lam);
  static cmatrix_t u3(double theta, double phi, double lam);

  // Complex arguments are implemented by taking std::real
  // of the input
  static cmatrix_t u1(complex_t lam) { return u1(std::real(lam)); }
  static cmatrix_t u2(complex_t phi, complex_t lam) {
    return u2(std::real(phi), std::real(lam));
  }
  static cmatrix_t u3(complex_t theta, complex_t phi, complex_t lam) {
    return u3(std::real(theta), std::real(phi), std::real(lam));
  };

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

const cmatrix_t Matrix::X90 = Utils::make_matrix<complex_t>(
    {{{1. / std::sqrt(2.), 0}, {0, -1. / std::sqrt(2.)}},
     {{0, -1. / std::sqrt(2.)}, {1. / std::sqrt(2.), 0}}});

const cmatrix_t Matrix::CX =
    Utils::make_matrix<complex_t>({{{1, 0}, {0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}, {1, 0}},
                                   {{0, 0}, {0, 0}, {1, 0}, {0, 0}},
                                   {{0, 0}, {1, 0}, {0, 0}, {0, 0}}});

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
    {"x90", &Matrix::X90},  {"cx", &Matrix::CX}, {"cz", &Matrix::CZ},
    {"swap", &Matrix::SWAP}};

cmatrix_t Matrix::identity(size_t dim) {
  cmatrix_t mat(dim, dim);
  for (size_t j = 0; j < dim; j++)
    mat(j, j) = {1.0, 0.0};
  return mat;
}

cmatrix_t Matrix::u1(double lambda) {
  cmatrix_t mat(2, 2);
  mat(0, 0) = {1., 0.};
  mat(1, 1) = std::exp(complex_t(0., lambda));
  return mat;
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

//------------------------------------------------------------------------------
} // end namespace Linalg
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif