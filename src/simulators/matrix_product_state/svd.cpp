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

// Modified: checked LP64 workspaces and exception-safe nonconvergence recovery.

/*
 * Adapted from: P. A. Businger and G. H. Golub, Comm. ACM 12, 564 (1969)
 */

#include "svd.hpp"
#include "framework/linalg/almost_equal.hpp"
#include "framework/utils.hpp"
#include <cassert>
#include <climits>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>

namespace AER {
// default values
constexpr auto mul_factor = 1e2;
constexpr long double tiny_factor = 1e30;
constexpr auto zero_threshold = 1e-50; // threshold for comparing FP values
constexpr auto THRESHOLD = 1e-9; // threshold for cutting values in reduce_zeros
constexpr auto NUM_SVD_TRIES = 15;

cmatrix_t diag(rvector_t S, uint_t m, uint_t n);

cmatrix_t diag(rvector_t S, uint_t m, uint_t n) {
  cmatrix_t Res = cmatrix_t(m, n);
  for (uint_t i = 0; i < m; i++) {
    for (uint_t j = 0; j < n; j++) {
      Res(i, j) = (i == j ? complex_t(S[i]) : 0);
    }
  }
  return Res;
}

cmatrix_t reshape_before_SVD(std::vector<cmatrix_t> data) {
  //	Turns 4 matrices A0,A1,A2,A3 to big matrix:
  //	A0 A1
  //	A2 A3
  cmatrix_t temp1 = AER::Utils::concatenate(data[0], data[1], 1),
            temp2 = AER::Utils::concatenate(data[2], data[3], 1);
  return AER::Utils::concatenate(temp1, temp2, 0);
}
std::vector<cmatrix_t> reshape_U_after_SVD(const cmatrix_t &U) {
  std::vector<cmatrix_t> Res(2);
  AER::Utils::split(U, Res[0], Res[1], 0);
  return Res;
}
std::vector<cmatrix_t> reshape_V_after_SVD(const cmatrix_t &V) {
  std::vector<cmatrix_t> Res(2);
  AER::Utils::split(AER::Utils::dagger(V), Res[0], Res[1], 1);
  return Res;
}
std::vector<cmatrix_t> reshape_VH_after_SVD(const cmatrix_t &V) {
  std::vector<cmatrix_t> Res(2);
  AER::Utils::split(V, Res[0], Res[1], 1);
  return Res;
}

//-------------------------------------------------------------
// function name: num_of_SV
// Description: Computes the number of none-zero singular values
//				in S
// Parameters: rvector_t S - vector of singular values from the
//			   SVD decomposition
// Returns: number of elements in S whose norm is greater than 0
//			(actually greater than threshold)
//-------------------------------------------------------------
uint_t num_of_SV(rvector_t S, double threshold) {
  uint_t sum = 0;
  for (uint_t i = 0; i < S.size(); ++i) {
    if (std::norm(S[i]) > threshold)
      sum++;
  }
  return sum;
}

double reduce_zeros(cmatrix_t &U, rvector_t &S, cmatrix_t &V,
                    uint_t max_bond_dimension, double truncation_threshold,
                    bool mps_lapack) {
  uint_t SV_num = num_of_SV(S, CHOP_THRESHOLD);
  uint_t new_SV_num = SV_num;

  if (max_bond_dimension < SV_num) {
    // in this case, leave only the first max_bond_dimension
    // values in S, and discard all the rest
    new_SV_num = max_bond_dimension;
  }
  // Remove the lowest Schmidt coefficients such that the sum of
  // their squares is less than trunction_threshold
  double sum_squares = 0;
  for (int_t i = new_SV_num - 1; i >= 0; i--) {
    if (sum_squares + std::norm(S[i]) < truncation_threshold && i > 0) {
      sum_squares += std::norm(S[i]);
    } else {
      new_SV_num = i + 1;
      break;
    }
  }

  // discarded_value is the sum of the squares of the Schmidt coeffients
  // that were discarded by approximation
  double discarded_value = 0.0;

  if (new_SV_num < SV_num) {
    for (uint_t i = new_SV_num; i < SV_num; i++) {
      discarded_value += std::norm(S[i]);
    }
  }

  U.resize(U.GetRows(), new_SV_num);
  S.resize(new_SV_num);
  // When using LAPACK function, V is V dagger
  if (mps_lapack) {
    V.resize(new_SV_num, V.GetColumns());
  } else {
    V.resize(V.GetRows(), new_SV_num);
  }

  // Check if we need to re-normalize the values of S
  double new_sum_squares = 0;
  for (uint_t i = 0; i < S.size(); i++)
    new_sum_squares += std::norm(S[i]);
  if (!Linalg::almost_equal(1.0 - new_sum_squares, 0., THRESHOLD)) {
    double sqrt_sum = std::sqrt(new_sum_squares);
    for (uint_t i = 0; i < S.size(); i++)
      S[i] /= sqrt_sum;
  }
  return discarded_value;
}

void validate_SVdD_result(const cmatrix_t &A, const cmatrix_t &U,
                          const rvector_t &S, const cmatrix_t &V) {
  const uint_t nrows = A.GetRows(), ncols = A.GetColumns();

  cmatrix_t diag_S = diag(S, nrows, ncols);
  cmatrix_t product = U * diag_S;
  product = product * V;

  for (uint_t ii = 0; ii < nrows; ii++)
    for (uint_t jj = 0; jj < ncols; jj++)
      if (!Linalg::almost_equal(std::abs(A(ii, jj)), std::abs(product(ii, jj)),
                                THRESHOLD)) {
        std::cout << std::abs(A(ii, jj)) << " vs " << std::abs(product(ii, jj))
                  << std::endl;
        throw std::runtime_error("Error: Wrong SVD calculations: A != USV*");
      }
}

void validate_SVD_result(const cmatrix_t &A, const cmatrix_t &U,
                         const rvector_t &S, const cmatrix_t &V) {
  const uint_t nrows = A.GetRows(), ncols = A.GetColumns();

  cmatrix_t diag_S = diag(S, nrows, ncols);
  cmatrix_t product = U * diag_S;
  product = product * AER::Utils::dagger(V);
  for (uint_t ii = 0; ii < nrows; ii++)
    for (uint_t jj = 0; jj < ncols; jj++)
      if (!Linalg::almost_equal(std::abs(A(ii, jj)), std::abs(product(ii, jj)),
                                THRESHOLD)) {
        throw std::runtime_error("Error: Wrong SVD calculations: A != USV*");
      }
}

// added cut-off at the end
status csvd(cmatrix_t &A, cmatrix_t &U, rvector_t &S, cmatrix_t &V) {
  int m = A.GetRows(), n = A.GetColumns(), size = std::max(m, n);
  rvector_t b(size, 0.0), c(size, 0.0), t(size, 0.0);
  double cs = 0.0, eps = 0.0, f = 0.0, g = 0.0, h = 0.0, sn = 0.0, w = 0.0,
         x = 0.0, y = 0.0, z = 0.0;
  double eta = 1e-10, tol = 1.5e-34;
  // using int and not uint_t because uint_t caused bugs in loops with condition
  // of >= 0
  int i = 0, j = 0, k = 0, k1 = 0, l = 0, l1 = 0;
  complex_t q = 0;
  // Transpose when m < n
  bool transposed = false;
  if (m < n) {
    transposed = true;
    A = AER::Utils::dagger(A);
    std::swap(m, n);
  }
  cmatrix_t temp_A = A;
  c[0] = 0;
  while (true) {
    k1 = k + 1;
    z = 0.0;
    for (i = k; i < m; i++) {
      z = z + norm(A(i, k));
    }
    b[k] = 0.0;
    if (tol < z) {
      z = std::sqrt(z);
      b[k] = z;
      w = std::abs(A(k, k));

      if (Linalg::almost_equal(w, 0.0, zero_threshold)) {
        q = complex_t(1.0, 0.0);
      } else {
        q = A(k, k) / w;
      }
      A(k, k) = q * (z + w);

      if (k != n - 1) {
        for (j = k1; j < n; j++) {

          q = complex_t(0.0, 0.0);
          for (i = k; i < m; i++) {
            q = q + std::conj(A(i, k)) * A(i, j);
          }
          q = q / (z * (z + w));

          for (i = k; i < m; i++) {
            A(i, j) = A(i, j) - q * A(i, k);
          }
        }
        //
        // Phase transformation.
        //
        q = -std::conj(A(k, k)) / std::abs(A(k, k));

        for (j = k1; j < n; j++) {
          A(k, j) = q * A(k, j);
        }
      }
    }
    if (k == n - 1)
      break;

    z = 0.0;
    for (j = k1; j < n; j++) {
      z = z + norm(A(k, j));
    }
    c[k1] = 0.0;

    if (tol < z) {
      z = std::sqrt(z);
      c[k1] = z;
      w = std::abs(A(k, k1));

      if (Linalg::almost_equal(w, 0.0, zero_threshold)) {
        q = complex_t(1.0, 0.0);
      } else {
        q = A(k, k1) / w;
      }
      A(k, k1) = q * (z + w);

      for (i = k1; i < m; i++) {
        q = complex_t(0.0, 0.0);

        for (j = k1; j < n; j++) {
          q = q + std::conj(A(k, j)) * A(i, j);
        }
        q = q / (z * (z + w));

        for (j = k1; j < n; j++) {
          A(i, j) = A(i, j) - q * A(k, j);
        }
      }
      //
      // Phase transformation.
      //
      q = -std::conj(A(k, k1)) / std::abs(A(k, k1));
      for (i = k1; i < m; i++) {
        A(i, k1) = A(i, k1) * q;
      }
    }
    k = k1;
  }

  eps = 0.0;
  for (k = 0; k < n; k++) {
    S[k] = b[k];
    t[k] = c[k];
    eps = std::max(eps, S[k] + t[k]);
  }
  eps = eps * eta;

  //
  // Initialization of U and V.
  //
  U.initialize(m, m);
  V.initialize(n, n);
  for (j = 0; j < m; j++) {
    for (i = 0; i < m; i++) {
      U(i, j) = complex_t(0.0, 0.0);
    }
    U(j, j) = complex_t(1.0, 0.0);
  }

  for (j = 0; j < n; j++) {
    for (i = 0; i < n; i++) {
      V(i, j) = complex_t(0.0, 0.0);
    }
    V(j, j) = complex_t(1.0, 0.0);
  }

  for (k = n - 1; k >= 0; k--) {
    while (true) {
      bool jump = false;
      for (l = k; l >= 0; l--) {

        if (std::abs(t[l]) < eps) {
          jump = true;
          break;
        } else if (std::abs(S[l - 1]) < eps) {
          break;
        }
      }
      if (!jump) {
        cs = 0.0;
        sn = 1.0;
        l1 = l - 1;

        for (i = l; i <= k; i++) {
          f = sn * t[i];
          t[i] = cs * t[i];

          if (std::abs(f) < eps) {
            break;
          }
          h = S[i];
          w = std::sqrt(f * f + h * h);
          S[i] = w;
          cs = h / w;
          sn = -f / w;

          for (j = 0; j < n; j++) {
            x = std::real(U(j, l1));
            y = std::real(U(j, i));
            U(j, l1) = complex_t(x * cs + y * sn, 0.0);
            U(j, i) = complex_t(y * cs - x * sn, 0.0);
          }
        }
      }
      w = S[k];
      if (l == k) {
        break;
      }
      x = S[l];
      y = S[k - 1];
      g = t[k - 1];
      h = t[k];
      f = ((y - w) * (y + w) + (g - h) * (g + h)) / (2.0 * h * y);
      g = std::sqrt(f * f + 1.0);
      if (f < -1.0e-13) { // if ( f < 0.0){ //didn't work when f was negative
                          // very close to 0 (because of numerical reasons)
        g = -g;
      }
      f = ((x - w) * (x + w) + (y / (f + g) - h) * h) / x;
      cs = 1.0;
      sn = 1.0;
      l1 = l + 1;
      for (i = l1; i <= k; i++) {
        g = t[i];
        y = S[i];
        h = sn * g;
        g = cs * g;
        w = std::sqrt(h * h + f * f);
        if (Linalg::almost_equal(w, 0.0, zero_threshold)) {
#ifdef DEBUG
          std::cout << "ERROR 1: w is exactly 0: h = " << h << " , f = " << f
                    << std::endl;
          std::cout << " w = " << w << std::endl;
#endif
        }
        t[i - 1] = w;
        cs = f / w;
        sn = h / w;
        f = x * cs + g * sn; // might be 0

        long double large_f = 0;
        if (Linalg::almost_equal(f, 0.0, zero_threshold)) {
#ifdef DEBUG
          std::cout << "f == 0 because "
                    << "x = " << x << ", cs = " << cs << ", g = " << g
                    << ", sn = " << sn << std::endl;
#endif
          long double large_x = x * tiny_factor;
          long double large_g = g * tiny_factor;
          long double large_cs = cs * tiny_factor;
          long double large_sn = sn * tiny_factor;
          large_f = large_x * large_cs + large_g * large_sn;

#ifdef DEBUG
          std::cout << large_x * large_cs << std::endl;
          ;
          std::cout << large_g * large_sn << std::endl;
          std::cout << "new f = " << large_f << std::endl;

#endif
        }
        g = g * cs - x * sn;
        h = y * sn; // h == 0 because y==0
        y = y * cs;

        for (j = 0; j < n; j++) {
          x = std::real(V(j, i - 1));
          w = std::real(V(j, i));
          V(j, i - 1) = complex_t(x * cs + w * sn, 0.0);
          V(j, i) = complex_t(w * cs - x * sn, 0.0);
        }

        bool tiny_w = false;
#ifdef DEBUG
        std::cout << " h = " << h << " f = " << f << " large_f = " << large_f
                  << std::endl;
#endif
        if (std::abs(h) < 1e-13 && std::abs(f) < 1e-13 &&
            !Linalg::almost_equal<long double>(large_f, 0.0, zero_threshold)) {
          tiny_w = true;
        } else {
          w = std::sqrt(h * h + f * f);
        }
        w = std::sqrt(h * h + f * f);
        if (Linalg::almost_equal(w, 0.0, zero_threshold) && !tiny_w) {

#ifdef DEBUG
          std::cout << "ERROR: w is exactly 0: h = " << h << " , f = " << f
                    << std::endl;
          std::cout << " w = " << w << std::endl;
#endif
          return FAILURE;
        }

        S[i - 1] = w;
        if (tiny_w) {
          cs = 1.0; // because h==0, so w = f
          sn = 0;
        } else {
          cs = f / w;
          sn = h / w;
        }

        f = cs * g + sn * y;
        x = cs * y - sn * g;
        for (j = 0; j < n; j++) {
          y = std::real(U(j, i - 1));
          w = std::real(U(j, i));
          U(j, i - 1) = complex_t(y * cs + w * sn, 0.0);
          U(j, i) = complex_t(w * cs - y * sn, 0.0);
        }
      }
      t[l] = 0.0;
      t[k] = f;
      S[k] = x;
    }

    if (w < -1e-13) //
    {
      S[k] = -w;
      for (j = 0; j < n; j++) {
        V(j, k) = -V(j, k);
      }
    }
  }

  //
  //  Sort the singular values.
  //
  for (k = 0; k < n; k++) {
    g = -1.0;
    j = k;
    for (i = k; i < n; i++) {
      if (g < S[i]) {
        g = S[i];
        j = i;
      }
    }

    if (j != k) {
      S[j] = S[k];
      S[k] = g;

      for (i = 0; i < n; i++) {
        q = V(i, j);
        V(i, j) = V(i, k);
        V(i, k) = q;
      }

      for (i = 0; i < n; i++) {
        q = U(i, j);
        U(i, j) = U(i, k);
        U(i, k) = q;
      }
    }
  }

  for (k = n - 1; k >= 0; k--) {
    if (!Linalg::almost_equal(b[k], 0.0, zero_threshold)) {
      q = -A(k, k) / std::abs(A(k, k));
      for (j = 0; j < m; j++) {
        U(k, j) = q * U(k, j);
      }
      for (j = 0; j < m; j++) {
        q = complex_t(0.0, 0.0);
        for (i = k; i < m; i++) {
          q = q + std::conj(A(i, k)) * U(i, j);
        }
        q = q / (std::abs(A(k, k)) * b[k]);
        for (i = k; i < m; i++) {
          U(i, j) = U(i, j) - q * A(i, k);
        }
      }
    }
  }

  for (k = n - 1 - 1; k >= 0; k--) {
    k1 = k + 1;
    if (!Linalg::almost_equal(c[k1], 0.0, zero_threshold)) {
      q = -std::conj(A(k, k1)) / std::abs(A(k, k1));

      for (j = 0; j < n; j++) {
        V(k1, j) = q * V(k1, j);
      }

      for (j = 0; j < n; j++) {
        q = complex_t(0.0, 0.0);
        for (i = k1; i < n; i++) {
          q = q + A(k, i) * V(i, j);
        }
        q = q / (std::abs(A(k, k1)) * c[k1]);
        for (i = k1; i < n; i++) {
          V(i, j) = V(i, j) - q * std::conj(A(k, i));
        }
      }
    }
  }
#ifdef DEBUG
  validate_SVD_result(temp_A, U, S, V);
#endif

  // Transpose again if m < n
  if (transposed)
    std::swap(U, V);

  return SUCCESS;
}

void csvd_wrapper(cmatrix_t &A, cmatrix_t &U, rvector_t &S, cmatrix_t &V,
                  bool lapack) {
  if (lapack) {
    lapack_csvd_wrapper(A, U, S, V);
  } else {
    qiskit_csvd_wrapper(A, U, S, V);
  }
}

void qiskit_csvd_wrapper(cmatrix_t &A, cmatrix_t &U, rvector_t &S,
                         cmatrix_t &V) {
  cmatrix_t copied_A = A;
  int times = 0;
#ifdef DEBUG
  std::cout << "1st try" << std::endl;
#endif
  status current_status = csvd(A, U, S, V);
  if (current_status == SUCCESS) {
    return;
  }

  while (times <= NUM_SVD_TRIES && current_status == FAILURE) {
    times++;
    copied_A = copied_A * mul_factor;
    A = copied_A;

#ifdef DEBUG
    std::cout << "SVD trial #" << times << std::endl;
#endif

    current_status = csvd(A, U, S, V);
  }
  if (times == NUM_SVD_TRIES) {
    std::stringstream ss;
    ss << "SVD failed";
    throw std::runtime_error(ss.str());
  }

  // Divide by mul_factor every singular value after we multiplied matrix a
  for (uint_t k = 0; k < S.size(); k++)
    S[k] /= pow(mul_factor, times);
}

namespace {
int checked_lapack_integer(size_t value) {
  static_assert(sizeof(int) == 4, "LP64 LAPACK requires 32-bit integers");
  if (value == 0 || value > static_cast<size_t>(INT_MAX))
    throw std::overflow_error("LAPACK dimension/workspace exceeds LP64 range");
  return static_cast<int>(value);
}

int queried_lapack_workspace(complex_t value, int info) {
  if (info != 0)
    throw std::runtime_error("SVD workspace query failed: LAPACK info=" +
                             std::to_string(info));
  if (!std::isfinite(value.real()) || value.real() < 1 ||
      value.real() > INT_MAX)
    throw std::runtime_error("SVD workspace query returned an invalid size");
  return static_cast<int>(value.real());
}

size_t gesdd_real_workspace(size_t m, size_t n) {
  // MKL 2026 ZGESDD, JOBZ='A': k*max(5*k+7, 2*max(m,n)+2*k+1).
  const size_t k = std::min(m, n), largest = std::max(m, n);
  const size_t limit = std::numeric_limits<size_t>::max();
  if (k > (limit - 7) / 5 || largest > (limit - 1) / 2 ||
      k > (limit - 1 - 2 * largest) / 2)
    throw std::overflow_error("SVD real workspace size overflow");
  const size_t factor = std::max(5 * k + 7, 2 * largest + 2 * k + 1);
  if (k && factor > limit / k)
    throw std::overflow_error("SVD real workspace size overflow");
  return std::max(size_t(1), k * factor);
}
void validate_fallback_factors(const cmatrix_t &U, const rvector_t &S,
                               const cmatrix_t &VH, double tolerance) {
  if (U.GetRows() != U.GetColumns() || VH.GetRows() != VH.GetColumns() ||
      S.size() != std::min(U.GetRows(), VH.GetRows()))
    throw std::runtime_error("fallback SVD factor dimensions differ");
  for (size_t i = 0; i < S.size(); ++i)
    if (!std::isfinite(S[i]) || S[i] < 0 || (i && S[i] > S[i - 1]))
      throw std::runtime_error("fallback singular values invalid or unordered");
  // Check the complete returned bases, including zero-singular-value subspaces.
  for (const auto *matrix : {&U, &VH}) {
    const size_t dim = matrix->GetRows();
    for (size_t i = 0; i < dim; ++i)
      for (size_t j = 0; j <= i; ++j) {
        complex_t inner = 0.;
        for (size_t k = 0; k < dim; ++k)
          inner += matrix == &U ? std::conj((*matrix)(k, i)) * (*matrix)(k, j)
                                : (*matrix)(i, k) * std::conj((*matrix)(j, k));
        if (!std::isfinite(inner.real()) || !std::isfinite(inner.imag()) ||
            std::abs(inner - complex_t(i == j ? 1. : 0.)) > tolerance)
          throw std::runtime_error("fallback singular vectors not orthonormal");
      }
  }
}
} // namespace

void lapack_csvd_wrapper(cmatrix_t &A, cmatrix_t &U, rvector_t &S,
                         cmatrix_t &V) {
  const int m = checked_lapack_integer(A.GetRows());
  const int n = checked_lapack_integer(A.GetColumns());
  const size_t min_dim = std::min(m, n), max_dim = std::max(m, n);
  if (min_dim > (static_cast<size_t>(INT_MAX) - max_dim) / 2)
    throw std::overflow_error("LAPACK workspace exceeds LP64 range");
  int lwork = checked_lapack_integer(2 * min_dim + max_dim);
  const cmatrix_t original = A;

  U.resize(m, m);
  V.resize(n, n);
  // Matrices retain their buffers if any allocation or validation throws.
  complex_t *lapackA = A.data(), *lapackU = U.data(), *lapackV = V.data();
  const auto singular_values = std::make_unique<double[]>(min_dim);
  const auto work = std::make_unique<complex_t[]>(lwork);
  int info = 0;
  const bool use_gesdd = m >= 64 && n >= 64;

  if (use_gesdd) {
    const auto iwork = std::make_unique<int[]>(8 * min_dim);
    const auto rwork = std::make_unique<double[]>(gesdd_real_workspace(m, n));
    lwork = -1;
    zgesdd_("A", &m, &n, lapackA, &m, singular_values.get(), lapackU, &m,
            lapackV, &n, work.get(), &lwork, rwork.get(), iwork.get(), &info);
    lwork = queried_lapack_workspace(work[0], info);
    const auto queried_work = std::make_unique<complex_t[]>(lwork);
    zgesdd_("A", &m, &n, lapackA, &m, singular_values.get(), lapackU, &m,
            lapackV, &n, queried_work.get(), &lwork, rwork.get(), iwork.get(),
            &info);
  } else {
    const auto rwork = std::make_unique<double[]>(5 * min_dim);
    zgesvd_("A", "A", &m, &n, lapackA, &m, singular_values.get(), lapackU, &m,
            lapackV, &n, work.get(), &lwork, rwork.get(), &info);
  }

  bool used_fallback = false;
  if (use_gesdd && info > 0) {
    bool finite_input = true;
    for (size_t j = 0; j < static_cast<size_t>(n); ++j)
      for (size_t i = 0; i < static_cast<size_t>(m); ++i)
        finite_input = finite_input && std::isfinite(original(i, j).real()) &&
                       std::isfinite(original(i, j).imag());
    if (finite_input) {
      // Both the failed factorization and a workspace query may overwrite A.
      auto restore_input = [&]() {
        std::copy(original.data(), original.data() + m * size_t(n), lapackA);
      };
      used_fallback = true;
      const auto rwork = std::make_unique<double[]>(5 * min_dim);
      complex_t optimal_work;
      lwork = -1;
      restore_input();
      zgesvd_("A", "A", &m, &n, lapackA, &m, singular_values.get(), lapackU, &m,
              lapackV, &n, &optimal_work, &lwork, rwork.get(), &info);
      lwork = queried_lapack_workspace(optimal_work, info);
      const auto fallback_work = std::make_unique<complex_t[]>(lwork);
      restore_input();
      zgesvd_("A", "A", &m, &n, lapackA, &m, singular_values.get(), lapackU, &m,
              lapackV, &n, fallback_work.get(), &lwork, rwork.get(), &info);
    }
  }
  if (info != 0)
    throw std::runtime_error("SVD failed: LAPACK info=" + std::to_string(info));

  S.assign(singular_values.get(), singular_values.get() + min_dim);
  validate_SVdD_result(original, U, S, V);
  if (used_fallback) {
    validate_fallback_factors(U, S, V, THRESHOLD);
    // The existing magnitude check alone cannot detect a phase error.
    for (size_t i = 0; i < static_cast<size_t>(m); ++i)
      for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
        complex_t value = 0.;
        for (size_t k = 0; k < S.size(); ++k)
          value += U(i, k) * S[k] * V(k, j);
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
            std::abs(value - original(i, j)) > THRESHOLD)
          throw std::runtime_error(
              "fallback complex SVD reconstruction failed");
      }
  }
}

} // namespace AER
