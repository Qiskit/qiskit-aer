/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#define _USE_MATH_DEFINES
#include <framework/stl_ostream.hpp>
#include <map>
#include <math.h>
#include <type_traits>

#include <framework/linalg/linalg.hpp>
#include <framework/types.hpp>
#include <framework/utils.hpp>

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "utils.hpp"

using namespace AER::Test::Utilities;

namespace {
// check if polar coordinates are almost equal
// r -> (0,inf)
// angle -> (-PI, PI)
template <typename T>
T eps() {
  return std::numeric_limits<T>::epsilon();
}

template <typename T>
bool check_polar_coords(T r, T angle, T r_2, T angle_2, T max_diff = eps<T>(),
                        T max_relative_diff = eps<T>());

template <typename T>
bool check_eigenvector(const std::vector<std::complex<T>> &expected_eigen,
                       const std::vector<std::complex<T>> &actual_eigen,
                       T max_diff = eps<T>(), T max_relative_diff = eps<T>());

// Sometimes eigenvectors differ by a factor and/or phase
// This function compares them taking this into account
template <typename T>
bool check_all_eigenvectors(const matrix<std::complex<T>> &expected,
                            const matrix<std::complex<T>> &actual,
                            T max_diff = eps<T>(),
                            T max_relative_diff = eps<T>());

template <typename T>
using scenarioData = std::tuple<std::string, matrix<std::complex<T>>,
                                matrix<std::complex<T>>, std::vector<T>>;

template <typename T>
matrix<std::complex<T>> herm_mat_2d();
template <typename T>
matrix<std::complex<T>> herm_mat_2d_eigenvectors();
template <typename T>
std::vector<T> herm_mat_2d_eigenvalues();
template <typename T>
scenarioData<T> get_herm_2d_scen();

template <typename T>
matrix<std::complex<T>> psd_mat_2d();
template <typename T>
matrix<std::complex<T>> psd_mat_2d_eigenvectors();
template <typename T>
std::vector<T> psd_mat_2d_eigenvalues();
template <typename T>
scenarioData<T> get_psd_2d_scen();

template <typename T>
matrix<std::complex<T>> psd_mat_2d_with_zero();
template <typename T>
matrix<std::complex<T>> psd_mat_2d_wiht_zero_eigenvectors();
template <typename T>
std::vector<T> psd_mat_2d_wiht_zero_eigenvalues();
template <typename T>
scenarioData<T> get_psd_mat_2d_wiht_zero_scen();
} // namespace

TEST_CASE("Basic Matrix Ops", "[matrix]") {
  auto mat = matrix<std::complex<double>>(2, 2);
  mat(0, 0) = std::complex<double>(1.0, 0.0);
  mat(0, 1) = std::complex<double>(1.0, -2.0);
  mat(1, 0) = std::complex<double>(1.0, 2.0);
  mat(1, 1) = std::complex<double>(0.5, 0.0);

  SECTION("matrix - col_index") {
    std::vector<std::complex<double>> col0{{1.0, 0.0}, {1.0, 2.0}};
    std::vector<std::complex<double>> col1{{1.0, -2.0}, {0.5, 0.0}};

    REQUIRE(compare(col0, mat.col_index(0)));
    REQUIRE(compare(col1, mat.col_index(1)));
  }

  SECTION("matrix - col_index") {
    std::vector<std::complex<double>> row0{{1.0, 0.0}, {1.0, -2.0}};
    std::vector<std::complex<double>> row1{{1.0, 2.0}, {0.5, 0.0}};

    REQUIRE(compare(row0, mat.row_index(0)));
    REQUIRE(compare(row1, mat.row_index(1)));
  }
}

TEMPLATE_TEST_CASE("Linear Algebra utilities", "[eigen_hermitian]", float,
                   double) {
  std::string scenario_name;
  matrix<std::complex<TestType>> herm_mat;
  matrix<std::complex<TestType>> expected_eigenvectors;
  std::vector<TestType> expected_eigenvalues;
  std::tie(scenario_name, herm_mat, expected_eigenvectors,
           expected_eigenvalues) =
      GENERATE(get_herm_2d_scen<TestType>(), get_psd_2d_scen<TestType>(),
               get_psd_mat_2d_wiht_zero_scen<TestType>());

  // We are checking results from a numerical method so we allow for some room
  // in comparisons
  TestType eps_threshold = 5 * eps<TestType>();

  SECTION(scenario_name + ": zheevx") {
    SECTION("sanity check - eigenvals/vecs should recreate original") {
      // sanity check
      matrix<std::complex<TestType>> sanity_value(herm_mat.GetRows(),
                                                  herm_mat.GetColumns());
      for (size_t j = 0; j < expected_eigenvalues.size(); j++) {
        sanity_value +=
            expected_eigenvalues[j] *
            AER::Utils::projector(expected_eigenvectors.col_index(j));
      }
      REQUIRE(compare(herm_mat, sanity_value, eps_threshold, eps_threshold));
    }
    SECTION("actual check - heevx returns correctly") {
      std::vector<TestType> eigenvalues;
      matrix<std::complex<TestType>> eigenvectors;
      eigensystem_hermitian(herm_mat, eigenvalues, eigenvectors);

      // test equality
      REQUIRE(check_all_eigenvectors(expected_eigenvectors, eigenvectors,
                                     eps_threshold, eps_threshold));
      REQUIRE(compare(expected_eigenvalues, eigenvalues, eps_threshold,
                      eps_threshold));
      // test reconstruction
      matrix<std::complex<TestType>> value(herm_mat.GetRows(),
                                           herm_mat.GetColumns());
      for (size_t j = 0; j < eigenvalues.size(); j++) {
        value +=
            AER::Utils::projector(eigenvectors.col_index(j)) * eigenvalues[j];
      }
      REQUIRE(compare(herm_mat, value, eps_threshold, eps_threshold));
    }
  }
}

TEST_CASE("Framework Utilities", "[almost_equal]") {
  SECTION("The maximum difference between two scalars over 1.0 is greater than "
          "epsilon, so they are amlmost equal") {
    double first = 1.0 + eps<double>();
    double actual = 1.0;
    // Because the max_diff param is bigger than epsilon, this should be almost
    // equal
    REQUIRE(AER::Linalg::almost_equal(first, actual)); //, 1e-15, 1e-15));
  }

  SECTION("The difference between two scalars really close to 0 should say are "
          "almost equal") {
    double first = 5e-323; // Really close to the min magnitude of double
    double actual = 6e-323;
    REQUIRE(AER::Linalg::almost_equal(first, actual)); //, 1e-323, 1e-323));
  }

  SECTION("The maximum difference between two complex of doubles over 1.0 is "
          "greater than epsilon, so they are almost equal") {
    std::complex<double> first = {eps<double>() + double(1.0),
                                  eps<double>() + double(1.0)};
    std::complex<double> actual{1.0, 1.0};
    // Because the max_diff param is bigger than epsilon, this should be almost
    // equal
    REQUIRE(AER::Linalg::almost_equal(first, actual)); //, 1e-15, 1e-15));
  }

  SECTION("The difference between two complex of doubles really close to 0 "
          "should say are almost equal") {
    std::complex<double> first = {
        5e-323, 5e-323}; // Really close to the min magnitude of double
    std::complex<double> actual = {6e-323, 6e-323};

    REQUIRE(AER::Linalg::almost_equal(first, actual)); // 1e-323, 1e-323));
  }
}

TEST_CASE("Test_utils", "[check_polar_coords]") {
  auto r = 1.0;
  auto angle = M_PI_2;
  auto r_2 = 1.0 + eps<double>();
  auto angle_2 = M_PI_2 + eps<double>();

  SECTION("Check 2 numbers that are equal") {
    REQUIRE(check_polar_coords(r, angle, r_2, angle_2));
  }

  SECTION("Check 2 numbers that differ in absolute value") {
    r_2 = r_2 + 1e3 * eps<double>();
    REQUIRE(!check_polar_coords(r, angle, r_2, angle_2));
  }

  SECTION("Check 2 numbers that differ in absolute value") {
    angle_2 = angle_2 + 1e3 * eps<double>();
    REQUIRE(!check_polar_coords(r, angle, r_2, angle_2));
  }

  SECTION("Check corner case: close to +/-0 angles") {
    angle = 0.0 - eps<double>() / 2.;
    angle_2 = -angle;
    REQUIRE(check_polar_coords(r, angle, r_2, angle_2));
  }

  SECTION("Check corner case: angle PI and angle -PI") {
    angle = M_PI - eps<double>();
    angle_2 = -angle;
    REQUIRE(check_polar_coords(r, angle, r_2, angle_2));
  }
}

namespace {
template <typename T>
matrix<std::complex<T>> herm_mat_2d() {
  auto mat = matrix<std::complex<T>>(2, 2);
  mat(0, 0) = std::complex<T>(1.0, 0.0);
  mat(0, 1) = std::complex<T>(1.0, 2.0);
  mat(1, 0) = std::complex<T>(1.0, -2.0);
  mat(1, 1) = std::complex<T>(0.5, 0.0);
  return mat;
}

template <typename T>
matrix<std::complex<T>> herm_mat_2d_eigenvectors() {
  auto mat = matrix<std::complex<T>>(2, 2);
  auto den = 3. * std::sqrt(5.);
  mat(0, 0) = std::complex<T>(-2 / den, -4 / den);
  mat(1, 0) = std::complex<T>(5 / den, 0.0);
  mat(0, 1) = std::complex<T>(-1. / 3., -2. / 3.);
  mat(1, 1) = std::complex<T>(-2. / 3., 0);
  return mat;
}

template <typename T>
std::vector<T> herm_mat_2d_eigenvalues() {
  return {-1.5, 3.0};
}

template <typename T>
scenarioData<T> get_herm_2d_scen() {
  return {"Hermitian matrix 2x2", herm_mat_2d<T>(),
          herm_mat_2d_eigenvectors<T>(), herm_mat_2d_eigenvalues<T>()};
}

template <typename T>
matrix<std::complex<T>> psd_mat_2d() {
  auto psd_matrix = matrix<std::complex<T>>(2, 2);

  psd_matrix(0, 0) = std::complex<T>(13., 0.);
  psd_matrix(0, 1) = std::complex<T>(0., 5.);

  psd_matrix(1, 0) = std::complex<T>(0., -5.);
  psd_matrix(1, 1) = std::complex<T>(2., 0.);

  return psd_matrix;
}

template <typename T>
matrix<std::complex<T>> psd_mat_2d_eigenvectors() {
  matrix<std::complex<T>> expected_eigenvectors(2, 2);

  expected_eigenvectors(0, 0) = std::complex<T>(0, -0.3605966767761846214491);
  expected_eigenvectors(0, 1) = std::complex<T>(0, -0.9327218431547380506075);

  expected_eigenvectors(1, 0) = std::complex<T>(0.9327218431547380506075, 0);
  expected_eigenvectors(1, 1) = std::complex<T>(-0.3605966767761846214491, 0);

  return expected_eigenvectors;
}

template <typename T>
std::vector<T> psd_mat_2d_eigenvalues() {
  return {0.06696562634074720854471252, 14.93303437365925212532147};
}

template <typename T>
scenarioData<T> get_psd_2d_scen() {
  return {"PSD matrix 2x2", psd_mat_2d<T>(), psd_mat_2d_eigenvectors<T>(),
          psd_mat_2d_eigenvalues<T>()};
}

template <typename T>
matrix<std::complex<T>> psd_mat_2d_with_zero() {
  auto psd_matrix = matrix<std::complex<T>>(2, 2);

  psd_matrix(0, 0) = std::complex<T>(1., 0.);
  psd_matrix(0, 1) = std::complex<T>(2., 0.);

  psd_matrix(1, 0) = std::complex<T>(2., 0.);
  psd_matrix(1, 1) = std::complex<T>(4., 0.);

  return psd_matrix;
}

template <typename T>
matrix<std::complex<T>> psd_mat_2d_wiht_zero_eigenvectors() {
  matrix<std::complex<T>> expected_eigenvectors(2, 2);

  expected_eigenvectors(0, 0) = std::complex<T>(-2. / std::sqrt(5.), 0);
  expected_eigenvectors(0, 1) = std::complex<T>(1. / std::sqrt(5.), 0);

  expected_eigenvectors(1, 0) = std::complex<T>(1. / std::sqrt(5.), 0);
  expected_eigenvectors(1, 1) = std::complex<T>(2. / std::sqrt(5.), 0);

  return expected_eigenvectors;
}

template <typename T>
std::vector<T> psd_mat_2d_wiht_zero_eigenvalues() {
  return {0.0, 5.0};
}

template <typename T>
scenarioData<T> get_psd_mat_2d_wiht_zero_scen() {
  return {"PSD matrix 2x2 with a zero eigen value", psd_mat_2d_with_zero<T>(),
          psd_mat_2d_wiht_zero_eigenvectors<T>(),
          psd_mat_2d_wiht_zero_eigenvalues<T>()};
}

template <typename T>
bool check_polar_coords(T r, T angle, T r_2, T angle_2, T max_diff,
                        T max_relative_diff) {
  if (!AER::Linalg::almost_equal(r, r_2, max_diff, max_relative_diff))
    return false;
  if (!AER::Linalg::almost_equal(angle, angle_2, max_diff, max_relative_diff)) {
    // May be corner case with PI and -PI
    T angle_plus = angle > 0. ? angle : angle + 2 * M_PI;
    T angle_2_plus = angle_2 > 0. ? angle_2 : angle_2 + 2 * M_PI;
    if (!AER::Linalg::almost_equal(angle_plus, angle_2_plus, max_diff,
                                   max_relative_diff))
      return false;
  }
  return true;
}

template <typename T>
bool check_eigenvector(const std::vector<std::complex<T>> &expected_eigen,
                       const std::vector<std::complex<T>> &actual_eigen,
                       T max_diff, T max_relative_diff) {
  auto div = expected_eigen[0] / actual_eigen[0];
  T r = std::abs(div);
  T angle = std::arg(div);
  for (size_t j = 1; j < expected_eigen.size(); j++) {
    auto div_2 = expected_eigen[j] / actual_eigen[j];
    T r_2 = std::abs(div_2);
    T angle_2 = std::arg(div_2);
    // Check that factor is consistent across components
    if (!check_polar_coords(r, angle, r_2, angle_2, max_diff,
                            max_relative_diff)) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool check_all_eigenvectors(const matrix<std::complex<T>> &expected,
                            const matrix<std::complex<T>> &actual, T max_diff,
                            T max_relative_diff) {
  auto col_num = expected.GetColumns();
  if (expected.size() != actual.size() ||
      expected.GetColumns() != expected.GetColumns()) {
    return false;
  }
  for (size_t i = 0; i < col_num; i++) {
    auto expected_eigen = expected.col_index(i);
    auto actual_eigen = actual.col_index(i);

    if (!check_eigenvector(expected_eigen, actual_eigen, max_diff,
                           max_relative_diff)) {
      std::cout << "Expected: " << std::setprecision(16) << expected
                << std::endl;
      std::cout << "Actual: " << actual << std::endl;
      return false;
    }
  }
  return true;
}
} // namespace