/**
 * This code is part of Qiskit.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 */

#include "framework/linalg/matrix_utils.hpp"
#include "framework/rng.hpp"
// The simulator uses the same implementation inclusion in
// matrix_product_state.hpp.
#include "simulators/matrix_product_state/matrix_product_state_internal.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using AER::MatrixProductState::MPS;
using AER::MatrixProductState::MPS_Tensor;
using Complex = std::complex<long double>;
constexpr long double tolerance = 1e-10L;

Complex widen(AER::complex_t value) { return {value.real(), value.imag()}; }

void check_close(Complex actual, Complex expected, const std::string &label) {
  if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << std::setprecision(17) << label << ": actual=" << actual
              << " expected=" << expected << '\n';
    throw std::runtime_error(label);
  }
}

// Contract the exported Gamma/lambda chain directly. This oracle does not use
// MPS observable, probability, normalization, routing or statevector methods.
std::array<Complex, 16>
retained_amplitudes(const AER::mps_container_t &tensors) {
  if (tensors.first.size() != 4 || tensors.second.size() != 3)
    throw std::runtime_error("Expected a four-site fixture");
  std::array<Complex, 16> result{};
  for (unsigned basis = 0; basis < 16; ++basis) {
    std::vector<Complex> row{1};
    for (unsigned site = 0; site < 4; ++site) {
      const auto &pair = tensors.first.at(site);
      const auto &gamma = (basis & (1u << site)) ? pair.second : pair.first;
      if (row.size() != gamma.GetRows())
        throw std::runtime_error("Invalid retained tensor dimensions");
      std::vector<Complex> next(gamma.GetColumns());
      for (unsigned left = 0; left < gamma.GetRows(); ++left)
        for (unsigned right = 0; right < gamma.GetColumns(); ++right)
          next[right] += row[left] * widen(gamma(left, right));
      if (site < 3)
        for (unsigned right = 0; right < next.size(); ++right)
          next[right] *= tensors.second.at(site).at(right);
      row = std::move(next);
    }
    if (row.size() != 1)
      throw std::runtime_error("Retained contraction must end in a scalar");
    result[basis] = row.front();
  }
  return result;
}

std::array<Complex, 2>
retained_norm_and_z3(const AER::mps_container_t &tensors) {
  const auto amplitudes = retained_amplitudes(tensors);
  std::array<Complex, 2> result{};
  for (unsigned basis = 0; basis < amplitudes.size(); ++basis) {
    const auto weight = std::norm(amplitudes[basis]);
    result[0] += weight;
    result[1] += (basis & 8u) ? -weight : weight;
  }
  return result;
}

// Reduced from the R8 diagonal control: compare amplitudes, not Z readouts,
// because a diagonal phase error can leave every computational probability
// intact.
void check_diagonal() {
  unsigned cases = 0;
  long double maximum_error = 0;
  for (const bool lapack : {false, true}) {
    MPS::set_mps_lapack_svd(lapack);
    AER::reg_t sites{3, 2, 0};
    do {
      MPS mps;
      mps.set_max_bond_dimension(4);
      mps.set_truncation_threshold(0);
      mps.initialize(4);
      for (unsigned site = 0; site < 4; ++site)
        mps.apply_u3(site, 0.43 + 0.19 * site, -0.31 + 0.13 * site,
                     0.59 - 0.11 * site);
      for (unsigned site = 0; site < 3; ++site)
        mps.apply_cnot(site, site + 1);
      MPS before(mps);
      auto expected = retained_amplitudes(before.copy_to_mps_container());
      AER::cvector_t phases(8);
      for (unsigned value = 0; value < phases.size(); ++value)
        phases[value] = std::polar(1.0, 0.13 * value + 0.07 * value * value);
      bool changes_amplitude = false, distinguishes_reversed_order = false;
      for (unsigned basis = 0; basis < expected.size(); ++basis) {
        unsigned selected = 0, reversed = 0;
        for (unsigned bit = 0; bit < sites.size(); ++bit) {
          selected |= ((basis >> sites[bit]) & 1u) << bit;
          reversed |= ((basis >> sites[sites.size() - 1 - bit]) & 1u) << bit;
        }
        const auto original = expected[basis];
        expected[basis] *= widen(phases[selected]);
        changes_amplitude |= std::abs(expected[basis] - original) > tolerance;
        distinguishes_reversed_order |=
            std::abs(expected[basis] - original * widen(phases[reversed])) >
            tolerance;
      }
      if (!changes_amplitude || !distinguishes_reversed_order)
        throw std::runtime_error(
            "Diagonal fixture must distinguish phase/order");
      // apply_diagonal_matrix supplies the public 1 x 2^n row representation.
      // Let the production method perform its own gate permutation/routing.
      const auto label = std::string(lapack ? "lapack" : "native") +
                         " diagonal sites=" + std::to_string(sites[0]) + "," +
                         std::to_string(sites[1]) + "," +
                         std::to_string(sites[2]);
      std::cout << label << std::endl;
      mps.apply_diagonal_matrix(sites, phases);
      MPS after(mps);
      const auto actual = retained_amplitudes(after.copy_to_mps_container());
      for (unsigned basis = 0; basis < actual.size(); ++basis) {
        check_close(actual[basis], expected[basis],
                    label + " amplitude " + std::to_string(basis));
        maximum_error =
            std::max(maximum_error, std::abs(actual[basis] - expected[basis]));
      }
      ++cases;
    } while (std::prev_permutation(sites.begin(), sites.end()));
  }
  std::cout << std::setprecision(17) << "diagonal cases=" << cases
            << " maximum_amplitude_error=" << maximum_error << '\n';
}

void check_queries(const MPS &mps, Complex norm, Complex z,
                   const std::string &label) {
  check_close(widen(mps.expectation_value_pauli({0, 1, 2, 3}, "IIII")), norm,
              label + " full identity");
  check_close(widen(mps.expectation_value_pauli({0, 1, 2, 3}, "ZIII")), z,
              label + " full Z3");
  check_close(widen(mps.expectation_value_pauli({0, 3}, "ZI")), z,
              label + " gapped Z3");
  // Repetition exercises both initial environment construction and cache reuse.
  for (unsigned repeat = 0; repeat < 3; ++repeat)
    check_close(widen(mps.expectation_value_pauli({3}, "Z")), z,
                label + " local Z3");
}

void check_state(MPS &mps, Complex expected_z, const std::string &label) {
  // These fixtures already have sorted logical order. Export a copy so that
  // obtaining the oracle cannot mutate the state being measured.
  MPS exported(mps);
  const auto tensors = exported.copy_to_mps_container();
  const auto reference = retained_norm_and_z3(tensors);
  check_close(reference[0], 1, label + " retained norm control");
  check_close(reference[1], expected_z, label + " retained Z3 control");
  std::cout << std::setprecision(17) << label << " retained_Z3=" << reference[1]
            << " full_Z3=" << mps.expectation_value_pauli({0, 1, 2, 3}, "ZIII")
            << " local_Z3=" << mps.expectation_value_pauli({3}, "Z") << '\n';
  check_queries(mps, reference[0], reference[1], label);
  MPS after(mps);
  if (json_t(after.copy_to_mps_container()) != json_t(tensors))
    throw std::runtime_error(label + " queries changed retained tensors");

  MPS copied(mps), assigned;
  assigned = mps;
  for (const auto *copy : {&copied, &assigned}) {
    if (copy->get_max_bond_dimension() != mps.get_max_bond_dimension() ||
        copy->get_truncation_threshold() != mps.get_truncation_threshold())
      throw std::runtime_error(label +
                               " copied state lost its instance settings");
  }
  mps.apply_x(3);
  check_queries(mps, reference[0], -reference[1], label + " after X3");
  check_queries(copied, reference[0], reference[1], label + " copied");
  check_queries(assigned, reference[0], reference[1], label + " assigned");
}

void check_circuit() {
  // Original five-gate witness, with pre-SWAP and untruncated controls.
  for (const bool lapack : {false, true}) {
    MPS::set_mps_lapack_svd(lapack);
    for (const unsigned bond : {2u, 4u}) {
      for (const bool swap : {false, true}) {
        MPS mps;
        mps.set_max_bond_dimension(bond);
        mps.set_truncation_threshold(0);
        mps.initialize(4);
        mps.apply_ry(0, std::numbers::pi / 3);
        mps.apply_cnot(0, 1);
        mps.apply_ry(2, std::numbers::pi / 4);
        mps.apply_cnot(2, 3);
        if (swap)
          mps.apply_swap(1, 2, true);
        const auto expected_z =
            swap && bond == 2 ? 1.0L
                              : std::cos(std::numbers::pi_v<long double> / 4);
        const auto label = std::string(lapack ? "lapack" : "native") +
                           " bond=" + std::to_string(bond) +
                           (swap ? " after SWAP" : " before SWAP");
        check_state(mps, expected_z, label);
      }
    }
  }
}

void check_retained(const char *path) {
  // Frozen output of the original native-SVD, bond-two five-gate witness.
  // This path reproduces the readout defect without executing truncating SVD.
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Cannot open retained fixture");
  json_t value;
  input >> value;
  const auto tensors = value.get<AER::mps_container_t>();
  MPS mps(tensors.first.size());
  mps.set_max_bond_dimension(2);
  mps.set_truncation_threshold(0);
  mps.initialize_from_mps(tensors);
  check_state(mps, 1, "retained witness");
}
void check_lifecycle() {
  MPS source;
  source.set_max_bond_dimension(2);
  source.set_truncation_threshold(0.001);
  source.initialize(4);
  MPS copied(source), assigned;
  assigned = source;
  source = source;
  const auto saved = source.move_to_mps_container();
  if (source.num_qubits() != 0 || saved.first.size() != 4)
    throw std::runtime_error("Destructive export did not move the payload");
  source.initialize(4);
  for (const MPS *state : {&source, &copied, &assigned}) {
    if (state->get_max_bond_dimension() != 2 ||
        state->get_truncation_threshold() != 0.001)
      throw std::runtime_error(
          "Copy/export/reinitialization lost instance settings");
  }
  source.initialize_from_mps(saved);
  check_queries(source, 1, 1, "restored after destructive export");
}
} // namespace

int main(int argc, char **argv) {
  try {
    MPS::set_omp_threads(1);
    MPS_Tensor::set_chop_threshold(1e-16);
    if (argc == 3 && std::string(argv[1]) == "circuit")
      check_circuit();
    else if (argc == 3 && std::string(argv[1]) == "diagonal")
      check_diagonal();
    else if (argc == 3 && std::string(argv[1]) == "lifecycle")
      check_lifecycle();
    else if (argc == 3 && std::string(argv[1]) == "retained")
      check_retained(argv[2]);
    else
      throw std::runtime_error(
          "Expected circuit|retained|diagonal|lifecycle and fixture path");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
