#include <iostream>
#include <complex>
#include <bitset>
#include <vector>
#include <chrono>
#include "qv_avx2.hpp"

#include "framework/utils.hpp"
#include "simulators/statevector/indexes.hpp"

using myclock_t = std::chrono::high_resolution_clock;

using namespace AER::QV;

// Numeric Types
using int_t = int_fast64_t;
using uint_t = uint_fast64_t;

template<typename data_t>
std::vector<std::complex<data_t>> convert(const std::vector<std::complex<data_t>> &v) {
  std::vector<std::complex<data_t>> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    ret[i] = v[i];
  return ret;
}

template<typename data_t>
void apply_diagonal_matrix(std::complex<data_t>* &data, size_t data_size, int threads, const std::vector<uint_t> &qubits, const std::vector<std::complex<data_t>> &diag) {
  const size_t N = qubits.size();
  auto func = [&](const areg_t<2> &inds,
      const std::vector<std::complex<data_t>> &_diag) -> void {
    for (int_t i = 0; i < 2; ++i) {
      const int_t k = inds[i];
      int_t iv = 0;
      for (int_t j = 0; j < N; j++)
        if ((k & (1ULL << qubits[j])) != 0)
           iv += (1ULL << j);
      if (_diag[iv] != (data_t)1.0)
      data[k] *= _diag[iv];
    }
  };
  apply_lambda(0, data_size, threads, func, areg_t<1>({{qubits[0]}}), convert(diag));
}

template<typename data_t>
bool equal(std::complex<data_t> left, std::complex<data_t> right, double round = 1e-9) {

  if (abs(left.real() - right.real()) > round)
    return false;
  if (abs(left.imag() - right.imag()) > round)
    return false;
  return true;
}

bool next_distinct_qubits(std::vector<uint64_t>& qubits, uint64_t qubit_size) {
  while (true) {
    ++qubits[0];
    for (auto i = 0; i < qubits.size(); ++i) {
      if (qubits[i] != qubit_size)
        break;
      if (i + 1 == qubits.size())
        return false;
      qubits[i] = 0;
      ++qubits[i + 1];
    }

    bool duplicate = false;
    for (auto i = 0; i < qubits.size(); ++i) {
      for (auto j = i + 1; j < qubits.size(); ++j) {
        if (qubits[i] == qubits[j]) {
          duplicate = true;
          break;
        }
      }
      if (duplicate)
        break;
    }
    if (!duplicate)
      return true;
  }
}

template<typename data_t>
std::complex<data_t>* alloc_qv(size_t size) {
  void* data;
  posix_memalign(&data, 64, sizeof(std::complex<data_t>) * size);
  return reinterpret_cast<std::complex<data_t>*>(data);
}

template<typename data_t>
int test_apply_diagonal_matrix_avx(int qubit_size, bool log_success = false) {

  auto data_size = 1UL << qubit_size;
  std::complex<data_t>* qv = alloc_qv<data_t>(data_size);

  for (auto input_qubit_size = 1; input_qubit_size < 8; ++input_qubit_size) {
    std::vector<uint64_t> input_qubits;
    input_qubits.assign(input_qubit_size, 0);
    --input_qubits[0];
    while (next_distinct_qubits(input_qubits, qubit_size)) {

      auto input_size = 1UL << input_qubit_size;
      std::complex<data_t> input[input_size];

      for (auto i = 0; i < input_size; ++i)
        input[i] = std::complex<double>((data_t) (i + 1), (data_t) (i + 2));

      for (auto i = 0; i < data_size; ++i)
        qv[i] = 1.;

      if (apply_diagonal_matrix_avx<data_t>((data_t*) qv, data_size, input_qubits.data(), input_qubit_size, (data_t*) input, 1) != Avx::Applied) {
        std::cout << "apply_matrix_avx: failed (not applied)" << std::endl;
        free(qv);
        return -1;
      }

      for (auto i = 0; i < data_size; ++i) {
        auto input_idx = 0;
        for (auto j = 0; j < input_qubit_size; ++j)
          if (i & (1UL << input_qubits[j]))
            input_idx |= (1UL << j);

        if (!equal(qv[i], input[input_idx])) {
          std::cout << "test_apply_diagonal_matrix_avx: failed (qv[" << i << "] == " << qv[i] << " != " << input[input_idx] << ")" << std::endl;
          free(qv);
          return -1;
        }
      }

      if (log_success) {
        std::cout << "  test_apply_diagonal_matrix_avx: success [";
        for (auto j = 0; j < input_qubit_size; ++j)
          if (j == 0)
            std::cout << input_qubits[j];
          else
            std::cout << ", " << input_qubits[j];
        std::cout << "]-q" << std::endl;
      }

    }
  }

  free(qv);

  return 1;
}

template<typename data_t>
double perf_apply_diagonal_matrix_avx(int qubit_size, int input_qubit_size, int iteration) {
  auto data_size = 1UL << qubit_size;
  std::complex<data_t>* qv = alloc_qv<data_t>(data_size);

  auto input_size = 1UL << input_qubit_size;
  std::complex<data_t> input[input_size];

  double total = 0.;

  std::vector<uint64_t> input_qubits;
  input_qubits.assign(input_qubit_size, 0);
  --input_qubits[0];
  while (next_distinct_qubits(input_qubits, qubit_size)) {
    if (iteration-- == 0)
      break;

    auto input_size = 1UL << input_qubit_size;
    std::complex<data_t> input[input_size];

    for (auto i = 0; i < input_size; ++i)
      input[i] = std::complex<double>((data_t) (i + 1), (data_t) (i + 2));

#pragma omp parallel
    for (auto i = 0; i < data_size; ++i)
      qv[i] = 1.;

    auto timer_start = myclock_t::now();
    if (apply_diagonal_matrix_avx<data_t>((data_t*) qv, data_size, input_qubits.data(), input_qubit_size, (data_t*) input, 8) != Avx::Applied) {
      std::cout << "apply_matrix_avx: failed (not applied)" << std::endl;
      free(qv);
      return -1;
    }
    auto timer_stop = myclock_t::now();
    total += std::chrono::duration<double>(timer_stop - timer_start).count();
  }

  free(qv);

  return total;
}

template<typename data_t>
double perf_apply_diagonal_matrix(int qubit_size, int input_qubit_size, int iteration) {
  auto data_size = 1UL << qubit_size;
  std::complex<data_t>* qv = alloc_qv<data_t>(data_size);

  auto input_size = 1UL << input_qubit_size;
  std::vector<std::complex<data_t>> input;
  input.assign(input_size, 0.);

  double total = 0.;

  std::vector<uint64_t> input_qubits;
  input_qubits.assign(input_qubit_size, 0);
  --input_qubits[0];
  while (next_distinct_qubits(input_qubits, qubit_size)) {
    if (iteration-- == 0)
      break;

    auto input_size = 1UL << input_qubit_size;

    for (auto i = 0; i < input_size; ++i)
      input[i] = std::complex<double>((data_t) (i + 1), (data_t) (i + 2));

#pragma omp parallel
    for (auto i = 0; i < data_size; ++i)
      qv[i] = 1.;

    auto timer_start = myclock_t::now();
    apply_diagonal_matrix<data_t>(qv, data_size, 8, input_qubits, input);
    auto timer_stop = myclock_t::now();
    total += std::chrono::duration<double>(timer_stop - timer_start).count();
  }

  free(qv);

  return total;
}

int main() {

  for (int q = 1; q < 9; ++q) {
    std::cout << "test_apply_diagonal_matrix_avx<double>(" << q << ");" << std::endl;
    test_apply_diagonal_matrix_avx<double>(q);
    std::cout << "test_apply_diagonal_matrix_avx<float>(" << q << ");" << std::endl;
    test_apply_diagonal_matrix_avx<float>(q);
  }

  std::cout << "performance - double" << std::endl;
  int iteration = 10;
  for (int input_size = 3; input_size < 6; ++input_size) {
    for (int qubit_size = 20; qubit_size < 26; ++qubit_size) {
      double simd_time = perf_apply_diagonal_matrix_avx<double>(qubit_size, input_size, iteration);
      double orig_time = perf_apply_diagonal_matrix<double>(qubit_size, input_size, iteration);
      std::cout << "  qubit: " << qubit_size << ", input: " << input_size << std::endl;
      std::cout << "    simd: " << simd_time << std::endl;
      std::cout << "    orig: " << orig_time << std::endl;
    }
  }

  std::cout << "performance - float" << std::endl;
  for (int input_size = 3; input_size < 6; ++input_size) {
    for (int qubit_size = 20; qubit_size < 26; ++qubit_size) {
      double simd_time = perf_apply_diagonal_matrix_avx<float>(qubit_size, input_size, iteration);
      double orig_time = perf_apply_diagonal_matrix<float>(qubit_size, input_size, iteration);
      std::cout << "  qubit: " << qubit_size << ", input: " << input_size << std::endl;
      std::cout << "    simd: " << simd_time << std::endl;
      std::cout << "    orig: " << orig_time << std::endl;
    }
  }
}
