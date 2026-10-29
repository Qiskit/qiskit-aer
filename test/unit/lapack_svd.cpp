/**
 * This code is part of Qiskit.
 *
 * Licensed under the Apache License, Version 2.0. See LICENSE.txt.
 */

#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// Compile the real wrapper and matrix class, including its internal checks.
#include "simulators/matrix_product_state/svd.cpp"

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename F>
void require_failure(F function) {
  bool failed = false;
  try {
    function();
  } catch (const std::exception &) {
    failed = true;
  }
  require(failed, "expected an exception");
}
} // namespace

#ifdef AER_TEST_INJECT_LAPACK
namespace {
std::string scenario;
int gesdd_calls = 0, gesvd_queries = 0, gesvd_calls = 0;
int live_arrays = 0, fail_after = -1;

using AER::complex_t;
void check_original(const complex_t *a, int m, int n) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      require(a[j * size_t(m) + i] ==
                  (i == j
                       ? complex_t(0, scenario == "unordered" && i == 0 ? 2 : 1)
                       : complex_t(0)),
              "fallback did not receive the original matrix");
}

void make_factors(int m, int n, double *s, complex_t *u, complex_t *vh) {
  std::fill(u, u + size_t(m) * m, complex_t(0));
  std::fill(vh, vh + size_t(n) * n, complex_t(0));
  for (int i = 0; i < m; ++i)
    u[i * size_t(m) + i] = complex_t(0, 1);
  for (int i = 0; i < n; ++i)
    vh[i * size_t(n) + i] = 1.;
  std::fill(s, s + std::min(m, n), 1.);
  if (scenario == "nonorthogonal_u" || scenario == "negative") {
    for (int i = 0; i < m; ++i)
      u[i * size_t(m) + i] *= scenario == "negative" ? -1. : 2.;
    std::fill(s, s + std::min(m, n), scenario == "negative" ? -1. : .5);
  }
  if (scenario == "nonorthogonal_v") {
    for (int i = 0; i < n; ++i)
      vh[i * size_t(n) + i] *= 2.;
    std::fill(s, s + std::min(m, n), .5);
  }
  if (scenario == "unordered") {
    s[0] = 1.;
    s[1] = 2.;
    u[0] = u[m + 1] = vh[0] = vh[n + 1] = 0.;
    u[m] = u[1] = complex_t(0, 1);
    vh[n] = vh[1] = 1.;
  }
  if (scenario == "phase")
    for (int i = 0; i < m; ++i)
      u[i * size_t(m) + i] *= -1.;
  if (scenario == "nonfinite_factors")
    u[0] = std::numeric_limits<double>::quiet_NaN();
  if (scenario == "nonorthogonal_nullspace")
    (m > n ? u[(m - 1) * size_t(m) + m - 1]
           : vh[(n - 1) * size_t(n) + n - 1]) *= 2.;
}

void query(complex_t *work, int *info, const char *prefix) {
  *info = scenario == std::string(prefix) + "_query_error" ? -12 : 0;
  work[0] = 512.;
  if (scenario == std::string(prefix) + "_query_nan")
    work[0] = std::numeric_limits<double>::quiet_NaN();
  if (scenario == std::string(prefix) + "_query_zero")
    work[0] = 0.;
  if (scenario == std::string(prefix) + "_query_overflow")
    work[0] = double(INT_MAX) + 1.;
}
} // namespace

void *operator new[](size_t bytes) {
  if (fail_after >= 0 && fail_after-- == 0)
    throw std::bad_alloc();
  void *value = std::malloc(bytes ? bytes : 1);
  if (!value)
    throw std::bad_alloc();
  ++live_arrays;
  return value;
}
void operator delete[](void *value) noexcept {
  if (value) {
    --live_arrays;
    std::free(value);
  }
}
void operator delete[](void *value, size_t) noexcept {
  ::operator delete[](value);
}

extern "C" void zgesdd_(const char *jobz, const int *m, const int *n,
                        complex_t *a, const int *lda, double *s, complex_t *u,
                        const int *ldu, complex_t *vh, const int *ldvh,
                        complex_t *work, const int *lwork, double *rwork,
                        int *iwork, int *info) {
  require(*jobz == 'A' && *lda == *m && *ldu == *m && *ldvh == *n,
          "incorrect GESDD arguments");
  if (*lwork == -1) {
    query(work, info, "primary");
    return;
  }
  // ASan checks a write to the last documented real/integer workspace entries.
  const size_t k = std::min(*m, *n), largest = std::max(*m, *n);
  rwork[k * std::max(5 * k + 7, 2 * largest + 2 * k + 1) - 1] = 1.;
  iwork[8 * k - 1] = 1;
  ++gesdd_calls;
  *info = scenario == "primary_success" ? 0
          : scenario == "invalid_args"  ? -4
                                        : 1;
  if (*info == 0)
    make_factors(*m, *n, s, u, vh);
  std::fill(a, a + size_t(*m) * *n, complex_t(777));
}

extern "C" void zgesvd_(const char *jobu, const char *jobvt, const int *m,
                        const int *n, complex_t *a, const int *lda, double *s,
                        complex_t *u, const int *ldu, complex_t *vh,
                        const int *ldvh, complex_t *work, const int *lwork,
                        double *rwork, int *info) {
  require(*jobu == 'A' && *jobvt == 'A' && *lda == *m && *ldu == *m &&
              *ldvh == *n,
          "incorrect GESVD arguments");
  check_original(a, *m, *n);
  rwork[5 * size_t(std::min(*m, *n)) - 1] = 1.;
  if (*lwork == -1) {
    ++gesvd_queries;
    query(work, info, "fallback");
    std::fill(a, a + size_t(*m) * *n, complex_t(888));
    return;
  }
  ++gesvd_calls;
  *info =
      scenario == "fallback_failure" || scenario == "ordinary_failure" ? 1 : 0;
  make_factors(*m, *n, s, u, vh);
}
#endif

int main() {
  using namespace AER;
  try {
    require(gesdd_real_workspace(86, 80) == 32560, "86x80 real workspace");
    require(gesdd_real_workspace(80, 86) == 32560, "80x86 real workspace");
    require(gesdd_real_workspace(64, 64) == 20928, "64x64 real workspace");
    require_failure([] { gesdd_real_workspace(SIZE_MAX, SIZE_MAX); });
    for (size_t dimension : {size_t(0), size_t(INT_MAX) + 1})
      require_failure([&] { checked_lapack_integer(dimension); });
    require(checked_lapack_integer(INT_MAX) == INT_MAX, "LP64 upper bound");
    for (auto shape : {std::pair<size_t, size_t>{0, 2},
                       {2, 0},
                       {size_t(INT_MAX) + 1, 1},
                       {1, size_t(INT_MAX) + 1},
                       {INT_MAX, 1},
                       {1, INT_MAX}}) {
      cmatrix_t a(shape.first, shape.second, static_cast<complex_t *>(nullptr));
      cmatrix_t u, v;
      rvector_t s;
      require_failure([&] { lapack_csvd_wrapper(a, u, s, v); });
    }
#ifdef AER_TEST_INJECT_LAPACK
    size_t cases = 0;
    for (auto shape : {std::pair<int, int>{64, 64}, {86, 80}, {80, 86}})
      for (const auto *label : {"good",
                                "primary_success",
                                "invalid_args",
                                "primary_query_error",
                                "primary_query_nan",
                                "primary_query_zero",
                                "primary_query_overflow",
                                "nonfinite_input",
                                "fallback_query_error",
                                "fallback_query_nan",
                                "fallback_query_zero",
                                "fallback_query_overflow",
                                "fallback_failure",
                                "phase",
                                "nonfinite_factors",
                                "nonorthogonal_u",
                                "nonorthogonal_v",
                                "negative",
                                "unordered",
                                "nonorthogonal_nullspace"}) {
        scenario = label;
        if (scenario == "nonorthogonal_nullspace" &&
            shape.first == shape.second)
          continue;
        gesdd_calls = gesvd_queries = gesvd_calls = 0;
        cmatrix_t a(shape.first, shape.second), u, v;
        for (int i = 0; i < std::min(shape.first, shape.second); ++i)
          a(i, i) = complex_t(0, scenario == "unordered" && i == 0 ? 2 : 1);
        if (scenario == "nonfinite_input")
          a(0, 0) = std::numeric_limits<double>::quiet_NaN();
        rvector_t s;
        const bool success =
            scenario == "good" || scenario == "primary_success";
        if (success)
          lapack_csvd_wrapper(a, u, s, v);
        else
          require_failure([&] { lapack_csvd_wrapper(a, u, s, v); });
        require(gesvd_queries <= 1 && gesvd_calls <= 1, "repeated recovery");
        if (scenario == "invalid_args" || scenario == "nonfinite_input" ||
            scenario.find("primary_query") == 0 ||
            scenario == "primary_success")
          require(gesvd_queries == 0 && gesvd_calls == 0,
                  "recovery after an ineligible primary result");
        if (scenario == "good")
          require(gesdd_calls == 1 && gesvd_queries == 1 && gesvd_calls == 1,
                  "missing recovery");
        ++cases;
      }
    for (const auto *label : {"good", "ordinary_failure"}) {
      scenario = label;
      gesdd_calls = gesvd_queries = gesvd_calls = 0;
      cmatrix_t a(2, 3), u, v;
      a(0, 0) = a(1, 1) = complex_t(0, 1);
      rvector_t s;
      if (scenario == "good")
        lapack_csvd_wrapper(a, u, s, v);
      else
        require_failure([&] { lapack_csvd_wrapper(a, u, s, v); });
      require(gesdd_calls == 0 && gesvd_queries == 0 && gesvd_calls == 1,
              "ordinary GESVD path changed");
      ++cases;
    }
    for (int allocation = 0; allocation < 7; ++allocation) {
      const int before = live_arrays;
      {
        scenario = "good";
        cmatrix_t a(86, 80), u, v;
        for (int i = 0; i < 80; ++i)
          a(i, i) = complex_t(0, 1);
        rvector_t s;
        fail_after = allocation;
        require_failure([&] { lapack_csvd_wrapper(a, u, s, v); });
        fail_after = -1;
        require(a.data() && u.data() && v.data(), "matrix ownership lost");
      }
      require(live_arrays == before, "array leaked after allocation failure");
      ++cases;
    }
    std::cout << cases << " injected-status/factor/allocation cases passed\n";
#else
    size_t cases = 0;
    for (auto shape : {std::pair<size_t, size_t>{1, 1},
                       {2, 3},
                       {3, 2},
                       {63, 64},
                       {64, 63},
                       {64, 64},
                       {86, 80},
                       {80, 86}})
      for (bool rank_deficient : {false, true}) {
        cmatrix_t a(shape.first, shape.second), u, v;
        for (size_t j = 0; j < shape.second; ++j)
          for (size_t i = 0; i < shape.first; ++i)
            a(i, j) = rank_deficient
                          ? complex_t(double(i + 1) * (j + 1), 0)
                          : complex_t(std::sin(double((i + 1) * (j + 1))),
                                      std::cos(double((i + 3) * (j + 2))));
        if (!rank_deficient)
          for (size_t i = 0; i < std::min(shape.first, shape.second); ++i)
            a(i, i) += 2. * std::max(shape.first, shape.second);
        const auto original = a;
        rvector_t s;
        lapack_csvd_wrapper(a, u, s, v);
        validate_fallback_factors(u, s, v, THRESHOLD);
        double squared_error = 0., squared_norm = 0.;
        for (size_t j = 0; j < shape.second; ++j)
          for (size_t i = 0; i < shape.first; ++i) {
            complex_t value = 0.;
            for (size_t k = 0; k < s.size(); ++k)
              value += u(i, k) * s[k] * v(k, j);
            squared_error += std::norm(value - original(i, j));
            squared_norm += std::norm(original(i, j));
          }
        require(std::sqrt(squared_error / squared_norm) < 1e-12,
                "complex reconstruction failed");
        ++cases;
      }
    std::cout << cases << " numerical matrix cases passed\n";
#endif
    std::cout << "workspace and six pre-allocation boundary cases passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
