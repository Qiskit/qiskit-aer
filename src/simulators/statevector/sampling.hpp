/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _sampling_hpp_
#define _sampling_hpp_

#include <vector>
#include <algorithm>
#include "simulators/statevector/indexes.hpp"

namespace AER {
namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;

// Get samples with saving memory with following methods of DATA.
template <typename Lambda>
reg_t sample_with_iterative_search(const std::vector<double> &rnds_,
                                   const uint_t num_qubits,
                                   const uint_t index_size,
                                   const Lambda probability_func,
                                   const uint_t omp_threads) {

  const int_t END = 1UL << num_qubits;
  const uint_t SHOTS = rnds_.size();
  const int_t INDEX_SIZE = std::min(index_size, num_qubits - 1);
  const int_t INDEX_END = 1UL << INDEX_SIZE;

  // sort random numbers
  auto rnds = rnds_;
  std::sort(rnds.begin(), rnds.end());

  reg_t samples;
  samples.assign(SHOTS, 0);

  // Initialize indices
  std::vector<double> end_probs;
  end_probs.assign(INDEX_END, 0.0);
  const uint_t LOOP = (END >> INDEX_SIZE);

  // Construct indices
  auto idxing = [&](const int_t i)->void {
    double total = .0;
    for (uint_t j = LOOP * i; j < LOOP * (i + 1); j++)
      total += probability_func(j);
    end_probs[i] = total;
  };
  QV::apply_lambda(0, INDEX_END, omp_threads, idxing);

  // accumulate indices
  for (uint_t i = 1; i < INDEX_END; ++i)
    end_probs[i] += end_probs[i - 1];

  // reduce rounding error
  double correction = 1.0 / end_probs[INDEX_END - 1];
  for (int_t i = 1; i < INDEX_END - 1; ++i)
    end_probs[i] *= correction;
  end_probs[INDEX_END - 1] = 1.0;

  // find starting index
  std::vector<int_t> starts;
  starts.assign(INDEX_END + 1, 0);
  starts[INDEX_END] = SHOTS;

  uint_t last_idx_start = 0;
  for (int_t i = 1; i < INDEX_END; ++i) {
    for (; last_idx_start < SHOTS; ++last_idx_start) {
      if (rnds[last_idx_start] < end_probs[i - 1])
        continue;
      break;
    }
    starts[i] = last_idx_start;
  }

  // sampling
  auto sampling = [&](const int_t i)->void {
    uint_t start_sample_idx = starts[i];
    uint_t end_sample_idx = starts[i + 1];
    auto sample = LOOP * i;
    double p = 0.0;
    if (i != 0)
      p = end_probs[i - 1];
    p += probability_func(sample);
    for (uint_t sample_idx = start_sample_idx; sample_idx < end_sample_idx; ++sample_idx) {
      auto rnd = rnds[sample_idx];
      while(sample < (LOOP * (i + 1)) && p < rnd) {
        ++sample;
        p += probability_func(sample);
      }
      samples[sample_idx] = sample;
    }
  };
  QV::apply_lambda(0, INDEX_END, omp_threads, sampling);

  return samples;
}

template <typename V, typename T>
int_t scan_inclusive(V& v, T& t, int_t left, int_t right) {
  //assert(left < right);

  // iterative search
  //  for (int_t i = left; i < right; ++i) {
  //    if (t < v[i])
  //      return i;
  //  }
  //  return right;

  // binary search
  int_t mid = 0;
  while (true) {
    if (left >= (right - 1))
      return t <= v[left] ? left: right;
    mid = (left + right) / 2;
    if (t <= v[mid])
      right = mid;
    else
      left = mid;
  }
}

// Get samples with consumption of memory with following methods of DATA.
template <typename Lambda>
reg_t sample_with_binary_search(const std::vector<double> &rnds,
                                const uint_t num_qubits,
                                const Lambda probability_func,
                                const uint_t omp_threads) {

  const int_t END = 1UL << num_qubits;
  const uint_t SHOTS = rnds.size();
  const uint_t PARTITION_SIZE = num_qubits < 15 ? 0: 10;
  const int_t PARTITION_END = 1UL << PARTITION_SIZE;
  const uint_t LOOP = (END >> PARTITION_SIZE);

  reg_t samples;
  samples.assign(SHOTS, 0);

  std::vector<std::vector<double>> acc_probs_list;
  std::vector<reg_t> acc_idxs_list;
  std::vector<double> start_probs;
  std::vector<double> end_probs;

  acc_probs_list.assign(PARTITION_END, std::vector<double>());
  acc_idxs_list.assign(PARTITION_END, reg_t());
  start_probs.assign(PARTITION_END, 0.);
  end_probs.assign(PARTITION_END, 0.);

  // generate prefix-sum vector for each partition
  auto prefix_sum = [&](const int_t i)->void {
    double accumulated = .0;
    for (uint_t j = LOOP * i; j < LOOP * (i + 1); j++) {
      auto norm = probability_func(j);
      if (!AER::Linalg::almost_equal(norm, 0.0)) {
        accumulated += norm;
        acc_probs_list[i].push_back(accumulated);
        acc_idxs_list[i].push_back(j);
      }
    }
    end_probs[i] = accumulated;
  };
  QV::apply_lambda(0, PARTITION_END, omp_threads, prefix_sum);

  for (int_t i = 1; i < PARTITION_END; ++i)
    start_probs[i] = end_probs[i -1] + start_probs[i - 1];

  // sampling
  auto sampling = [&](const int_t i)->void {
    double rnd = rnds[i];
    // binary search for partition
    int_t partition_idx = scan_inclusive(start_probs, rnd, 0, PARTITION_END);
    if (partition_idx == PARTITION_END)
      partition_idx = PARTITION_END - 1;

    rnd -= start_probs[partition_idx];
    // binary search for which range rnd is in
    int_t sample_idx = scan_inclusive(acc_probs_list[partition_idx], rnd, 0, acc_probs_list[partition_idx].size());
    if (sample_idx == acc_probs_list[partition_idx].size())
      sample_idx = acc_probs_list[partition_idx].size() - 1;

    samples[i] = acc_idxs_list[partition_idx][sample_idx];
  };
  QV::apply_lambda(0, SHOTS, omp_threads, sampling);

  return samples;
}

}
}

#endif
