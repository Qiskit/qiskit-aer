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

#ifndef _clifford_hpp_
#define _clifford_hpp_

#include "framework/types.hpp"
#include "framework/utils.hpp"

#include "framework/json_parser.hpp"
#include "pauli.hpp"

#include <omp.h>

namespace AER {
namespace Clifford {

/*******************************************************************************
 *
 * Clifford Class
 *
 ******************************************************************************/

class Clifford {
public:
  using phase_t = int8_t;
  using phasevec_t = std::vector<phase_t>;

  friend class CliffordThrust;

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------
  Clifford() = default;
  explicit Clifford(const uint64_t nqubit);

  // initialize from existing state (copy)
  void initialize(const Clifford &obj);

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------
  // initialize
  void initialize(const uint64_t nqubit);

  // Get number of qubits of the Clifford table
  uint64_t num_qubits() const { return num_qubits_; }

  // Return true if the number of qubits is 0
  bool empty() const { return (num_qubits_ == 0); }

  // Return JSON serialization of QubitVector;
  json_t json() const;

  // Access stabilizer table
  //  Pauli::Pauli<BV::BinaryVector> &operator[](uint64_t j) {return table_[j];}
  //  const Pauli::Pauli<BV::BinaryVector>& operator[](uint64_t j) const {return
  //  table_[j];}

  // set stabilizer
  void set_destabilizer(const int i, const Pauli::Pauli<BV::BinaryVector> &P);
  void set_stabilizer(const int i, const Pauli::Pauli<BV::BinaryVector> &P);

  // set phase
  void set_destabilizer_phases(const int i, const bool p);
  void set_stabilizer_phases(const int i, const bool p);

  // Set the state of the simulator to a given Clifford
  void apply_set_stabilizer(const Clifford &clifford);

  //-----------------------------------------------------------------------
  // Apply basic Clifford gates
  //-----------------------------------------------------------------------

  // Apply Controlled-NOT (CX) gate
  void append_cx(const uint64_t qubit_ctrl, const uint64_t qubit_trgt);

  // Apply Hadamard (H) gate
  void append_h(const uint64_t qubit);

  // Apply Phase (S, square root of Z) gate
  void append_s(const uint64_t qubit);

  // Apply Pauli::Pauli<BV::BinaryVector> X gate
  void append_x(const uint64_t qubit);

  // Apply Pauli::Pauli<BV::BinaryVector> Y gate
  void append_y(const uint64_t qubit);

  // Apply Pauli::Pauli<BV::BinaryVector> Z gate
  void append_z(const uint64_t qubit);

  //-----------------------------------------------------------------------
  // Measurement
  //-----------------------------------------------------------------------

  // If we perform a single qubit Z measurement,
  // will the outcome be random or deterministic.
  bool is_deterministic_outcome(const uint64_t &qubit) const;

  // Return the outcome (0 or 1) of a single qubit Z measurement, and
  // update the stabilizer to the conditional (post measurement) state if
  // the outcome was random.
  bool measure_and_update(const uint64_t qubit, const uint64_t randint);

  double expval_pauli(const reg_t &qubits, const std::string &pauli);

  //-----------------------------------------------------------------------
  // Configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() { return json_chop_threshold_; }

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint64_t get_omp_threads() { return omp_threads_; }

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint64_t get_omp_threshold() { return omp_threshold_; }

protected:
  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  std::vector<Pauli::Pauli<BV::BinaryVector>> destabilizer_table_;
  std::vector<Pauli::Pauli<BV::BinaryVector>> stabilizer_table_;
  BV::BinaryVector destabilizer_phases_;
  BV::BinaryVector stabilizer_phases_;
  uint64_t num_qubits_ = 0;

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  uint64_t omp_threads_ = 1; // Disable multithreading by default
  uint64_t omp_threshold_ =
      1000; // Qubit threshold for multithreading when enabled
  double json_chop_threshold_ = 0; // Threshold for chopping small values
                                   // in JSON serialization

  //-----------------------------------------------------------------------
  // Helper functions
  //-----------------------------------------------------------------------

  // Check if there exists stabilizer or destabilizer row anticommuting
  // with Z[qubit]. If so return pair (true, row), else return (false, 0)
  std::pair<bool, uint64_t> z_anticommuting(const uint64_t qubit) const;

  // Check if there exists stabilizer or destabilizer row anticommuting
  // with X[qubit]. If so return pair (true, row), else return (false, 0)
  std::pair<bool, uint64_t> x_anticommuting(const uint64_t qubit) const;
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Config settings
//------------------------------------------------------------------------------

void Clifford::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}

void Clifford::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

void Clifford::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

Clifford::Clifford(uint64_t nq) : num_qubits_(nq) { initialize(nq); }

void Clifford::initialize(uint64_t nq) {
  num_qubits_ = nq;

  destabilizer_table_.resize(nq);
  stabilizer_table_.resize(nq);

  int nid = omp_get_num_threads();
  auto init_func = [this, nq](AER::int_t i) {
    destabilizer_table_[i].X.setLength(nq);
    destabilizer_table_[i].Z.setLength(nq);
    destabilizer_table_[i].X.setValue(1, i);

    stabilizer_table_[i].X.setLength(nq);
    stabilizer_table_[i].Z.setLength(nq);
    stabilizer_table_[i].Z.setValue(1, i);
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0, nq,
      init_func, omp_threads_);

  // Add phases
  destabilizer_phases_.setLength(nq);
  stabilizer_phases_.setLength(nq);
}

void Clifford::initialize(const Clifford &obj) {
  destabilizer_table_ = obj.destabilizer_table_;
  stabilizer_table_ = obj.stabilizer_table_;
  destabilizer_phases_ = obj.destabilizer_phases_;
  stabilizer_phases_ = obj.stabilizer_phases_;
  num_qubits_ = obj.num_qubits_;
  omp_threads_ = obj.omp_threads_;
  omp_threshold_ = obj.omp_threshold_;
  json_chop_threshold_ = obj.json_chop_threshold_;
}

//------------------------------------------------------------------------------
// Apply Clifford gates
//------------------------------------------------------------------------------

void Clifford::append_cx(const uint64_t qcon, const uint64_t qtar) {
  const uint64_t mask = (~0ull);

  int nid = omp_get_num_threads();
  auto cx_func = [this, qtar, qcon, mask](AER::int_t i) {
    destabilizer_phases_(i) =
        destabilizer_phases_(i) ^
        (destabilizer_table_[qcon].X(i) & destabilizer_table_[qtar].Z(i) &
         (destabilizer_table_[qtar].X(i) ^ destabilizer_table_[qcon].Z(i) ^
          mask));
    stabilizer_phases_(i) =
        stabilizer_phases_(i) ^
        (stabilizer_table_[qcon].X(i) & stabilizer_table_[qtar].Z(i) &
         (stabilizer_table_[qtar].X(i) ^ stabilizer_table_[qcon].Z(i) ^ mask));

    destabilizer_table_[qtar].X(i) =
        destabilizer_table_[qtar].X(i) ^ destabilizer_table_[qcon].X(i);
    destabilizer_table_[qcon].Z(i) =
        destabilizer_table_[qtar].Z(i) ^ destabilizer_table_[qcon].Z(i);
    stabilizer_table_[qtar].X(i) =
        stabilizer_table_[qtar].X(i) ^ stabilizer_table_[qcon].X(i);
    stabilizer_table_[qcon].Z(i) =
        stabilizer_table_[qtar].Z(i) ^ stabilizer_table_[qcon].Z(i);
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
      destabilizer_phases_.blockLength(), cx_func, omp_threads_);
}

void Clifford::append_h(const uint64_t qubit) {
  int nid = omp_get_num_threads();
  auto h_func = [this, qubit](AER::int_t i) {
    destabilizer_phases_(i) ^=
        (destabilizer_table_[qubit].X(i) & destabilizer_table_[qubit].Z(i));
    stabilizer_phases_(i) ^=
        (stabilizer_table_[qubit].X(i) & stabilizer_table_[qubit].Z(i));
    // exchange X and Z
    uint64_t t = destabilizer_table_[qubit].X(i);
    destabilizer_table_[qubit].X(i) = destabilizer_table_[qubit].Z(i);
    destabilizer_table_[qubit].Z(i) = t;
    t = stabilizer_table_[qubit].X(i);
    stabilizer_table_[qubit].X(i) = stabilizer_table_[qubit].Z(i);
    stabilizer_table_[qubit].Z(i) = t;
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
      destabilizer_phases_.blockLength(), h_func, omp_threads_);
}

void Clifford::append_s(const uint64_t qubit) {
  int nid = omp_get_num_threads();
  auto s_func = [this, qubit](AER::int_t i) {
    destabilizer_phases_(i) ^=
        (destabilizer_table_[qubit].X(i) & destabilizer_table_[qubit].Z(i));
    destabilizer_table_[qubit].Z(i) ^= destabilizer_table_[qubit].X(i);
    stabilizer_phases_(i) ^=
        (stabilizer_table_[qubit].X(i) & stabilizer_table_[qubit].Z(i));
    stabilizer_table_[qubit].Z(i) ^= stabilizer_table_[qubit].X(i);
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
      destabilizer_phases_.blockLength(), s_func, omp_threads_);
}

void Clifford::append_x(const uint64_t qubit) {
  int nid = omp_get_num_threads();
  auto x_func = [this, qubit](AER::int_t i) {
    destabilizer_phases_(i) ^= destabilizer_table_[qubit].Z(i);
    stabilizer_phases_(i) ^= stabilizer_table_[qubit].Z(i);
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
      destabilizer_phases_.blockLength(), x_func, omp_threads_);
}

void Clifford::append_y(const uint64_t qubit) {
  int nid = omp_get_num_threads();
  auto y_func = [this, qubit](AER::int_t i) {
    destabilizer_phases_(i) ^=
        (destabilizer_table_[qubit].Z(i) ^ destabilizer_table_[qubit].X(i));
    stabilizer_phases_(i) ^=
        (stabilizer_table_[qubit].Z(i) ^ stabilizer_table_[qubit].X(i));
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
      destabilizer_phases_.blockLength(), y_func, omp_threads_);
}

void Clifford::append_z(const uint64_t qubit) {
  int nid = omp_get_num_threads();
  auto z_func = [this, qubit](AER::int_t i) {
    destabilizer_phases_(i) ^= destabilizer_table_[qubit].X(i);
    stabilizer_phases_(i) ^= stabilizer_table_[qubit].X(i);
  };
  AER::Utils::apply_omp_parallel_for(
      (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
      destabilizer_phases_.blockLength(), z_func, omp_threads_);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
std::pair<bool, uint64_t>
Clifford::z_anticommuting(const uint64_t qubit) const {
  for (uint_t i = 0; i < stabilizer_table_[qubit].X.blockLength(); i++) {
    if (stabilizer_table_[qubit].X(i) != 0) {
      uint_t p = i << stabilizer_table_[qubit].X.BLOCK_BITS;
      for (uint_t j = 0; j < stabilizer_table_[qubit].X.BLOCK_SIZE; j++) {
        if (stabilizer_table_[qubit].X[p + j])
          return std::make_pair(true, p + j);
      }
    }
  }
  return std::make_pair(false, 0);
}

std::pair<bool, uint64_t>
Clifford::x_anticommuting(const uint64_t qubit) const {
  for (uint_t i = 0; i < stabilizer_table_[qubit].Z.blockLength(); i++) {
    if (stabilizer_table_[qubit].Z(i) != 0) {
      uint_t p = i << stabilizer_table_[qubit].Z.BLOCK_BITS;
      for (uint_t j = 0; j < stabilizer_table_[qubit].Z.BLOCK_SIZE; j++) {
        if (stabilizer_table_[qubit].Z[p + j])
          return std::make_pair(true, p + j);
      }
    }
  }
  return std::make_pair(false, 0);
}

void Clifford::set_destabilizer(const int idx,
                                const Pauli::Pauli<BV::BinaryVector> &P) {
  for (int64_t i = 0; i < static_cast<int64_t>(num_qubits_); i++) {
    destabilizer_table_[i].X.setValue(P.X[i], idx);
    destabilizer_table_[i].Z.setValue(P.Z[i], idx);
  }
}

void Clifford::set_stabilizer(const int idx,
                              const Pauli::Pauli<BV::BinaryVector> &P) {
  for (int64_t i = 0; i < static_cast<int64_t>(num_qubits_); i++) {
    stabilizer_table_[i].X.setValue(P.X[i], idx);
    stabilizer_table_[i].Z.setValue(P.Z[i], idx);
  }
}

void Clifford::set_destabilizer_phases(const int i, const bool p) {
  destabilizer_phases_.setValue(p, i);
}

void Clifford::set_stabilizer_phases(const int i, const bool p) {
  stabilizer_phases_.setValue(p, i);
}

void Clifford::apply_set_stabilizer(const Clifford &clifford) {
  destabilizer_table_ = clifford.destabilizer_table_;
  stabilizer_table_ = clifford.stabilizer_table_;
  destabilizer_phases_ = clifford.destabilizer_phases_;
  stabilizer_phases_ = clifford.stabilizer_phases_;
}

//------------------------------------------------------------------------------
// Measurement
//------------------------------------------------------------------------------

bool Clifford::is_deterministic_outcome(const uint64_t &qubit) const {
  // Clifford state measurements only have three probabilities:
  // (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
  // The random case happens if there is a row anti-commuting with Z[qubit]
  return !z_anticommuting(qubit).first;
}

bool Clifford::measure_and_update(const uint64_t qubit,
                                  const uint64_t randint) {
  // Clifford state measurements only have three probabilities:
  // (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
  // The random case happens if there is a row anti-commuting with Z[qubit]
  auto anticom = z_anticommuting(qubit);

  int nid = omp_get_num_threads();
  if (anticom.first) {
    bool outcome = (randint == 1);
    auto row = anticom.second;

    uint64_t rS = 0ull - (uint64_t)stabilizer_phases_[row];

    auto measure_non_determinisitic_func = [this, rS, row,
                                            qubit](AER::int_t i) {
      uint64_t row_mask = ~0ull;
      if ((row >> destabilizer_phases_.BLOCK_BITS) == (uint_t)i)
        row_mask ^= (1ull << (row & destabilizer_phases_.BLOCK_MASK));

      uint64_t d_mask = row_mask & destabilizer_table_[qubit].X(i);
      uint64_t s_mask = row_mask & stabilizer_table_[qubit].X(i);

      if (d_mask != 0 || s_mask != 0) {
        // calculating exponents by 2-bits integer * 64-qubits at once
        uint64_t d0 = 0, d1 = 0;
        uint64_t s0 = 0, s1 = 0;
        for (size_t q = 0; q < num_qubits_; q++) {
          uint64_t t0, t1;
          uint64_t rX = 0ull - (uint64_t)stabilizer_table_[q].X[row];
          uint64_t rZ = 0ull - (uint64_t)stabilizer_table_[q].Z[row];

          // destabilizer
          t0 = destabilizer_table_[q].X(i) & rZ;
          t1 = destabilizer_table_[q].Z(i) ^ rX;

          d1 ^= (t0 & d0);
          d0 ^= t0;
          d1 ^= (t0 & t1);

          t0 = rX & destabilizer_table_[q].Z(i);
          t1 = rZ ^ destabilizer_table_[q].X(i);
          t1 ^= t0;

          d1 ^= (t0 & d0);
          d0 ^= t0;
          d1 ^= (t0 & t1);

          destabilizer_table_[q].X(i) ^= (d_mask & rX);
          destabilizer_table_[q].Z(i) ^= (d_mask & rZ);

          // stabilizer
          t0 = stabilizer_table_[q].X(i) & rZ;
          t1 = stabilizer_table_[q].Z(i) ^ rX;

          s1 ^= (t0 & s0);
          s0 ^= t0;
          s1 ^= (t0 & t1);

          t0 = rX & stabilizer_table_[q].Z(i);
          t1 = rZ ^ stabilizer_table_[q].X(i);
          t1 ^= t0;

          s1 ^= (t0 & s0);
          s0 ^= t0;
          s1 ^= (t0 & t1);

          stabilizer_table_[q].X(i) ^= (s_mask & rX);
          stabilizer_table_[q].Z(i) ^= (s_mask & rZ);
        }
        d1 ^= (rS ^ destabilizer_phases_(i));
        destabilizer_phases_(i) =
            (destabilizer_phases_(i) & (~d_mask)) | (d1 & d_mask);
        s1 ^= (rS ^ stabilizer_phases_(i));
        stabilizer_phases_(i) =
            (stabilizer_phases_(i) & (~s_mask)) | (s1 & s_mask);
      }
    };
    AER::Utils::apply_omp_parallel_for(
        (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
        destabilizer_phases_.blockLength(), measure_non_determinisitic_func,
        omp_threads_);

    // Update state
    auto measure_update_func = [this, row](AER::int_t q) {
      destabilizer_table_[q].X.setValue(stabilizer_table_[q].X[row], row);
      destabilizer_table_[q].Z.setValue(stabilizer_table_[q].Z[row], row);
      stabilizer_table_[q].X.setValue(0, row);
      stabilizer_table_[q].Z.setValue(0, row);
    };
    AER::Utils::apply_omp_parallel_for(
        (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
        num_qubits_, measure_update_func, omp_threads_);

    destabilizer_phases_.setValue(stabilizer_phases_[row], row);
    stabilizer_table_[qubit].Z.setValue(1, row);
    stabilizer_phases_.setValue(outcome, row);
    return outcome;
  } else {
    // Deterministic outcome
    uint_t outcome = 0;
    Pauli::Pauli<BV::BinaryVector> accum(num_qubits_);
    uint_t blocks = destabilizer_phases_.blockLength();

    if (blocks < 2) {
      for (uint_t ib = 0; ib < blocks; ib++) {
        uint_t destabilizer_mask = destabilizer_table_[qubit].X(ib);
        uint_t exponent_l = 0ull;
        uint_t exponent_h = 0ull;

        for (uint_t q = 0; q < num_qubits_; q++) {
          uint_t tl, th, add;
          uint_t accumX = 0ull - (uint_t)accum.X[q];
          uint_t accumZ = 0ull - (uint_t)accum.Z[q];

          tl = accumX & stabilizer_table_[q].Z(ib);
          th = accumZ ^ stabilizer_table_[q].X(ib);

          add = tl & exponent_l;
          exponent_l ^= tl;
          exponent_h ^= add;
          exponent_h ^= (tl & th);

          tl = stabilizer_table_[q].X(ib) & accumZ;
          th = stabilizer_table_[q].Z(ib) ^ accumX;
          th ^= tl;

          add = tl & exponent_l;
          exponent_l ^= tl;
          exponent_h ^= add;
          exponent_h ^= (tl & th);

          add = stabilizer_table_[q].X(ib) & destabilizer_mask;
          accumX &= AER::Utils::popcount(add) & 1;
          add = stabilizer_table_[q].Z(ib) & destabilizer_mask;
          accumZ &= AER::Utils::popcount(add) & 1;

          accum.X.setValue((bool)accumX, q);
          accum.Z.setValue((bool)accumZ, q);
        }
        exponent_h ^= stabilizer_phases_(ib);
        outcome ^= (exponent_h & destabilizer_mask);

        if ((exponent_l & destabilizer_mask) != 0) {
          throw std::runtime_error("Clifford: rowsum error");
        }
      }
    } else {
      uint_t blockSize = destabilizer_phases_.blockSize();

      // loop for cache blocking
      for (uint_t ii = 0; ii < blocks; ii++) {
        uint_t destabilizer_mask = destabilizer_table_[qubit].X(ii);
        if (destabilizer_mask == 0)
          continue;

        uint_t exponent_l = 0;
        uint_t exponent_lc = 0;
        uint_t exponent_h = 0;

        auto measure_determinisitic_func =
            [this, &accum, &exponent_l, &exponent_lc, &exponent_h, blocks,
             blockSize, destabilizer_mask, ii](AER::int_t qq) {
              uint_t qs = qq * blockSize;
              uint_t qe = qs + blockSize;
              if (qe > num_qubits_)
                qe = num_qubits_;

              uint_t local_exponent_l = 0;
              uint_t local_exponent_h = 0;

              for (uint_t q = qs; q < qe; q++) {
                uint_t sX = stabilizer_table_[q].X(ii);
                uint_t sZ = stabilizer_table_[q].Z(ii);

                uint_t accumX = (0ull - (uint_t)accum.X[q]);
                uint_t accumZ = (0ull - (uint_t)accum.Z[q]);

                // exponents for this block
                uint_t t0, t1;

                t0 = accumX & sZ;
                t1 = accumZ ^ sX;

                local_exponent_h ^= (t0 & local_exponent_l);
                local_exponent_l ^= t0;
                local_exponent_h ^= (t0 & t1);

                t0 = sX & accumZ;
                t1 = sZ ^ accumX;
                t1 ^= t0;

                local_exponent_h ^= (t0 & local_exponent_l);
                local_exponent_l ^= t0;
                local_exponent_h ^= (t0 & t1);

                // update accum
                accumX &= AER::Utils::popcount(sX & destabilizer_mask) & 1;
                accum.X.setValue((accumX != 0), q);
                accumZ &= AER::Utils::popcount(sZ & destabilizer_mask) & 1;
                accum.Z.setValue((accumZ != 0), q);
              }

#pragma omp atomic
              exponent_lc |= local_exponent_l;
#pragma omp atomic
              exponent_l ^= local_exponent_l;
#pragma omp atomic
              exponent_h ^= local_exponent_h;
            };
        AER::Utils::apply_omp_parallel_for(
            (num_qubits_ > omp_threshold_ && omp_threads_ > 1 && nid == 1), 0,
            blocks, measure_determinisitic_func, omp_threads_);

        // if exponent_l is 0 and any of local_exponent_l is
        // 1, then flip exponent_h
        exponent_h ^= (exponent_lc ^ exponent_l);
        exponent_h ^= stabilizer_phases_(ii);
        outcome ^= (exponent_h & destabilizer_mask);
      }
    }
    return ((AER::Utils::popcount(outcome) & 1) != 0);
  }
}

double Clifford::expval_pauli(const reg_t &qubits, const std::string &pauli) {
  // Construct Pauli on N-qubits
  Pauli::Pauli<BV::BinaryVector> P(num_qubits_);
  uint_t phase = 0;
  for (size_t i = 0; i < qubits.size(); ++i) {
    switch (pauli[pauli.size() - 1 - i]) {
    case 'X':
      P.X.set1(qubits[i]);
      break;
    case 'Y':
      P.X.set1(qubits[i]);
      P.Z.set1(qubits[i]);
      phase += 1;
      break;
    case 'Z':
      P.Z.set1(qubits[i]);
      break;
    default:
      break;
    };
  }

  // Check if there is a stabilizer that anti-commutes with an odd number of
  // qubits If so expectation value is 0
  for (size_t i = 0; i < num_qubits_; i++) {
    size_t num_anti = 0;
    for (const auto &qubit : qubits) {
      if (P.Z[qubit] & stabilizer_table_[qubit].X[i]) {
        num_anti++;
      }
      if (P.X[qubit] & stabilizer_table_[qubit].Z[i]) {
        num_anti++;
      }
    }
    if (num_anti % 2 == 1)
      return 0.0;
  }

  // Otherwise P is (-1)^a prod_j S_j^b_j for Clifford stabilizers
  // If P anti-commutes with D_j then b_j = 1.
  // Multiply P by stabilizers with anti-commuting destabilizers
  auto PZ = P.Z; // Make a copy of P.Z
  for (size_t i = 0; i < num_qubits_; i++) {
    // Check if destabilizer anti-commutes
    size_t num_anti = 0;
    for (const auto &qubit : qubits) {
      if (P.Z[qubit] & destabilizer_table_[qubit].X[i]) {
        num_anti++;
      }
      if (P.X[qubit] & destabilizer_table_[qubit].Z[i]) {
        num_anti++;
      }
    }
    if (num_anti % 2 == 0)
      continue;

    // If anti-commutes multiply Pauli by stabilizer
    phase += 2 * (uint_t)stabilizer_phases_[i];
    for (size_t k = 0; k < num_qubits_; k++) {
      phase += stabilizer_table_[k].Z[i] & stabilizer_table_[k].X[i];
      phase += 2 * (PZ[k] & stabilizer_table_[k].X[i]);
      PZ.setValue(PZ[k] ^ stabilizer_table_[k].Z[i], k);
    }
  }
  return (phase % 4) ? -1.0 : 1.0;
}

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

json_t Clifford::json() const {
  json_t js = json_t::object();
  // Add destabilizers
  json_t stab;
  for (size_t i = 0; i < num_qubits_; i++) {
    // Destabilizer
    std::string label = (destabilizer_phases_[i] == 0) ? "+" : "-";

    Pauli::Pauli<BV::BinaryVector> P(num_qubits_);
    for (size_t j = 0; j < num_qubits_; j++) {
      P.X.setValue(destabilizer_table_[j].X[i], j);
      P.Z.setValue(destabilizer_table_[j].Z[i], j);
    }
    label += P.str();
    js["destabilizer"].push_back(label);

    // Stabilizer
    label = (stabilizer_phases_[i] == 0) ? "+" : "-";
    for (size_t j = 0; j < num_qubits_; j++) {
      P.X.setValue(stabilizer_table_[j].X[i], j);
      P.Z.setValue(stabilizer_table_[j].Z[i], j);
    }
    label += P.str();
    js["stabilizer"].push_back(label);
  }
  return js;
}

inline void to_json(json_t &js, const Clifford &clif) { js = clif.json(); }

template <typename inputdata_t>
void build_from(const inputdata_t &input, Clifford &clif) {
  bool has_keys = AER::Parser<inputdata_t>::check_keys(
      {"stabilizer", "destabilizer"}, input);
  if (!has_keys)
    throw std::invalid_argument("Invalid Clifford JSON.");

  std::vector<std::string> stab, destab;
  AER::Parser<inputdata_t>::get_value(stab, "stabilizer", input);
  AER::Parser<inputdata_t>::get_value(destab, "destabilizer", input);

  const auto nq = stab.size();
  if (nq != destab.size()) {
    throw std::invalid_argument("Invalid Clifford JSON: stabilizer and "
                                "destabilizer lengths do not match.");
  }

  clif.initialize(nq);
  for (size_t i = 0; i < nq; i++) {
    std::string label;
    // Get destabilizer
    label = destab[i];
    switch (label[0]) {
    case '-':
      clif.set_destabilizer_phases(i, 1);
      clif.set_destabilizer(
          i, Pauli::Pauli<BV::BinaryVector>(label.substr(1, nq)));
      break;
    case '+':
      clif.set_destabilizer(
          i, Pauli::Pauli<BV::BinaryVector>(label.substr(1, nq)));
      break;
    case 'I':
    case 'X':
    case 'Y':
    case 'Z':
      clif.set_destabilizer(i, Pauli::Pauli<BV::BinaryVector>(label));
      break;
    default:
      throw std::invalid_argument("Invalid Stabilizer JSON string.");
    }
    // Get stabilizer
    label = stab[i];
    switch (label[0]) {
    case '-':
      clif.set_stabilizer_phases(i, 1);
      clif.set_stabilizer(i,
                          Pauli::Pauli<BV::BinaryVector>(label.substr(1, nq)));
      break;
    case '+':
      clif.set_stabilizer(i,
                          Pauli::Pauli<BV::BinaryVector>(label.substr(1, nq)));
      break;
    case 'I':
    case 'X':
    case 'Y':
    case 'Z':
      clif.set_stabilizer(i, Pauli::Pauli<BV::BinaryVector>(label));
      break;
    default:
      throw std::invalid_argument("Invalid Stabilizer JSON string.");
    }
  }
}

inline void from_json(const json_t &js, Clifford &clif) {
  build_from(js, clif);
}

//------------------------------------------------------------------------------
} // end namespace Clifford
} // namespace AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class statevector_t>
std::ostream &operator<<(std::ostream &out,
                         const AER::Clifford::Clifford &clif) {
  out << clif.json().dump();
  return out;
}

//------------------------------------------------------------------------------
#endif
