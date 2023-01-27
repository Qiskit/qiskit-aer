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
#include <math.h>
#include <framework/stl_ostream.hpp>
#include <map>
#include <type_traits>

#include <framework/linalg/linalg.hpp>
#include <framework/types.hpp>
#include <framework/utils.hpp>


#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "simulators/statevector/qubitvector.hpp"
#include "utils.hpp"

using namespace AER::Test::Utilities;
using namespace AER::QV;

TEST_CASE("AER::QV::QubitVector") {

    SECTION("QubitVector()") {
        QubitVector<double> qv;
        REQUIRE(compare<int>(1, qv.size()));
        REQUIRE(compare<std::string>("statevector", qv.name()));
    }

    SECTION("QubitVector(size_t num_qubits)") {
        for (auto qubit = 0; qubit < 10; ++qubit) {
            QubitVector<double> qv(qubit);
            REQUIRE(compare<int>(pow(2, qubit), qv.size()));
        }
    }

    SECTION("QubitVector(size_t num_qubits, std::complex<data_t>* data, bool copy=false)") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = 0;
        {
            QubitVector<double> qv(4, data, false);
            for (uint_t i = 0; i < 16; ++i)
                REQUIRE(compare<std::complex<double>>(0, qv[i]));
            data[0] = 1;
            REQUIRE(compare<std::complex<double>>(1, qv[0]));
            data[0] = 0;
        }
        {
            QubitVector<double> qv(4, data, true);
            data[0] = 1;
            REQUIRE(compare<std::complex<double>>(0, qv[0]));
            data[0] = 0;
        }
    }

    SECTION("QubitVector(const QubitVector& obj)") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);
        auto copy = qv;

        for (uint_t i = 0; i < 16; ++i)
          data[i] = 0;

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(0, qv[i]));

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, copy[i]));
    }

    SECTION("QubitVector &operator=(QubitVector&& obj)") {
        QubitVector<double> qvlist[1];

        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);
        qvlist[0] = std::move(qv);

        for (uint_t i = 0; i < 16; ++i)
          data[i] *= 2;

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i*2, qvlist[0][i]));
    }

    SECTION("std::complex<data_t> &operator[](uint_t element)") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);

        for (uint_t i = 0; i < 16; ++i)
          qv[i] = i * 2;

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i*2, qv[i]));
    }

    SECTION("std::complex<data_t> operator[](uint_t element) const") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, qv[i]));
    }

    SECTION("void set_state(uint_t pos, std::complex<data_t>& val)") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);
        std::complex<double> val = 0;
        qv.set_state(15, val);

        for (uint_t i = 0; i < 15; ++i)
            REQUIRE(compare<std::complex<double>>(i, qv[i]));
        REQUIRE(compare<std::complex<double>>(0, qv[15]));
    }

    SECTION("std::complex<data_t> get_state(uint_t pos) const") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, qv.get_state(i)));
    }

    SECTION("std::complex<data_t>* &data()") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);
        std::complex<double>* &ref = qv.data();

        REQUIRE(ref == data);
    }

    SECTION("void set_num_qubits(size_t num_qubits)") {
        std::complex<double> data[16];
        for (uint_t i = 0; i < 16; ++i)
          data[i] = i;

        QubitVector<double> qv(4, data, false);

        qv.set_num_qubits(5);

        for (uint_t i = 0; i < 32; ++i)
          qv[i] = i;

        for (uint_t i = 0; i < 16; ++i)
          data[i] *= 2;

        for (uint_t i = 0; i < 32; ++i)
            REQUIRE(compare<std::complex<double>>(i, qv[i]));
    }

    SECTION("uint_t num_qubits() const") {
        std::complex<double> data[16];
        QubitVector<double> qv(4, data, false);
        REQUIRE(compare<uint_t>(4, qv.num_qubits()));
        qv.set_num_qubits(5);
        REQUIRE(compare<uint_t>(5, qv.num_qubits()));
    }

    SECTION("uint_t size() const") {
        std::complex<double> data[16];
        QubitVector<double> qv(4, data, false);
        REQUIRE(compare<uint_t>(16, qv.size()));
        qv.set_num_qubits(5);
        REQUIRE(compare<uint_t>(32, qv.size()));
    }

    SECTION("uint_t required_memory_mb() const") {
        QubitVector<double> qv_double;

        REQUIRE(compare<uint_t>(8 /*double*/ * 2 /*complex*/ * pow(2, 30 - 20) /*2^30/mb*/, qv_double.required_memory_mb(30)));
        REQUIRE(compare<uint_t>(8 /*double*/ * 2 /*complex*/ * pow(2, 40 - 20) /*2^30/mb*/, qv_double.required_memory_mb(40)));

        QubitVector<float> qv_single;
        REQUIRE(compare<uint_t>(4 /*float*/ * 2 /*complex*/ * pow(2, 30 - 20) /*2^30/mb*/, qv_single.required_memory_mb(30)));
        REQUIRE(compare<uint_t>(4 /*float*/ * 2 /*complex*/ * pow(2, 40 - 20) /*2^30/mb*/, qv_single.required_memory_mb(40)));
    }

    SECTION("cvector_t<data_t> vector() const") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;
        auto vec = qv.vector();
        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, vec[i]));
    }

    SECTION("cdict_t<data_t> vector_ket(double epsilon = 0) const") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;
        auto ket = qv.vector_ket(8);

        for (uint_t i = 0; i < 9; ++i) {
            std::string key = AER::Utils::int2hex(i);
            REQUIRE(compare<std::complex<double>>(0, ket[key]));
        }
        for (uint_t i = 9; i < 16; ++i) {
            std::string key = AER::Utils::int2hex(i);
            REQUIRE(compare<std::complex<double>>(i, ket[key]));
        }
    }

    SECTION("AER::Vector<std::complex<data_t>> copy_to_vector() const") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;
        auto data = qv.copy_to_vector();

        REQUIRE(compare<uint_t>(16, data.size()));

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, data[i]));

        REQUIRE(compare<uint_t>(4, qv.num_qubits()));

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, qv[i]));

    }

    SECTION("AER::Vector<std::complex<data_t>> move_to_vector() const") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;
        auto data = qv.move_to_vector();

        REQUIRE(compare<uint_t>(16, data.size()));

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i, data[i]));

        REQUIRE(compare<uint_t>(0, qv.num_qubits()));

    }

    SECTION("json_t json() const") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        auto json = qv.json();

        REQUIRE(compare<uint_t>(16, json.size()));
        for (uint_t i = 0; i < 16; ++i) {
            REQUIRE(compare<uint_t>(2, json[i].size()));
            REQUIRE(compare<double>(i, json[i][0]));
            REQUIRE(compare<double>(0, json[i][1]));
        }
    }

    SECTION("void zero()") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        qv.zero();

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(0, qv[i]));
    }

    SECTION("void initialize()") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        qv.initialize();

        REQUIRE(compare<std::complex<double>>(1, qv[0]));
        for (uint_t i = 1; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(0, qv[i]));
    }

    SECTION("void initialize_from_vector(const list_t &vec)") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        std::vector<std::complex<double>> vec;
        for (uint_t i = 0; i < 16; ++i)
            vec.push_back(i * 2);

        qv.initialize_from_vector(vec);

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(vec[i], qv[i]));
    }

    SECTION("void initialize_from_vector(std::vector<std::complex<data_t>> &&vec)") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        std::vector<std::complex<double>> vec;
        for (uint_t i = 0; i < 16; ++i)
            vec.push_back(i * 2);

        qv.initialize_from_vector(std::move(vec));

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i * 2, qv[i]));
    }

    SECTION("void initialize_from_vector(AER::Vector<std::complex<data_t>> &&vec)") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        AER::Vector<std::complex<double>> vec(16);
        for (uint_t i = 0; i < 16; ++i)
            vec[i] = i * 2;

        qv.initialize_from_vector(std::move(vec));

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i * 2, qv[i]));
    }

    SECTION("void initialize_from_data(const std::complex<data_t>* data, const size_t num_states)") {
        QubitVector<double> qv;
        qv.set_num_qubits(4);
        for (uint_t i = 0; i < 16; ++i)
            qv[i] = i;

        std::complex<double> vec[16];
        for (uint_t i = 0; i < 16; ++i)
            vec[i] = i * 2;

        qv.initialize_from_data(vec, 16);

        for (uint_t i = 0; i < 16; ++i)
            REQUIRE(compare<std::complex<double>>(i * 2, qv[i]));
    }

//   void apply_matrix(const reg_t &qubits, const cvector_t<double> &mat);
//   void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cvector_t<double> &mat);
//   void apply_diagonal_matrix(const reg_t &qubits, const cvector_t<double> &mat);
//   void apply_permutation_matrix(const reg_t &qubits,
//                                 const std::vector<std::pair<uint_t, uint_t>> &pairs);
//   void apply_mcx(const reg_t &qubits);
//   void apply_mcy(const reg_t &qubits);
//   void apply_mcphase(const reg_t &qubits, const std::complex<double> phase);
//   void apply_mcu(const reg_t &qubits, const cvector_t<double> &mat);
//   void apply_mcswap(const reg_t &qubits);
//   void apply_multi_swaps(const reg_t &qubits);
//   void apply_rotation(const reg_t &qubits, const Rotation r, const double theta);
//   void apply_pauli(const reg_t &qubits, const std::string &pauli,
//                    const complex_t &coeff = 1);
//   virtual double probability(const uint_t outcome) const;
//   virtual std::vector<double> probabilities() const;
//   virtual std::vector<double> probabilities(const reg_t &qubits) const;
//   virtual reg_t sample_measure(const std::vector<double> &rnds) const;
//   virtual void apply_bfunc(const Operations::Op &op){}
//   virtual void set_conditional(int_t reg){}
//   virtual void apply_roerror(const Operations::Op &op, std::vector<RngEngine> &rng){}
//   template <typename storage_t>
//   void read_measured_data(storage_t& creg){}
//   double norm() const;
//   double norm(const uint_t qubit, const cvector_t<double> &mat) const;
//   double norm(const reg_t &qubits, const cvector_t<double> &mat) const;
//   double norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const;
//   double norm_diagonal(const reg_t &qubits, const cvector_t<double> &mat) const;
//   double expval_pauli(const reg_t &qubits, const std::string &pauli,const complex_t initial_phase=1.0) const;
//   double expval_pauli(const reg_t &qubits, const std::string &pauli,
//                       const QubitVector<data_t>& pair_chunk, 
//                       const uint_t z_count,
//                       const uint_t z_count_pair,const complex_t initial_phase=1.0) const;
//   void set_sample_measure_index_size(int n) {sample_measure_index_size_ = n;}
//   int get_sample_measure_index_size() {return sample_measure_index_size_;}
}