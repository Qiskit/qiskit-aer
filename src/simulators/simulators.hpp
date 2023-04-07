/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019. 2023.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_simulators_hpp_
#define _aer_simulators_hpp_

#include "simulators/density_matrix/densitymatrix_state.hpp"
#include "simulators/extended_stabilizer/extended_stabilizer_state.hpp"
#include "simulators/matrix_product_state/matrix_product_state.hpp"
#include "simulators/stabilizer/stabilizer_state.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/superoperator/superoperator_state.hpp"
#include "simulators/tensor_network/tensor_net_state.hpp"
#include "simulators/unitary/unitary_state.hpp"

namespace AER {

// Simulation methods
enum class Method {
  automatic,
  statevector,
  density_matrix,
  matrix_product_state,
  stabilizer,
  extended_stabilizer,
  unitary,
  superop,
  tensor_network
};

enum class Device { CPU, GPU, ThrustCPU };

// Simulation precision
enum class Precision { Double, Single };

const std::unordered_map<Method, std::string> method_names_ = {
    {Method::automatic, "automatic"},
    {Method::statevector, "statevector"},
    {Method::density_matrix, "density_matrix"},
    {Method::matrix_product_state, "matrix_product_state"},
    {Method::stabilizer, "stabilizer"},
    {Method::extended_stabilizer, "extended_stabilizer"},
    {Method::unitary, "unitary"},
    {Method::superop, "superop"},
    {Method::tensor_network, "tensor_network"}};

//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
