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

#ifndef _pauli_hpp_
#define _pauli_hpp_

#include <iostream>
#include "binary_vector.hpp"

namespace Pauli {

/*******************************************************************************
 *
 * Pauli Class
 *
 ******************************************************************************/

class Pauli {
public:
  // Symplectic binary vectors
  BV::BinaryVector X;
  BV::BinaryVector Z;

  // Construct an empty pauli
  Pauli() : X(0), Z(0) {};

  // Construct an n-qubit identity Pauli
  explicit Pauli(uint64_t len) : X(len), Z(len) {};

  // Construct an n-qubit Pauli from a string label;
  explicit Pauli(const std::string &label);

  // Return the string representation of the Pauli
  std::string str() const;

  // exponent g of i such that P(x1,z1) P(x2,z2) = i^g P(x1+x2,z1+z2)
  static int8_t phase_exponent(const Pauli& pauli1, const Pauli& pauli2);
};


/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

Pauli::Pauli(const std::string &label) {
  const auto num_qubits = label.size();
  X = BV::BinaryVector(num_qubits);
  Z = BV::BinaryVector(num_qubits);
  // The label corresponds to tensor product order
  // So the first element of label is the last qubit and vice versa
  for (size_t i =0; i < num_qubits; i++) {
    const auto qubit_i = num_qubits - 1 - i;
    switch (label[i]) {
      case 'I':
        break;
      case 'X':
        X.set1(qubit_i);
        break;
      case 'Y':
        X.set1(qubit_i);
        Z.set1(qubit_i);
        break;
      case 'Z':
        Z.set1(qubit_i);
        break;
      default:
        throw std::invalid_argument("Invalid Pauli label");
    }
  }
}

std::string Pauli::str() const {
  // Check X and Z are same length
  const auto num_qubits = X.getLength();
  if (Z.getLength() != num_qubits)
    throw std::runtime_error("Pauli::str X and Z vectors are different length.");
  std::string label;
  for (size_t i =0; i < num_qubits; i++) {
    const auto qubit_i = num_qubits - 1 - i;
    if (!X[qubit_i] && !Z[qubit_i])
      label.push_back('I');
    else if (X[qubit_i] && !Z[qubit_i])
      label.push_back('X');
    else if (X[qubit_i] && Z[qubit_i])
      label.push_back('Y');
    else if (!X[qubit_i] && Z[qubit_i])
      label.push_back('Z');
  }
  return label;
}

int8_t Pauli::phase_exponent(const Pauli& pauli1, const Pauli& pauli2) {
  int8_t exponent = 0;
  for (size_t q = 0; q < pauli1.X.getLength(); q++) {
    exponent += pauli2.X[q] * pauli1.Z[q] * (1 + 2 * pauli2.Z[q] + 2 * pauli1.X[q]);
    exponent -= pauli1.X[q] * pauli2.Z[q] * (1 + 2 * pauli1.Z[q] + 2 * pauli2.X[q]);
    exponent %= 4;
  }
  if (exponent < 0)
      exponent += 4;
  return exponent;
}

//------------------------------------------------------------------------------
} // end namespace Pauli
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class statevector_t>
inline std::ostream &operator<<(std::ostream &out, const Pauli::Pauli &pauli) {
  out << pauli.str();
  return out;
}

//------------------------------------------------------------------------------
#endif

