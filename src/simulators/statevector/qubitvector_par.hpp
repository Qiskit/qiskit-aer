/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */



#ifndef _qv_qubit_vector_par_hpp_
#define _qv_qubit_vector_par_hpp_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "framework/json.hpp"

#include "simulators/statevector/qubitvector.hpp"
#include "simulators/statevector/qsim_par/QSUnitManager.h"


namespace QV {

	/*
// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;
template <size_t N> using areg_t = std::array<uint_t, N>;
	*/

//============================================================================
// QubitVectorPar class
//============================================================================

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize_from_vector(cvector_t)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <typename data_t = complex_t*>
class QubitVectorPar : public QubitVector<data_t> {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVectorPar();
  explicit QubitVectorPar(size_t num_qubits);
  virtual ~QubitVectorPar();
  QubitVectorPar(const QubitVectorPar& obj) = delete;
  QubitVectorPar &operator=(const QubitVectorPar& obj) = delete;

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
//  complex_t &operator[](uint_t element);
//  complex_t operator[](uint_t element) const;

  // Returns a reference to the underlying data_t data class
//  data_t &data() {return data_;}

  // Returns a copy of the underlying data_t data class
//  data_t data() const {return data_;}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------
  // Set the size of the vector in terms of qubit number
  virtual void set_num_qubits(size_t num_qubits);

  // Set all entries in the vector to 0.
  void zero();

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the data vector does not match the number of qubits
  // an exception is raised.
  void initialize_from_vector(const cvector_t &data);

  // Initializes the vector to a custom initial state.
  // If num_states does not match the number of qubits an exception is raised.
  void initialize_from_data(const data_t &data, const size_t num_states);

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const reg_t &qubits, const cvector_t &mat);

  // Apply a N-qubit matrix constructed from composition of 1 and 2 qubit matrices.
  // The sets of qubits and matrices are passed as vectors, where each individual matrix
  // is input as a column-major vectorized matrix.
  void apply_matrix_sequence(const std::vector<reg_t> &regs, const std::vector<cvector_t> &mats);

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_multiplexer(const reg_t &control_qubits, const reg_t &target_qubits, const cvector_t &mat);

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const uint_t qubit, const cvector_t &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const reg_t &qubits, const cvector_t &mat);
  
  // Swap pairs of indicies in the underlying vector
  void apply_permutation_matrix(const reg_t &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a general N-qubit multi-controlled X-gate
  // If N=1 this implements an optimized X gate
  // If N=2 this implements an optimized CX gate
  // If N=3 this implements an optimized Toffoli gate
  void apply_mcx(const reg_t &qubits);

  // Apply a general multi-controlled Y-gate
  // If N=1 this implements an optimized Y gate
  // If N=2 this implements an optimized CY gate
  // If N=3 this implements an optimized CCY gate
  void apply_mcy(const reg_t &qubits);

  // Apply a general multi-controlled Z-gate
  // If N=1 this implements an optimized Z gate
  // If N=2 this implements an optimized CZ gate
  // If N=3 this implements an optimized CCZ gate
  void apply_mcz(const reg_t &qubits);
  
  // Apply a general multi-controlled single-qubit unitary gate
  // If N=1 this implements an optimized single-qubit gate
  // If N=2 this implements an optimized CU gate
  // If N=3 this implements an optimized CCU gate
  void apply_mcu(const reg_t &qubits, const cvector_t &mat);

  // Apply a general multi-controlled SWAP gate
  // If N=2 this implements an optimized SWAP  gate
  // If N=3 this implements an optimized Fredkin gate
  void apply_mcswap(const reg_t &qubits);

  //-----------------------------------------------------------------------
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  double probability(const uint_t outcome) const;

  // Return the probabilities for all measurement outcomes in the current vector
  // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  rvector_t probabilities() const;

  // Return the Z-basis measurement outcome probabilities [P(0), P(1)]
  // for measurement of specified qubit
  rvector_t probabilities(const uint_t qubit) const;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  rvector_t probabilities(const reg_t &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  std::vector<uint_t> sample_measure(const std::vector<double> &rnds) const;

  //-----------------------------------------------------------------------
  // Norms
  //-----------------------------------------------------------------------
  
  // Returns the norm of the current vector
  double norm() const;

  // These functions return the norm <psi|A^dagger.A|psi> obtained by
  // applying a matrix A to the vector. It is equivalent to returning the
  // expectation value of A^\dagger A, and could probably be removed because
  // of this.

  // Return the norm for of the vector obtained after apply the 1-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  double norm(const uint_t qubit, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  double norm(const reg_t &qubits, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the 1-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const uint_t qubit, const cvector_t &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const reg_t &qubits, const cvector_t &mat) const;

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
	//actual data is here
	QSUnitManager* m_pUnits;
	int nprocs,myrank;

};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
QubitVectorPar<data_t>::QubitVectorPar(size_t num_qubits) : QubitVector<data_t>(num_qubits)
{
	m_pUnits = NULL;
}

template <typename data_t>
QubitVectorPar<data_t>::QubitVectorPar() : QubitVectorPar(0)
{
	m_pUnits = NULL;
}

template <typename data_t>
QubitVectorPar<data_t>::~QubitVectorPar()
{
	if(m_pUnits){
		delete m_pUnits;
	}
}


//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorPar<data_t>::zero()
{
	if(m_pUnits){
		m_pUnits->Clear();
	}
}

template <typename data_t>
void QubitVectorPar<data_t>::set_num_qubits(size_t num_qubits) {
  QubitVector<data_t>::num_qubits_ = num_qubits;
  QubitVector<data_t>::data_size_ = BITS[num_qubits];

  // Free any currently assigned memory
  if (QubitVector<data_t>::data_)
    free(QubitVector<data_t>::data_);

  if (QubitVector<data_t>::checkpoint_) {
    free(QubitVector<data_t>::checkpoint_);
    QubitVector<data_t>::checkpoint_ = nullptr;
  }

  // Allocate memory for new vector
  QubitVector<data_t>::data_ = reinterpret_cast<complex_t*>(malloc(sizeof(complex_t)));	//dummy

	if(num_qubits > 0){
		if(m_pUnits){
			//reuse statevectors if number of qubit is the same
			if(m_pUnits->Qubit() != num_qubits){
				delete m_pUnits;
				m_pUnits = NULL;
			}
		}

#ifdef QSIM_MPI
		MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#endif

		if(m_pUnits == NULL){
			//actual data
			m_pUnits = new QSUnitManager(num_qubits);
			m_pUnits->Init();
		}
	}
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorPar<data_t>::initialize()
{
	if(m_pUnits){
		QSDoubleComplex c = 1.0;
		m_pUnits->Clear();
		m_pUnits->SetValue(c,0);
	}
}

template <typename data_t>
void QubitVectorPar<data_t>::initialize_from_vector(const cvector_t &statevec)
{
	if (QubitVector<data_t>::data_size_ != statevec.size()) {
		std::string error = "QubitVector::initialize input vector is incorrect length (" + 
			std::to_string(QubitVector<data_t>::data_size_) + "!=" +
			std::to_string(statevec.size()) + ")";
		throw std::runtime_error(error);
	}
	if(m_pUnits){
		m_pUnits->Copy((QSComplex*)&statevec[0]);
	}
}

template <typename data_t>
void QubitVectorPar<data_t>::initialize_from_data(const data_t &statevec, const size_t num_states)
{
	if (QubitVector<data_t>::data_size_ != num_states) {
		std::string error = "QubitVector::initialize input vector is incorrect length (" +
			std::to_string(QubitVector<data_t>::data_size_) + "!=" + std::to_string(num_states) + ")";
		throw std::runtime_error(error);
	}
	if(m_pUnits){
		m_pUnits->Copy((QSComplex*)&statevec[0]);
	}
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template <typename data_t>
void QubitVectorPar<data_t>::apply_matrix(const reg_t &qubits,
                                       const cvector_t &mat)
{
	const size_t N = qubits.size();

	int i,j,m;
	cvector_t mt(mat);

	int qt[32];
	for(i=0;i<N;i++){
		qt[i] = (int)qubits[i];
	}
	if(N == 2){
		if(qt[0] > qt[1]){
			complex_t t;
			i = qt[0];
			qt[0] = qt[1];
			qt[1] = i;

			for(i=0;i<4;i++){
				t = mt[i + 4];
				mt[i + 4] = mt[i + 8];
				mt[i + 8] = t;
			}

			for(i=0;i<4;i++){
				t = mt[i*4 + 1];
				mt[i*4 + 1] = mt[i*4 + 2];
				mt[i*4 + 2] = t;
			}
		}
	}

	m_pUnits->MatMult((QSDoubleComplex*)&mt[0],qt,N);
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_matrix_sequence(const std::vector<reg_t> &regs,
                                         const std::vector<cvector_t> &mats) {
  if (mats.size() == 0)
    return;


#ifdef DEBUG
  if (regs.size() != mats.size());
    throw std::runtime_error("QubitVector<data_t>::apply_matrix_sequence allows same size of qubitss and mats.");
#endif

  bool at_most_two = true;
  // check 1 or 2 qubits
  for (const reg_t& reg: regs) {
    if (reg.size() > 2) {
      at_most_two = false;
      break;
    }
  }

  if (!at_most_two) {
    for (size_t i = 0; i < regs.size(); ++i)
      apply_matrix(regs[i], mats[i]);
    return;
  }


  reg_t sorted_qubits;
  for (const reg_t& reg: regs)
    for (const uint_t qubit: reg)
      if (std::find(sorted_qubits.begin(), sorted_qubits.end(), qubit) == sorted_qubits.end())
        sorted_qubits.push_back(qubit);

  std::sort(sorted_qubits.begin(), sorted_qubits.end());

  std::vector<cvector_t> sorted_mats;

  for (size_t i = 0; i < regs.size(); ++i) {
    const reg_t& reg = regs[i];
    const cvector_t& mat = mats[i];
    sorted_mats.push_back(QubitVector<data_t>::expand_matrix(reg, sorted_qubits, mat));
  }

  auto U = sorted_mats[0];
  const auto dim = BITS[sorted_qubits.size()];

  for (size_t m = 1; m < sorted_mats.size(); m++) {

    cvector_t u_tmp(U.size(), 0.);
    const cvector_t& u = sorted_mats[m];

    for (size_t i = 0; i < dim; ++i)
      for (size_t j = 0; j < dim; ++j)
        for (size_t k = 0; k < dim; ++k)
          u_tmp[i + j * dim] += u[i + k * dim] * U[k + j * dim];

    U = u_tmp;
  }

  apply_matrix(sorted_qubits, U);
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_multiplexer(const reg_t &control_qubits,
		const reg_t &target_qubits,
		const cvector_t &mat) {

	printf(" apply_multiplexer NOT SUPPORTED : %d, %d \n",control_qubits.size(),target_qubits.size());
			/*
  // General implementation
  const size_t control_count = control_qubits.size();
  const size_t target_count  = target_qubits.size();
  const uint_t DIM = BITS[(target_count+control_count)];
  const uint_t columns = BITS[target_count];
  const uint_t blocks = BITS[control_count];
  // Lambda function for stacked matrix multiplication
  auto lambda = [&](const indexes_t &inds, const cvector_t &_mat)->void {
    auto cache = std::make_unique<complex_t[]>(DIM);
    for (uint_t i = 0; i < DIM; i++) {
      const auto ii = inds[i];
      cache[i] = data_[ii];
      data_[ii] = 0.;
    }
    // update state vector
    for (uint_t b = 0; b < blocks; b++)
      for (uint_t i = 0; i < columns; i++)
        for (uint_t j = 0; j < columns; j++)
	{
	  data_[inds[i+b*columns]] += _mat[i+b*columns + DIM * j] * cache[b*columns+j];
	}
  };
  
  // Use the lambda function
  auto qubits = target_qubits;
  for (const auto &q : control_qubits) {qubits.push_back(q);}
  apply_lambda(lambda, qubits, mat);
			*/
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_diagonal_matrix(const reg_t &qubits,
                                                const cvector_t &diag) {
	const size_t N = qubits.size();

//	printf(" apply_diagonal_matrix : %d , %d\n",N,qubits[0]);

	m_pUnits->MatMultDiagonal((QSDoubleComplex*)&diag[0],(int*)&qubits[0],N);
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_permutation_matrix(const reg_t& qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs) {
	const size_t N = qubits.size();

	printf(" apply_permutation_matrix : %d , %d   NOT SUPPORTED\n",N,qubits[0]);

}


/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Multi-controlled gates
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorPar<data_t>::apply_mcx(const reg_t &qubits) {

	// Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();

	if(N == 1){		//X
		m_pUnits->X(qubits[0]);
	}
	else if(N == 2){		//CX
		m_pUnits->CX(qubits[1],qubits[0]);
	}

/*
  switch (N) {
    case 1: {
      // Lambda function for X gate
      auto lambda = [&](const areg_t<2> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CX gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for Toffli gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled X gate
      auto lambda = [&](const indexes_t &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
	*/
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_mcy(const reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = MASKS[N];
  const complex_t I(0., 1.);

	if(N == 1){		//Y
		m_pUnits->Y(qubits[0]);
	}

/*
  switch (N) {
    case 1: {
      // Lambda function for Y gate
      auto lambda = [&](const areg_t<2> &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CY gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for CCY gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled Y gate
      auto lambda = [&](const indexes_t &inds)->void {
        const complex_t cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
	*/
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_mcz(const reg_t &qubits) {
  const size_t N = qubits.size();

	if(N == 1){		//Z
		m_pUnits->Z(qubits[0]);
	}

/*
  switch (N) {
    case 1: {
      // Lambda function for Z gate
      auto lambda = [&](const areg_t<2> &inds)->void {
        data_[inds[1]] *= -1.;
      };
      apply_lambda(lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CZ gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        data_[inds[3]] *= -1.;
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for CCZ gate
      auto lambda = [&](const areg_t<8> &inds)->void {
         data_[inds[7]] *= -1.;
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled X gate
      auto lambda = [&](const indexes_t &inds)->void {
         data_[inds[MASKS[N]]] *= -1.;
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
	*/
}


template <typename data_t>
void QubitVectorPar<data_t>::apply_mcswap(const reg_t &qubits) {
  // Calculate the swap positions for the last two qubits.
  // If N = 2 this is just a regular SWAP gate rather than a controlled-SWAP gate.
  const size_t N = qubits.size();
  const size_t pos0 = MASKS[N - 1];
  const size_t pos1 = pos0 + BITS[N - 2];

	printf(" apply_mcswap : %d NOT SUPPORTED\n",N);
	/*
  switch (N) {
    case 2: {
      // Lambda function for SWAP gate
      auto lambda = [&](const areg_t<4> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for C-SWAP gate
      auto lambda = [&](const areg_t<8> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled SWAP gate
      auto lambda = [&](const indexes_t &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      apply_lambda(lambda, qubits);
    }
  } // end switch
	*/
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_mcu(const reg_t &qubits,const cvector_t &mat){
	const size_t N = qubits.size();

	if(N == 1){
		if(mat[1] == 0.0 && mat[2] == 0.0){
		    const cvector_t diag = {{mat[0], mat[3]}};

			m_pUnits->MatMultDiagonal((QSDoubleComplex*)&diag[0],(int*)&qubits[0],1);
		}
		else{
			m_pUnits->MatMult((QSDoubleComplex*)&mat[0],(int*)&qubits[0],N);
		}
	}

                                    	/*
  // Check if matrix is actually diagonal and if so use 
  // diagonal matrix lambda function
  // TODO: this should be changed to not check doubles with ==
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t diag = {{mat[0], mat[3]}};
    // Diagonal version
    switch (N) {
      case 1: {
        // If N=1 this is just a single-qubit matrix
        apply_diagonal_matrix(qubits[0], diag);
        return;
      }
      case 2: {
        // Lambda function for CU gate
        auto lambda = [&](const areg_t<4> &inds,
                          const cvector_t &_diag)->void {
          data_[pos0] = _diag[0] * data_[pos0];
          data_[pos1] = _diag[1] * data_[pos1];
        };
        apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), diag);
        return;
      }
      case 3: {
        // Lambda function for CCU gate
        auto lambda = [&](const areg_t<8> &inds,
                          const cvector_t &_diag)->void {
          data_[pos0] = _diag[0] * data_[pos0];
          data_[pos1] = _diag[1] * data_[pos1];
        };
        apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), diag);
        return;
      }
      default: {
        // Lambda function for general multi-controlled U gate
        auto lambda = [&](const indexes_t &inds,
                          const cvector_t &_diag)->void {
          data_[pos0] = _diag[0] * data_[pos0];
          data_[pos1] = _diag[1] * data_[pos1];
        };
        apply_lambda(lambda, qubits, diag);
        return;
      }
    } // end switch
  }

  // Non-diagonal version
  switch (N) {
    case 1: {
      // If N=1 this is just a single-qubit matrix
      apply_matrix(qubits[0], mat);
      return;
    }
    case 2: {
      // Lambda function for CU gate
      auto lambda = [&](const areg_t<4> &inds,
                        const cvector_t &_mat)->void {
      const auto cache = data_[pos0];
      data_[pos0] = _mat[0] * data_[pos0] + _mat[2] * data_[pos1];
      data_[pos1] = _mat[1] * cache + _mat[3] * data_[pos1];
      };
      apply_lambda(lambda, areg_t<2>({{qubits[0], qubits[1]}}), mat);
      return;
    }
    case 3: {
      // Lambda function for CCU gate
      auto lambda = [&](const areg_t<8> &inds,
                        const cvector_t &_mat)->void {
      const auto cache = data_[pos0];
      data_[pos0] = _mat[0] * data_[pos0] + _mat[2] * data_[pos1];
      data_[pos1] = _mat[1] * cache + _mat[3] * data_[pos1];
      };
      apply_lambda(lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}), mat);
      return;
    }
    default: {
      // Lambda function for general multi-controlled U gate
      auto lambda = [&](const indexes_t &inds,
                        const cvector_t &_mat)->void {
      const auto cache = data_[pos0];
      data_[pos0] = _mat[0] * data_[pos0] + _mat[2] * data_[pos1];
      data_[pos1] = _mat[1] * cache + _mat[3] * data_[pos1];
      };
      apply_lambda(lambda, qubits, mat);
      return;
    }
  } // end switch
                                    	*/
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVectorPar<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t& mat) {
	int qt = (int)qubit;

//	printf(" apply_matrix(1) : %d\n",qubit);

	m_pUnits->MatMult((QSDoubleComplex*)&mat[0],&qt,1);
}

template <typename data_t>
void QubitVectorPar<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t& diag) {
	int qt = (int)qubit;

//	printf(" apply_diagonal_matrix(1) : %d\n",qubit);

	m_pUnits->MatMultDiagonal((QSDoubleComplex*)&diag[0],&qt,1);
}


/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
double QubitVectorPar<data_t>::norm() const {
	/*
  // Lambda function for norm
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    (void)val_im; // unused
    val_re += std::real(data_[k] * std::conj(data_[k]));
  };
  return std::real(apply_reduction_lambda(lambda));
	*/
	return 0.0;
}

template <typename data_t>
double QubitVectorPar<data_t>::norm(const reg_t &qubits, const cvector_t &mat) const {
	printf(" norm\n");
	return 0.0;
/*
  const uint_t N = qubits.size();
  const uint_t DIM = BITS[N];
  // Error checking
  #ifdef DEBUG
  check_vector(mat, 2 * N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const indexes_t &inds, const cvector_t &_mat, 
                    double &val_re, double &val_im)->void {
    (void)val_im; // unused
    for (size_t i = 0; i < DIM; i++) {
      complex_t vi = 0;
      for (size_t j = 0; j < DIM; j++)
        vi += _mat[i + DIM * j] * data_[inds[j]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_reduction_lambda(lambda, qubits, mat));
	*/
}

template <typename data_t>
double QubitVectorPar<data_t>::norm_diagonal(const reg_t &qubits, const cvector_t &mat) const {
	printf(" norm diag\n");
	return 0.0;
/*
  const uint_t N = qubits.size();
  const uint_t DIM = BITS[N];

  // Error checking
  #ifdef DEBUG
  check_vector(mat, N);
  #endif

  // Lambda function for N-qubit matrix norm
  auto lambda = [&](const indexes_t &inds,
                    const cvector_t &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    for (size_t i = 0; i < DIM; i++) {
      const auto vi = _mat[i] * data_[inds[i]];
      val_re += std::real(vi * std::conj(vi));
    }
  };
  // Use the lambda function
  return std::real(apply_reduction_lambda(lambda, qubits, mat));*/
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
double QubitVectorPar<data_t>::norm(const uint_t qubit, const cvector_t &mat) const {
		printf(" norm\n");
	return 0.0;
/*
  // Error handling
  #ifdef DEBUG
  check_vector(mat, 2);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const areg_t<2> &inds,
                    const cvector_t &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    const auto v0 = _mat[0] * data_[inds[0]] + _mat[2] * data_[inds[1]];
    const auto v1 = _mat[1] * data_[inds[0]] + _mat[3] * data_[inds[1]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(lambda, areg_t<1>({{qubit}}), mat));
	*/
}

template <typename data_t>
double QubitVectorPar<data_t>::norm_diagonal(const uint_t qubit, const cvector_t &mat) const {
	printf(" norm diag\n");
	return 0.0;
/*
	// Error handling
  #ifdef DEBUG
  check_vector(mat, 1);
  #endif
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const areg_t<2> &inds,
                    const cvector_t &_mat,
                    double &val_re,
                    double &val_im)->void {
    (void)val_im; // unused
    const auto v0 = _mat[0] * data_[inds[0]];
    const auto v1 = _mat[1] * data_[inds[1]];
    val_re += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  return std::real(apply_reduction_lambda(lambda, areg_t<1>({{qubit}}), mat));
	*/
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/

template <typename data_t>
double QubitVectorPar<data_t>::probability(const uint_t outcome) const {
	/*
  const auto v = data_[outcome];
  return std::real(v * std::conj(v));
	*/
	return 0.0;
}

template <typename data_t>
rvector_t QubitVectorPar<data_t>::probabilities() const {
	/*
  rvector_t probs(data_size_);
  const int_t END = data_size_;
  probs.assign(data_size_, 0.);

#pragma omp parallel for if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  for (int_t j=0; j < END; j++) {
    probs[j] = probability(j);
  }
  return probs;
	*/
	rvector_t probs(1, 0.);
	return probs;
}

template <typename data_t>
rvector_t QubitVectorPar<data_t>::probabilities(const reg_t &qubits) const {

	const size_t N = qubits.size();
	rvector_t probs((1ull << N), 0.);

	if(N == 1){
		probs[0] = m_pUnits->Dot(qubits[0]);
		probs[1] = 1.0 - probs[0];

#ifdef QSIM_DEBUG
		printf(" prob[%d] : (%e, %e) \n",qubits[0],probs[0],probs[1]);
#endif
	}

	return probs;
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------

template <typename data_t>
rvector_t QubitVectorPar<data_t>::probabilities(const uint_t qubit) const {

	double p0,p1;

	p0 = m_pUnits->Dot(qubit);
	p1 = 1.0 - p0;

	return rvector_t({p0,p1});
}


//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVectorPar<data_t>::sample_measure(const std::vector<double> &rnds) const {

	const int_t SHOTS = rnds.size();
	reg_t samples;
	int i;


	samples.assign(SHOTS, 0);

	if(m_pUnits != NULL){
		QSDouble* rs;
		QSUint* pos;

		rs = new QSDouble[SHOTS];
		pos = new QSUint[SHOTS];

		for(i=0;i<SHOTS;i++){
			rs[i] = rnds[i];
		}
		m_pUnits->Measure_FindPos(rs,pos,SHOTS);

		for(i=0;i<SHOTS;i++){
			samples[i] = pos[i];
		}
		delete[] rs;
		delete[] pos;
	}

  return samples;
}

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
