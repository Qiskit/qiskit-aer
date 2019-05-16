/**
 * Copyright 2019, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_qasm_controller_opt_hpp_
#define _aer_qasm_controller_opt_hpp_

#include "base/controller.hpp"
#include "simulators/qasm/qasm_controller.hpp"
#include "simulators/statevector/statevector_state.hpp"
#include "simulators/statevector/qubitvector_par.hpp"

#ifdef QSIM_MPI
#include <mpi.h>
#endif



namespace AER {
namespace Simulator {

//=========================================================================
// QasmControllerPar class
//=========================================================================

class QasmControllerPar : public QasmController {

public:
  QasmControllerPar();

protected:

  virtual OutputData run_circuit(const Circuit &circ,
                                 uint_t shots,
                                 uint_t rng_seed) const override;
};

//-------------------------------------------------------------------------
// Constructor
//-------------------------------------------------------------------------
QasmControllerPar::QasmControllerPar()
{
	add_circuit_optimization(Transpile::ReduceNop());
	add_circuit_optimization(Transpile::Fusion(4));
	add_circuit_optimization(Transpile::TruncateQubits());
}

OutputData QasmControllerPar::run_circuit(const Circuit &circ,
                                       uint_t shots,
                                       uint_t rng_seed) const
{
#ifdef QSIM_MPI
	//to make sure using the same random seed number on all processes
	MPI_Bcast(&rng_seed,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
#endif

	// Execute according to simulation method
	switch (simulation_method(circ)) {
		case Method::statevector:
		// Statvector simulation
		return run_circuit_helper<Statevector::State<QV::QubitVectorPar<complex_t*>>>(circ,
                                                      shots,
                                                      rng_seed,
                                                      initial_statevector_); // allow custom initial state
		default:
		return QasmController::run_circuit(circ, shots, rng_seed);
	}
}

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
