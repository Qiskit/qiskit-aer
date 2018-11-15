/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_qasm_controller_hpp_
#define _aer_qasm_controller_hpp_

#include "base/controller.hpp"
#include "qv_state.hpp"

namespace AER {
namespace Simulator {

//=========================================================================
// StatevectorController class
//=========================================================================

class StatevectorController : public Base::Controller {
private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // This simulator will only return a single shot, regardless of the
  // input shot number
  virtual OutputData run_circuit(const Circuit &circ,
                                 uint_t shots,
                                 uint_t rng_seed,
                                 int num_threads_state) const override;
};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

OutputData StatevectorController::run_circuit(const Circuit &circ,
                                       uint_t shots,
                                       uint_t rng_seed,
                                       int num_threads_state) const {  
  
  // Check if circuit can run on a statevector simulator
  // TODO: Should we make validate circuit a static method of the class?
  bool statevec_valid = QubitVector::State<>().validate_circuit(circ);  
  // throw exception listing the invalid instructions
  if (statevec_valid == false) {
    QubitVector::State<>().validate_circuit_except(circ);
  }

  // Initialize statevector
  QubitVector::State<> state;
  state.set_config(Base::Controller::state_config_);
  state.set_available_threads(num_threads_state);
  
  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  OutputData data;
  data.set_config(Base::Controller::data_config_);
  
  // Run single shot collecting measure data or snapshots
  state.initialize_qreg(circ.num_qubits);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(circ.ops, data, rng);
  state.add_creg_to_data(data);
  
  // Add final state to the data
  data.add_additional_data("statevector", state.qreg());

  return data;
}


//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
