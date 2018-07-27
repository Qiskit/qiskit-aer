# Engine Class Specifications

Authors: Christopher J. Wood
Last Updated: 24.07.2018

## General Engine Types

This is a working document for the `Engine` class outline. Currently we envision the following general engine types:

1. **Base Engine**
	* Single shot
	* Doesn't support measurements or conditionals
	* Passes all instructions to backend
	* Returns final state at end of the circuit.
2. **QASM Engine**:
	* Inherits from Base Engine
	* Multiple-shot 
	* Has classical registers and handles measurement and conditionals.
	* Returns a count dictionary for memory classical bits
	* Optionally returns list of all memory bit or register bit values for each shot
3. **Observables Engine**:
	* Inherits from base
	* Multiple-shot
	* Returns expectation values (conditional on measurements in future)
	* Measurements could be implemented as (projector) expectation values in measurement basis
	* Doesn't allow conditionals (for now?)
	* Can be specialized for certain backends (eg. statevector, clifford).
4. **Snapshot Engine**:
	* Inherets from observables engine
	* Multiple-shot
	* Can place observables mid circuit (not possible in real devices)
	* Can use snapshot commands.
	* Doesn't allow measure or conditionals (for now?)
	* Allows reset
