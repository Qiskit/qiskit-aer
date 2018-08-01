# Engine Class Specifications

Authors: Christopher J. Wood
Last Updated: 24.07.2018

## General Engine Types

This is a working document for the `Engine` class outline. Currently we envision the following general engine types:

1. **State Engine**
	* Use case: statevector simulator, and unitary simulator
	* Inherits from Base Engine
	* Single shot
	* No measure, reset, conditionals
	* No noise
	* Returns final state at end of the circuit.
	* Allows state snapshots

2. **QASM Engine**:
	* Use case: emulation of real hardware experiments
	* Inherits from Base Engine
	* Multiple-shot
	* Allows measure, reset, conditionals
	* Allows noise
	* Has classical registers and handles measurement and conditionals.
	* Returns a count dictionary for memory classical bits
	* Optionally returns list of all memory bit or register bit values for each shot
	
3. **Observables Engine**:
	* Use case: emulation of real hardware experiments including post-processing of data (eg. Chemistry)
	* Inherits from QASM engine
	* Multiple-shot
	* Allows measure, reset, conditionals
	* Allows noise
	* Returns counts (if measure/memory present)
	* Returns expectation values (conditional on memory value from QASM engine if counts exist)
	* Expectation values can be either average over shots, or value for each shot.

4. **Theory Engine**:
	* Use case: QCVV and QEC research
	* Inherits from Observables Engine & State Engine
	* Multiple-shot
	* Allows measure, reset, conditionals
	* Allows noise
	* Allows state and observables snapshots	
	* Can returns QASM Engine data and Observables Engine data
	* Mid-circuit snapshots are not conditional on memory values
	* Mid-circuit observables snapshots can be single shot or averaged over shots
	* Mid-circuit state snapshots cannot be averaged over shots (unless specialized to specific state class)
	* Can we introduce new state snapshot that returns inner product with a target state?
