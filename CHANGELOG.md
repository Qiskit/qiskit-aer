Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).

> **Types of changes:**
>
> -   **Added**: for new features.
> -   **Changed**: for changes in existing functionality.
> -   **Deprecated**: for soon-to-be removed features.
> -   **Removed**: for now removed features.
> -   **Fixed**: for any bug fixes.
> -   **Security**: in case of vulnerabilities.

[UNRELEASED](https://github.com/Qiskit/qiskit-aer/compare/0.3.3...HEAD)
=======================================================================

Added
-----
- Added support for probabilities snapshot and Pauli expectation value snapshot in the stabilizer simulator (\#423)

Changed
-------

Removed
-------

Fixed
-----
- Fixes bug where cu3 was being applied as cu1 for unitary_simulator (\#483)

[0.3.3](https://github.com/Qiskit/qiskit-aer/compare/0.3.2...0.3.3) - 2019-11-14
====================================================================================

Added
-----
- Added controlled gates (``cu1``, ``cu2``, ``cu3``) to simulator basis_gates (\#417)
- Added multi-controlled gates (``mcx``, ``mcy``, ``mcz``, ``mcu1``, ``mcu2``, ``mcu3``)
  to simulator basis gates (\#417)
- Added gate definitions to simulator backend configurations (\#417)

Changed
-------
- Improved pershot snapshot data container performance (\#405)
- Add basic linear algebra functions for numeric STL classes (\#406)
- Improved average snapshot data container performance (\#407)

Removed
-------

Fixed
-----


[0.3.2](https://github.com/Qiskit/qiskit-aer/compare/0.3.1...0.3.2) - 2019-10-16
===============================================================================

Added
-----

Changed
-------

Removed
-------

Fixed
-----
- Fix sdist to always attempt to build (\#401)
- New (efficient) implementation for expectation_value_pauli in MPS simulation method (\#344)


[0.3.1](https://github.com/Qiskit/qiskit-aer/compare/0.3.0...0.3.1) - 2019-10-15
===============================================================================


Added
-----
- Added tests for the Fredkin gate (#357)
- Added tests for the cu1 gate (#360)
- Added tests for statevector and stabilizer snapshots (\#355)
- Added tests for density matrix snapshot (\#374)
- Added tests for probabilities snapshot (\#380)
- Added support for reset() in MPS simulation method (\#393)
- Added tests for matrix and Pauli expectation value snapshot (\#386)

Changed
-------
- Changes signature of SnapshotExpectationValue extension and the way qubit position parameters are parsed in expectation_value_matrix qobj instructions (\#386)
- Change signature of SnapshotProbabilities extension (\#380)
- Change signature of SnapshotDensityMatrix extension (\#374)
- Stabilizer snapshot returns stabilizer instead of full Clifford table (\#355)
- Signature of SnapshotStatevector and SnapshotStabilizer (\#355)
- Changed all names from tensor_network_state to matrix_product_state (\#356)
- Update device noise model to consume asymmetric readout errors from backends (\#354)
- Update device noise model to use gate_length (\#352)
- Refactoring code and introducing floating point comparison func (\#338)

Removed
-------

Fixed
-----


[0.3.0](https://github.com/Qiskit/qiskit-aer/compare/0.2.3...0.3.0) - 2019-08-21
===============================================================================

Added
-----
- New simulation method for qasm simulator: tensor_network (\#56)
- Added superop qobj instruction and superoperator matrix utils (\#289)
- Added support for conditional unitary, kraus, superop qobj instructions (\#291)
- Add "validation_threshold" config parameter to Aer backends (\#290)
- Added support for apply_measure in tensor_network_state. Also changed
  sample_measure to use apply_measure (\#299)
- Added density matrix simulation method to QasmSimulator (\#295, \#253)
- Adds delay measure circuit optimization (\#317)
- Added sampling for sampling with readout-errors (\#222)
- Added support of single precision for statevector and density matrix simulation (\#286, \#315)
- Noise model inserter module (\#239)

Changed
-------
- Added density matrix method to automatic QasmSimulator methods (\#316)

Removed
-------


Fixed
-----
- Bug in handling parallelization in matrix_product_state.cpp (PR\#292)
- Added support for multiplication by coeff in tensor_network_state expectation value snapshots (PR\#294)
- Change name of qasm simulation method from tensor_network to matrix_product_state (\#320)


[0.2.3](https://github.com/Qiskit/qiskit-aer/compare/0.2.2...0.2.3) - 2019-07-11
===============================================================================

Fixed
-----
-   Bug in measure sampling conditional checking with conditional instructions (\#280)


[0.2.2](https://github.com/Qiskit/qiskit-aer/compare/0.2.1...0.2.2) - 2019-07-10
================================================================================

Added
-----
-   Added multi-controlled phase gate to `QubitVector` and changed
    multi-controlled Z and multi-controlled u1 gates to use this method (\# 258)
-   Added optimized anti-diagonal single-qubit gates to QubitVector (\# 258)

Changed
-------
-   Improve performance of matrix fusion circuit optimization and move fusion
    code out of `QubitVector` class and into Fusion optimization class (\#255)

Removed
-------
-   Remove `matrix_sequence` Op type from `Op` class (\#255)

Fixed
-----
-   Change maximum parameter for depolarizing_error to allow for error channel
    with no identity component. (\#243)
-   Fixed 2-qubit depolarizing-only error parameter calculation in
    basic_device_noise_model (\#243)
-   Set maximum workers to ThreadPoolExecutor in AerJob to limit thread creation (\#259)

[0.2.1](https://github.com/Qiskit/qiskit-aer/compare/0.2.0...0.2.1) - 2019-05-20
================================================================================

Added
-----
-   Added 2-qubit Pauli and reset approximation to noise transformation (\#236)
-   Added `to_instruction` method to `ReadoutError` (\#257).

Changed
-------

-   When loading qobj check if all instructions are conditional and raise an
    exception if an unsupported instruction is conditional (\#271)
-   Deprecate the use of \".as\_dict()\" in favor of \".to\_dict()\"
    (\#228)
-   Set simulator seed from \"seed\_simulator\" in qobj (\#210)

Removed
-------

Fixed
-----

-   Fix memory error handling for huge circuits (\#216)
-   Fix equality expressions in Python code (\#208)

[0.2.0](https://github.com/Qiskit/qiskit-aer/compare/0.1.1...0.2.0) - 2019-05-02
================================================================================

Added
-----

-   Add multiplexer gate (\#192)
-   Add [remap\_noise\_model]{.title-ref} function to noise.utils
    (\#181)
-   Add [\_\_eq\_\_]{.title-ref} method to NoiseModel, QuantumError,
    ReadoutError (\#181)
-   Add support for labelled gates in noise models (\#175).
-   Add optimized mcx, mcy, mcz, mcu1, mcu2, mcu3, gates to QubitVector
    (\#124)
-   Add optimized controlled-swap gate to QubitVector (\#142)
-   Add gate-fusion optimization for QasmContoroller, which is enabled
    by setting fusion\_enable=true (\#136)
-   Add better management of failed simulations (\#167)
-   Add qubits truncate optimization for unused qubits (\#164)
-   Add ability to disable depolarizing error on device noise model
    (\#131)
-   Add initialise simulator instruction to statevector\_state (\#117,
    \#137)
-   Add coupling maps to simulators (\#93)
-   Add circuit optimization framework (\#83)
-   Add benchmarking (\#71, \#177)
-   Add wheels support for Debian-like distributions (\#69)
-   Add autoconfiguration of threads for qasm simulator (\#61)
-   Add Simulation method based on Stabilizer Rank Decompositions (\#51)

Changed
-------

-   Add basis\_gates kwarg to NoiseModel init (\#175).
-   Depreciated \"initial\_statevector\" backend option for
    QasmSimulator and StatevectorSimulator (\#185)
-   Rename \"chop\_threshold\" backend option to \"zero\_threshold\" and
    change default value to 1e-10 (\#185).
-   Add an optional parameter to [NoiseModel.as\_dict()]{.title-ref} for
    returning dictionaries that can be serialized using the standard
    [json]{.title-ref} library directly. (\#165)
-   Refactor thread management (\#50)

Removed
-------

Fixed
-----

-   Improve noise transformations (\#162)
-   Improve error reporting (\#160)
-   Improve efficiency of parallelization with max\_memory\_mb a new
    parameter of backend\_opts (\#61)
-   Improve u1 performance in statevector (\#123)
-   Fix OpenMP clashing problems on MacOS for the Terra Addon (\#46)

[0.1.1](https://github.com/Qiskit/qiskit-aer/compare/0.1.0...0.1.1) - 2019-01-24
================================================================================

Added
-----

-   Adds version information when using the standalone simulator (\#36)
-   Adds a Clifford stabilizer simulation method to the QasmSimulator
    (\#13)
-   Improve Circuit and NoiseModel instructions checking (\#31)
-   Add reset\_error function to Noise models (\#34)
-   Improve error reporting at installation time (\#29)
-   Validate n\_qubits before execution (\#24)
-   Add qobj method to AerJob (\#19)

Removed
-------

-   Reference model tests removed from the codebase (\#27)

Fixed
-----

-   Fix Contributing guide (\#33)
-   Fix an import in Terra integration tests (\#33)
-   Fix non-OpenMP builds (\#19)

[0.1.0](https://github.com/Qiskit/qiskit-aer/compare/0.0.0...0.1.0) - 2018-12-19
================================================================================

Added
-----

-   QASM Simulator
-   Statevector Simulator
-   Unitary Simulator
-   Noise models
-   Terra integration
-   Standalone Simulators support
