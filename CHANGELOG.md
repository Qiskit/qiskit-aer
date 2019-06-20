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

[UNRELEASED](https://github.com/Qiskit/qiskit-aer/compare/0.2.1...HEAD)
=======================================================================

Added
-----
-   New simulation method for qasm simulator: tensor_network (\#56)

Changed
-------

Removed
-------

Fixed
-----

[0.2.1](https://github.com/Qiskit/qiskit-aer/compare/0.2.0...0.2.1) - 2019-05-20
================================================================================

Added
-----

Changed
-------

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
