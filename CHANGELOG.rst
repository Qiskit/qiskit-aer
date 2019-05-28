Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_.

  **Types of changes:**

  - **Added**: for new features.
  - **Changed**: for changes in existing functionality.
  - **Deprecated**: for soon-to-be removed features.
  - **Removed**: for now removed features.
  - **Fixed**: for any bug fixes.
  - **Security**: in case of vulnerabilities.

`UNRELEASED`_
=============

Added
-----

Changed
-------

Removed
-------

Fixed
-----


`0.2.1`_ - 2019-05-20
=====================

Added
-----

Changed
-------
- Set simulator seed from "seed_simulator" in qobj (#210)

Removed
-------

Fixed
-----
- Fix memory error handling for huge circuits (#216)
- Fix equality expressions in Python code (#208)

`0.2.0`_ - 2019-05-02
=====================

Added
-----
- Add multiplexer gate (#192)
- Add `remap_noise_model` function to noise.utils (#181)
- Add `__eq__` method to NoiseModel, QuantumError, ReadoutError (#181)
- Add support for labelled gates in noise models (#175).
- Add optimized mcx, mcy, mcz, mcu1, mcu2, mcu3, gates to QubitVector (#124)
- Add optimized controlled-swap gate to QubitVector (#142)
- Add gate-fusion optimization for QasmContoroller, which is enabled by setting fusion_enable=true (#136)
- Add better management of failed simulations (#167)
- Add qubits truncate optimization for unused qubits (#164)
- Add ability to disable depolarizing error on device noise model (#131)
- Add initialise simulator instruction to statevector_state (#117, #137)
- Add coupling maps to simulators (#93)
- Add circuit optimization framework (#83)
- Add benchmarking (#71, #177)
- Add wheels support for Debian-like distributions (#69)
- Add autoconfiguration of threads for qasm simulator (#61)
- Add Simulation method based on Stabilizer Rank Decompositions (#51)

Changed
-------
- Add basis_gates kwarg to NoiseModel init (#175).
- Depreciated "initial_statevector" backend option for QasmSimulator and StatevectorSimulator (#185)
- Rename "chop_threshold" backend option to "zero_threshold" and change default value to 1e-10 (#185).
- Add an optional parameter to `NoiseModel.as_dict()` for returning dictionaries that can be
  serialized using the standard `json` library directly. (#165)
- Refactor thread management (#50)

Removed
-------

Fixed
-----
- Improve noise transformations (#162)
- Improve error reporting (#160)
- Improve efficiency of parallelization with max_memory_mb a new parameter of backend_opts (#61)
- Improve u1 performance in statevector (#123)
- Fix OpenMP clashing problems on MacOS for the Terra Addon (#46)

`0.1.1`_ - 2019-01-24
=====================

Added
-----
- Adds version information when using the standalone simulator (#36)
- Adds a Clifford stabilizer simulation method to the QasmSimulator (#13)
- Improve Circuit and NoiseModel instructions checking (#31)
- Add reset_error function to Noise models (#34)
- Improve error reporting at installation time (#29)
- Validate n_qubits before execution (#24)
- Add qobj method to AerJob (#19)

Removed
-------
- Reference model tests removed from the codebase (#27)

Fixed
-----
- Fix Contributing guide (#33)
- Fix an import in Terra integration tests (#33)
- Fix non-OpenMP builds (#19)



`0.1.0`_ - 2018-12-19
=====================

Added
-----
- QASM Simulator
- Statevector Simulator
- Unitary Simulator
- Noise models
- Terra integration
- Standalone Simulators support


.. _UNRELEASED: https://github.com/Qiskit/qiskit-aer/compare/0.2.1...HEAD
.. _0.2.1: https://github.com/Qiskit/qiskit-aer/compare/0.2.0...0.2.1
.. _0.2.0: https://github.com/Qiskit/qiskit-aer/compare/0.1.1...0.2.0
.. _0.1.1: https://github.com/Qiskit/qiskit-aer/compare/0.1.0...0.1.1
.. _0.1.0: https://github.com/Qiskit/qiskit-aer/compare/0.0.0...0.1.0

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
