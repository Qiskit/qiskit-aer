.. _circuit:

Additional circuit methods
==========================

.. currentmodule:: qiskit.circuit

On import, Aer adds several simulation-specific methods to :class:`~qiskit.circuit.QuantumCircuit` for convenience.
These methods are not available until Aer is imported (``import qiskit_aer``).

Setting a custom simulator state
--------------------------------

The set instructions can also be added to circuits by using the
following ``QuantumCircuit`` methods which are patched when importing Aer.

.. automethod:: QuantumCircuit.set_density_matrix
.. automethod:: QuantumCircuit.set_stabilizer
.. automethod:: QuantumCircuit.set_unitary
.. automethod:: QuantumCircuit.set_superop
.. automethod:: QuantumCircuit.set_matrix_product_state

Saving Simulator Data
--------------------------------

The save instructions can also be added to circuits by using the
following ``QuantumCircuit`` methods which are patched when importing Aer.

.. note ::
  Each save method has a default label for accessing from the
  circuit result data, however duplicate labels in results will result
  in an exception being raised. If you use more than 1 instance of a
  specific save instruction you must set a custom label for the
  additional instructions.

.. automethod:: QuantumCircuit.save_amplitudes
.. automethod:: QuantumCircuit.save_amplitudes_squared
.. automethod:: QuantumCircuit.save_clifford
.. automethod:: QuantumCircuit.save_density_matrix
.. automethod:: QuantumCircuit.save_expectation_value
.. automethod:: QuantumCircuit.save_expectation_value_variance
.. automethod:: QuantumCircuit.save_matrix_product_state
.. automethod:: QuantumCircuit.save_probabilities
.. automethod:: QuantumCircuit.save_probabilities_dict
.. automethod:: QuantumCircuit.save_stabilizer
.. automethod:: QuantumCircuit.save_state
.. automethod:: QuantumCircuit.save_statevector
.. automethod:: QuantumCircuit.save_statevector_dict
.. automethod:: QuantumCircuit.save_superop
.. automethod:: QuantumCircuit.save_unitary
