.. _circuit:

Quantum Circuit (``qiskit.circuit``)
=======================================
.. class:: QuantumCircuit(*regs, name=None, global_phase=0, metadata=None)
   
  Create a new circuit.

  A circuit is a list of instructions bound to some registers. 
 
  .. attribute:: Parameters

    * **regs** (list(:class:`Register`) or list(``int``) or list(list(:class:`Bit`))): The registers to be included in the circuit.

      * If a list of :class:`Register` objects, represents the :class:`QuantumRegister` and/or :class:`ClassicalRegister` objects to include in the circuit.
        For example:

           * :code:`QuantumCircuit(QuantumRegister(4))`
           * :code:`QuantumCircuit(QuantumRegister(4), ClassicalRegister(3))`
           * :code:`QuantumCircuit(QuantumRegister(4, 'qr0'), QuantumRegister(2, 'qr1'))`

      * If a list of ``int``, the amount of qubits and/or classical bits to include in the circuit. It can either be a single int for just the number of quantum bits, or 2 ints for the number of quantum bits and classical bits, respectively.

        For example:

           * :code:`QuantumCircuit(4) # A QuantumCircuit with 4 qubits`
           * :code:`QuantumCircuit(4, 3) # A QuantumCircuit with 4 qubits and 3 classical bits`

      * If a list of python lists containing :class:`Bit` objects, a collection of :class:`Bit` s to be added to the circuit.


    * **name** (*str*): the name of the quantum circuit. If not set, an automatically generated string will be assigned.
    * **global_phase** (*float or ParameterExpression*): The global phase of the circuit in radians.
    * **metadata** (*dict*): Arbitrary key value metadata to associate with the circuit. This gets stored as free-form data in a dict in the :attr:`~qiskit.circuit.QuantumCircuit.metadata` attribute. It will not be directly used in the circuit.

  .. attribute:: Raises

    **CircuitError** – if the circuit name, if given, is not valid.c

  .. rubric:: Methods

  **Setting a Custom Simulator State**

  The set instructions can also be added to circuits by using the
  following ``QuantumCircuit`` methods which are patched when importing Aer.

  .. currentmodule:: qiskit_aer.library

  .. autosummary::
      :toctree: ../stubs/

      set_statevector
      set_density_matrix
      set_stabilizer
      set_unitary
      set_superop
      set_matrix_product_state

  **Saving Simulator Data**

  The save instructions can also be added to circuits by using the
  following ``QuantumCircuit`` methods which are patched when importing Aer.

  .. note ::
    Each save method has a default label for accessing from the
    circuit result data, however duplicate labels in results will result
    in an exception being raised. If you use more than 1 instance of a
    specific save instruction you must set a custom label for the
    additional instructions.

  .. autosummary::
    :toctree: ../stubs/

    save_amplitudes
    save_amplitudes_squared
    save_clifford
    save_density_matrix
    save_expectation_value
    save_expectation_value_variance
    save_matrix_product_state
    save_probabilities
    save_probabilities_dict
    save_stabilizer
    save_state
    save_statevector
    save_statevector_dict
    save_superop
    save_unitary

