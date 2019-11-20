
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.aer import noise

# Choose a real device to simulate
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
device = provider.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map

# Generate an Aer noise model for device
noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates

# Generate a quantum circuit
qc = QuantumCircuit(2, 2)

qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Perform noisy simulation
backend = Aer.get_backend('qasm_simulator')
job_sim = execute(qc, backend,
                  coupling_map=coupling_map,
                  noise_model=noise_model,
                  basis_gates=basis_gates)

sim_result = job_sim.result()
print(sim_result.get_counts(qc))
              