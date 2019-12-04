#!/usr/bin/env python
# coding: utf-8

# ##  Open Pulse Simulator - Rabi Example

# If running the openpulse branch from source, may need to add a symbolic link in site-packages. Also run `python setup.py build_ext --inplace` in openpulse first.

# In[1]:


#Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Import Qiskit classes classes
import qiskit
#from qiskit.providers.aer.noise import NoiseModel
#from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error

import qiskit.pulse as pulse
import qiskit.pulse.pulse_lib as pulse_lib
from qiskit.compiler import assemble
import random
from qiskit.test.mock import FakeOpenPulse2Q


# In[2]:


#Get a pulse configuration from the fake backend
backend_real = FakeOpenPulse2Q()
configuration = backend_real.configuration()
system = pulse.PulseChannelSpec.from_backend(backend_real)


# In[3]:


#Get pulse simulator backend
backend_sim = qiskit.Aer.get_backend('pulse_simulator')


# ## Simulations
# 
# ### These need to be run in sequential order
# 
# [Rabi Oscillation](#rabi)
# Apply a pulse to Q0 and measure the population evoluation versus pulse amplitude
# 
# [Measurement Level 1 for the Readout Test](#readout_test)
# <br> Prepare the |0> and |1> states and look at the measurement level 1 output 
# 
# [CR Oscillation](#cr)
# <br> Look at the Q1 State when we drive Q0 at the frequency of Q1
# 
# [T1](#t1)
# <br>Demonstrating noise with pulse

# <a id='rabi'></a>
# # Simulate a Rabi Oscillation

# First, we are going to simulate a rabi oscillation, i.e., meausure the qubit state population versus the amplitude of a drive pulse. This is the same example that is in the tutorials.

# ### Build Pulse Schedule 

# Build on qubit 0

# In[4]:


#qubit to use for exeperiment
qubit = 0
# exp configuration
exps = 41
shots = 512

# Rabi pulse
drive_amps = np.linspace(0, 0.9, exps)
drive_samples = 128
drive_sigma = 16

# Measurement pulse
meas_amp = 0.025
meas_samples = 1200
meas_sigma = 4
meas_risefall = 25

# Measurement pulse (common for all experiment)
meas_pulse = pulse_lib.gaussian_square(duration=meas_samples, amp=meas_amp,
                                       sigma=meas_sigma, risefall=meas_risefall, 
                                       name='meas_pulse')
acq_cmd=pulse.Acquire(duration=meas_samples)

# create measurement schedule
measure_and_acquire = meas_pulse(system.qubits[qubit].measure) | acq_cmd(system.acquires, system.memoryslots)

# Create schedule
schedules = []
for ii, drive_amp in enumerate(drive_amps):
    # drive pulse
    rabi_pulse = pulse_lib.gaussian(duration=drive_samples, 
                                    amp=drive_amp, 
                                    sigma=drive_sigma, name='rabi_pulse_%d' % ii)
    
    # add commands to schedule
    schedule = pulse.Schedule(name='rabi_exp_amp_%s' % drive_amp)
    
    schedule += rabi_pulse(system.qubits[qubit].drive)
    schedule += measure_and_acquire << schedule.duration
 
    schedules.append(schedule)


# ### Design the Hamiltonian

# Let's build a transmon Hamiltonian with anharmonicity to test the Rabi oscillation and CR

# In[5]:


hamiltonian = {}
hamiltonian['h_str'] = []
#Q0 terms
hamiltonian['h_str'].append('np.pi*(2*v0-alpha0)*O0')
hamiltonian['h_str'].append('np.pi*alpha0*O0*O0')
hamiltonian['h_str'].append('2*np.pi*r*X0||D0')
hamiltonian['h_str'].append('2*np.pi*r*X0||U1')
hamiltonian['h_str'].append('2*np.pi*r*X1||U0')

#Q1 terms
hamiltonian['h_str'].append('np.pi*(2*v1-alpha1)*O1')
hamiltonian['h_str'].append('np.pi*alpha1*O1*O1')
hamiltonian['h_str'].append('2*np.pi*r*X1||D1')

#Exchange coupling betwene Q0 and Q1
hamiltonian['h_str'].append('2*np.pi*j*(Sp0*Sm1+Sm0*Sp1)')
hamiltonian['vars'] =  {'v0': 5.00, 'v1': 5.1, 'j': 0.01, 
                        'r': 0.02, 'alpha0': -0.33, 'alpha1': -0.33}

#set the qubit dimensions to 3
hamiltonian['qub'] = {'0': 3, '1': 3}


# ### Setup backend_options for simulation

# First copy the real-device configuration to our backend options, then override the Hamiltonian with our custom Hamiltonian, and add any additional solver options.

# In[6]:


backend_options = configuration.to_dict()
backend_options['hamiltonian'] = hamiltonian
backend_options['dt'] = 1.0
backend_options['ode_options'] = {}
backend_options['use_cpp_ode_func'] = False


# ### Restrict the Qubits Used in the Simulation 

# We can use a qubit whitelist (`qubit_list`) to restrict the set of qubits used in the solution. The pulse simulator will appropriately alter the Hamiltonian. To start let's assume the list contains the first 2 qubits.

# In[7]:


backend_options['qubit_list'] = [0, 1]
#backend_options['qubit_list'] = None


# ### Assemble the qobj with the backend config file and the qubit_list

# We have to do this step twice to get the dressed frequencies for setting the LO's. Note here that we set `meas_level=1` and `meas_return=avg` which will return the average probability for the qubit to be in the |1> state.

# In[8]:


rabi_qobj_no_lo = assemble(schedules, backend_real, 
                           meas_level=1, meas_return='avg', 
                           memory_slots=2,
                           shots=shots)


# In[9]:


evals, estates = backend_sim.get_dressed_energies(rabi_qobj_no_lo,
                                                  backend_options=backend_options)


# In[10]:


evals/2/np.pi


# In[11]:


qubit_lo_freq = [evals[1]/2/np.pi, evals[3]/2/np.pi]

rabi_qobj = assemble(schedules, backend_real, 
                     meas_level=1, meas_return='avg', 
                     memory_slots=2, qubit_lo_freq=qubit_lo_freq,
                     shots=shots)


# ### Simulate

# In[12]:


#Note: this is how to run bypassing the backend
#opsys = qiskit.providers.aer.openpulse.qobj.digest.digest_pulse_obj(rabi_qobj.to_dict())
#simdata = qiskit.providers.aer.openpulse.solver.opsolve.opsolve(opsys)


# In[13]:


sim_result = backend_sim.run(rabi_qobj, backend_options=backend_options).result()


# In[14]:


#get the end time of the simulation in dt
sim_result.results[0].header.ode_t


# In[15]:


#get the statevector IN THE FRAME OF THE ORIGINAL HAMILTONIAN
sim_result.get_statevector(0)


# Extract the qubit populations 

# In[16]:


amp_data_Q0 = []
amp_data_Q1 = []

for exp_idx in range(len(drive_amps)):
    exp_mem = sim_result.get_memory(exp_idx)
    amp_data_Q0.append(np.abs(exp_mem[0]))
    amp_data_Q1.append(np.abs(exp_mem[1]))


# In[17]:


#Fit the data
fit_func = lambda x,A,B,T,phi: (A*np.cos(2*np.pi*x/T+phi)+B)
fitparams, conv = curve_fit(fit_func, drive_amps, amp_data_Q0, [0.5,0.5,0.6,1.5])

#get the pi amplitude
pi_amp = (fitparams[3])*fitparams[2]/2/np.pi

plt.plot(drive_amps, amp_data_Q0, label='Q0')
plt.plot(drive_amps, amp_data_Q1, label='Q1')
plt.plot(drive_amps, fit_func(drive_amps, *fitparams), color='black', linestyle='dashed', label='Fit')
plt.axvline(pi_amp, color='black', linestyle='dashed')
plt.legend()
plt.xlabel('Pulse amplitude, a.u.', fontsize=20)
plt.ylabel('Signal, a.u.', fontsize=20)
plt.title('Rabi on Q0', fontsize=20)
plt.grid(True)

print('Pi Amplitude %f'%(pi_amp))

