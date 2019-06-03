import subprocess
import json
import os
import pathlib
from subprocess import PIPE
from stateinfo import StateInfo

class SimulationThread():
    def __init__(self, config, state_info, job_id, qobj):
        self.state_info = state_info
        self.job_id = job_id
        self.qobj = qobj
        self.command = config["simulator_command"].split(',')
        #self.cpu_path = config["qiskit_cpu_path"].split(',')

        self.abs_path = pathlib.Path(__file__).resolve().parent
        p = pathlib.Path(self.abs_path/config["qiskit_qobj_path"])
        p.mkdir(exist_ok=True)

        filename = job_id + ".qobj"
        self.qobj_path = p / filename

        with self.qobj_path.open('w') as outfile: 
            json.dump(qobj, outfile, indent=2, sort_keys=True)

        #self.run()

    def run(self):
        
        _command = self.command
        #num_qubit = self.qobj["config"]["n_qubits"]
 
        #if num_qubit >= 20:
        #  command = self.path
        #else:
        #  command = self.cpu_path

        _command.append(str(self.qobj_path))
        try:
            with subprocess.Popen(_command, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
                cin = json.dumps(self.qobj).encode()
                cout, cerr = proc.communicate(cin)
            if proc.returncode:
                print("Simulation ERROR : %s" % proc.returncode)
            
            sim_output = json.loads(cout.decode())
            self.state_info.update_status_job(self.job_id, "COMPLETED")
            self.state_info.set_result(self.job_id, sim_output) 
            #os.remove(self.file_name)

        except FileNotFoundError:
            print("execution file is not found")
