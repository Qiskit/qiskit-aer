import json 
import os
import pathlib
import datetime

class StateInfo:
    def __init__(self, config):

        self.abs_path = pathlib.Path(__file__).resolve().parent
        self.job_path= pathlib.Path(self.abs_path/config["qiskit_job_path"])
        self.result_path= pathlib.Path(self.abs_path/config["qiskit_result_path"])

        self.job_path.mkdir(exist_ok=True)
        self.result_path.mkdir(exist_ok=True)

    def create_job(self, job_id, create_time): 

        post_data = { "id" : job_id, 
                      "status" : "RUNNING",
                      "creationData" : create_time
        }

        filename = job_id + ".json"
        job_file_path = self.job_path / filename

        with job_file_path.open('w') as outfile: 
            json.dump(post_data, outfile, indent=2, sort_keys=True)

    def update_status_job(self, job_id, status):
        filename = job_id + ".json"
        job_file_path = self.job_path / filename

        if not job_file_path.exists():
            print("Cannot find job file ", str(job_file_path))
            return None

        job_data = None

        with job_file_path.open('r') as outfile: 
            job_data = json.load(outfile)

        job_data["status"] = status
        with job_file_path.open('w') as outfile: 
            json.dump(job_data, outfile, indent=2, sort_keys=True)
             
    def set_result(self, job_id, result):

        result["backend_name"] = "remote_qasm_simulator"
        result["backend_version"] = "0.3.0"
        result["date"] = datetime.datetime.now().isoformat()
        result["job_id"] = job_id
        post_data  = { "id" : job_id, 
                       "qObjectResult" : result}

        filename = job_id + ".json"
        result_file_path = self.result_path / filename

        with result_file_path.open('w') as outfile: 
            json.dump(post_data, outfile, indent=2, sort_keys=True)

    def get_job_status(self, job_id): 

        filename = job_id + ".json"
        job_file_path = self.job_path / filename

        if not job_file_path.exists():
            print("Cannot find job file ", str(job_file_path))
            return None

        job_data = None

        with job_file_path.open('r') as outfile: 
            job_data = json.load(outfile)

        return job_data

    def get_result(self, job_id):
        filename = job_id + ".json"
        result_file_path = self.result_path / filename

        if not result_file_path.exists():
            print("Cannot find result file ", str(result_file_path))
            return

        result_data = None
        with result_file_path.open('r') as outfile:
            result_data = json.load(outfile)

        return result_data
