import json
from flask import Flask, request, jsonify
from requests import post, get
from time import sleep
import uuid
import datetime
import threading 
import configparser
from simulationthread import SimulationThread
from stateinfo import StateInfo
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)

DEFAULT_CONFIGURATION = {
    'allow_q_object': True,
    'backend_name': 'qasm_simulator', 
    'backend_version': '0.1.547',
    'n_qubits': 32,
    'simulator': True,
    'conditional': True,
    'open_pulse': False,
    'memory': False,
    'local': False,
    'max_shots': 8192,
    'max_experiments': 1024,
    'basis_gates': ['u1', 'u2', 'u3', 'cx'],
    'gates': [
        {
            'name': 'u1',
            'parameters': ['lambda'],
            'qasm_def': 'gate u1(lambda) q { U(0,0,lambda) q; }'
        },
        {
            'name': 'u2',
            'parameters': ['phi', 'lambda'],
            'qasm_def': 'gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }'
        },
        {
            'name': 'u3',
            'parameters': ['theta', 'phi', 'lambda'],
            'qasm_def': 'gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }'
        },
        {
            'name': 'cx',
            'parameters': ['c', 't'],
            'qasm_def': 'gate cx c,t { CX c,t; }'
        }
    ]
}

lock = threading.Lock()
config = configparser.ConfigParser()
config.read("server.ini")
thread_num = int(config["simulator"]["thread_num"])
_port = int(config["flask"]["port"])
state_info = StateInfo(config["simulator"])

executor = ThreadPoolExecutor(thread_num)

@app.route("/users/loginWithToken", methods=['POST'])
def user_login(): 
    path = "/users/loginWithToken"
    dummy_id = {"id":"dummy","ttl":1209600,"created":datetime.datetime.today(),"userId":"dummy"}

    return jsonify(dummy_id)

@app.route("/Backends/qasm_simulator/defaults", methods=['GET'])
def get_default_backend():
    backends = []
    backends.append(DEFAULT_CONFIGURATION)

    return jsonify(backends)

@app.route("/Backends/v/1", methods=['GET'])
def get_backend():
    path = "/Backends/v/1"
    backends = []
    backends.append(DEFAULT_CONFIGURATION)

    return jsonify(backends)

@app.route("/Jobs/<job_id>/status", methods=['GET'])
def get_job(job_id):

    job_status = state_info.get_job_status(job_id)
    if job_status:
        return jsonify(job_status)
    else:
        print("no job ", job_id)
        return 404

@app.route("/Jobs/<job_id>", methods=['GET'])
def get_result(job_id):

    job_result = state_info.get_result(job_id)
    if job_result:
        return jsonify(job_result)
    else:
        print("no result ", job_id)
        return 404

def create_job_data(qObject):
    shot_num = qObject["qObject"]["config"]["shots"]
    job_data = {}
    qasm_data = {}

    qasm_list = []

    qasm_data["status"] = "WORKING_IN_PROGRESS"
    qasm_data["executionId"] = uuid.uuid4().hex
    qasm_list.append(qasm_data)

    job_data["qasms"] = qasm_list
    job_data["qObject"] = qObject["qObject"]
    job_data["backend"] = qObject["backend"]
    job_data["kind"] = "q-object"
    job_data["shots"] =  shot_num
    job_data["status"] = "RUNNING"
    job_data["maxCredits"] = 10
    job_data["userCredits"] = 0

    start_time = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
    job_data["creationData"] = start_time
    job_data["deleted"] = False

    _job_id = 0
    while(True): 
        lock.acquire()
        _job_id = uuid.uuid4().hex 
        lock.release()
        if not state_info.get_job_status(_job_id):
             break

    job_data["id"] = _job_id
    job_data["userId"] = uuid.uuid4().hex

    return job_data

@app.route("/Jobs", methods=['POST'])
def post_job():
    path = "/Jobs"
    post_data = request.json
    backend = post_data["backend"]

    if (backend["name"] == "qasm_simulator"):
        return_data = create_job_data(post_data)
        job_id = return_data["id"]
        state_info.create_job(job_id, return_data["creationData"]) 
        sim_thread = SimulationThread(config["simulator"], state_info, job_id, post_data["qObject"])
        executor.submit(sim_thread.run)
        return jsonify(return_data)
    else:
        print("No backends")
        content = {"backned" : backend["name"]}
        return jsonify(content), 404

if __name__ == '__main__':
    app.run(port=_port)

