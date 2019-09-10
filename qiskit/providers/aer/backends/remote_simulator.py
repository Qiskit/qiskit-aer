# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name, bad-continuation

"""RemoteSimulator module

This module is used for connecting to Remote Simulator
"""
import logging
import random
import copy

from collections import Counter
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.jobstatus import JobStatus
from .aerbackend import AerBackend
from .remote_node import RemoteNode
from ..version import __version__
from ..aererror import AerError

logger = logging.getLogger(__name__)


class RemoteSimulator(AerBackend):
    """ RemoteSimulator class """

    def __init__(self, configuration=None, provider=None,
                 kwargs=None):
        """RemoteSimulator for a distributed environment.
        This module is a pseudo simulator for qiskit aer.:

        Args:
            configuration (BackendConfiguration): backend configuration.
            provider (AerProvider): provider responsible for this backend.
            kwargs (string) : Options for the simulator.

        Raises:
            AerError: If Job submission error occurs or No return results
        """

        self._config = configuration
        self._provider = provider
        self._gpu = False
        self._connect_config = None
        self._protocol = None

        if "http_hosts" in kwargs:
            self._host_list = kwargs["http_hosts"]
            self._protocol = "http"

        if "ssh_hosts" in kwargs:
            self._host_list = kwargs["ssh_hosts"]
            self._protocol = "ssh"

        if "ssh_configs" in kwargs:            
            self._connect_config = kwargs["ssh_configs"]
            del kwargs["ssh_configs"]

        self._nodelist = []

        if self._host_list:
            self._get_config(self._host_list, self._protocol, self._connect_config)

        super().__init__(None, self._config, self._provider, self)

    def run_job(self, qobj, noise):
        """ Submit Job to the remote nodes.
        If there is no seed in qobj, Set seed.

        Args:
            qobj(Qobj) : Submittion qobj
            noise (boolean) : Noise simulation or not

        Returns:
            dict: List of Submission information from nodes
        """
        submit_info_list = []
        shots = qobj["config"]["shots"]

        if noise and shots > 1 and len(self._nodelist) > 1:
            qobj_config = qobj["config"]
            seed = random.randint(1, 65536)

            if "seed" not in qobj_config:
                qobj["config"]["seed"] = seed

            if "seed_simulator" not in qobj_config:
                qobj["config"]["seed_simulator"] = seed

            qobj_list = self._gen_qobj_for_map(qobj, shots, len(self._nodelist))

            print("remote run job")
            for index, each_qobj in enumerate(qobj_list):
                submit = self._job_submit(self._nodelist[index], each_qobj)
                print("remote run")
                print(submit)
                submit_info_list.append(submit)
                print("list append")
        else:
            # Always Select First Node without noise (Temporary Solution)
            node = self._nodelist[0]
            submit_info_list.append(self._job_submit(node, qobj))

        return submit_info_list

    def get_job(self, job_id, node_status):
        """
        Get job results from nodes and reduce them.

        Args:
            job_id (string): Job ID
            node_status (AerNodeStatus): List of each node status
        Returns:
            Qobj: Result
        Raises:
            AerError: Not receive results
        """
        node_qobj = []
        max_total_time = 0
        circ_time_list = []
        # find the maximum simulation time and set it as time_taken value
        for node_index, node in enumerate(node_status):
            qobj = node._remote_node.get_job(node._job_id)
            if qobj is not None:
                total_time = qobj["qObjectResult"]["metadata"]["time_taken"]
                circ_results = qobj["qObjectResult"]["results"]

                for circ_index, circ in enumerate(circ_results):
                    circ_time = circ["time_taken"]
                    if node_index == 0:
                        circ_time_list.append([circ_time])
                    else:
                        each_circ_time = circ_time_list[circ_index]
                        each_circ_time.append(circ_time)

                if max_total_time < total_time:
                    max_total_time = total_time
                node_qobj.append(qobj)
            else:
                raise AerError("Can't get result from remote node")

        result_qobj = self._generate_result_qobj(job_id, node_qobj, max_total_time, circ_time_list)
        return result_qobj

    def get_status_job(self, job_id, node_status):
        """
        Get status from each node module.

        If all status of nodes become COMPLETED,
        this module sets return COMPLETED status.

        Args:
            job_id (string) : job id
            node_status (AerNodeStatus): List of each node status

        Returns:
            dict: Job Status
        """
        node_job_status = []
        for node in node_status:
            if node._status is not JobStatus.DONE:
                status = node._remote_node.get_status_job(node._job_id)
                node_job_status.append(status)

        status = "COMPLETED"
        for each_node_status in node_job_status:
            if each_node_status["status"] != "COMPLETED":
                status = "RUNNING"

        return_data = {"id": job_id, "status": status, "node_status": node_job_status}
        return return_data

    def _job_submit(self, node, qobj):
        """
        Submit job to each node.

        Returns:
            dict: submission info
        """
        submit_info = {}
        submit_info["node"] = node
        submit_info["info"] = node.execute_job(qobj)
        return submit_info

    def _gen_qobj_for_map(self, qobj, shots, node_num):
        """
        copy qobj data for each retemo node
        """
        qobj_list = []

        chunk_size, mod = divmod(shots, node_num)
        task_size = [chunk_size] * node_num

        for i in range(mod):
            task_size[i] = task_size[i] + 1

        for shot_size in task_size:
            qobj["config"]["shots"] = shot_size
            qobj_list.append(copy.deepcopy(qobj))

        return qobj_list

    def _generate_result_qobj(self, job_id, node_qobj, max_total_time, circ_time_list):
        """
        Generate results data from each qobj
        """
        first_data = node_qobj[0]
        first_data["id"] = job_id
        first_qobj_result = first_data["qObjectResult"]
        first_qobj_result["job_id"] = job_id
        first_qobj_result["metadata"]["time_taken"] = max_total_time

        shots = []
        count_data = []
        for qobj in node_qobj:
            shot_list = []
            count_list = []

            for circ_result in qobj["qObjectResult"]["results"]:
                count_list.append(circ_result["data"]["counts"])
                shot_list.append(circ_result["shots"])

            shots.append(shot_list)
            count_data.append(count_list)

        for i in range(1, len(shots)):
            shots[0] = [x + y for (x, y) in zip(shots[0], shots[i])]
            for l in range(len(count_data[0])):
                count_data[0][l] = dict(Counter(count_data[0][l]) + Counter(count_data[i][l]))

        for i in range(len(shots[0])):
            _each_result = first_qobj_result["results"]
            _each_result[i]["shots"] = shots[0][i]
            _each_result[i]["data"]["counts"] = count_data[0][i]
            _each_result[i]["time_taken"] = max(circ_time_list[i])

        return first_data

    def _get_config(self, host_list, proto, config):
        """
        Get config from each remote node
        """
        for index, host in enumerate(host_list):
            each_config = None
            if config:
                each_config = config[index]
            self._nodelist.append(RemoteNode(host, proto, each_config))

        self._host_list = host_list
        self._config = self._generate_config()
        self._backend_name = self._config.backend_name

    def _generate_config(self):
        """
        Generate configuration from each node.
        Max shots, n_qubit, max_experiments is set to the smallest value.
        """
        remote_config_list = []

        for node in self._nodelist:
            remote_config_list.append(BackendConfiguration.to_dict(node._config))

        base_config = remote_config_list[0].copy()
        base_config["backend_name"] = "remote_simulator"
        host_name = self._protocol + "_hosts"
        base_config[host_name] = self._host_list

        conditional = True
        open_pulse = True
        memory = True
        allow_q_object = True
        n_qubits = base_config["n_qubits"]
        max_shots = 0
        max_experiments = 0

        for config in remote_config_list:
            if config["conditional"] is False:
                conditional = False

            if n_qubits > config["n_qubits"]:
                n_qubits = config["n_qubits"]

            if config["open_pulse"] is False:
                open_pulse = False

            if config["memory"] is False:
                memory = False

            if config["allow_q_object"] is False:
                allow_q_object = False

            max_shots = max_shots + config["max_shots"]
            max_experiments = max_experiments + config["max_experiments"]

        base_config["conditional"] = conditional
        base_config["open_pulse"] = open_pulse
        base_config["memory"] = memory
        base_config["allow_q_opbject"] = allow_q_object
        base_config["n_qubits"] = n_qubits
        base_config["max_shots"] = max_shots
        base_config["max_experiments"] = max_experiments

        return BackendConfiguration.from_dict(base_config)
