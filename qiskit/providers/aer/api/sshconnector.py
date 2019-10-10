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

# pylint: disable=unbalanced-tuple-unpacking, unused-variable, unused-argument

"""This module implements ssh connector to the remote node."""

import json
import logging
import uuid
import concurrent.futures
import datetime
import paramiko

logger = logging.getLogger(__name__)


class SshConnector:
    """
    Connector for Remote Node via SSH command
    """
    def __init__(self, host=None, connect_config=None):
        print(connect_config)
        self._host = host
        self._results = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        if "ssh_key" in connect_config:
            self._key = connect_config["ssh_key"]

        if "ssh_user" in connect_config:
            self._user = connect_config["ssh_user"]

        if "qobj_path" in connect_config:
            self._qobj_path = connect_config["qobj_path"]

        if "conf_path" in connect_config:
            self._conf_path = connect_config["conf_path"]

        if "run_command" in connect_config:
            self._run_command = connect_config["run_command"]

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def _check_backend(self, backend):
        """
        Check if the name of a backend is valid to run
        """
        # First check against hacks for old backend names
        original_backend = backend
        backend = backend.lower()

        # Check for new-style backends
        backends = self.available_backends()
        for backend_ in backends:
            if backend_.get('backend_name', '') == original_backend:
                return original_backend
        # backend unrecognized
        return None

    def _run_submit(self, qobj, job_id):
        """
        Put qobj file to remote host by scp
        """
        self._client.connect(self._host, username=self._user, key_filename=self._key)
        file_name = self._qobj_path + "/" + job_id + ".qobj"
        sftp = self._client.open_sftp()
        r_file = sftp.open(file_name, "a", -1)
        r_file.write(json.dumps(qobj))
        r_file.flush()
        sftp.close()

        exec_cmd = self._run_command + " " + file_name
        stdin, stdout, stderr = self._client.exec_command(exec_cmd)
        result_string = stdout.read()
        ssh_error = stderr.read()
        self._client.close()

        if ssh_error:
            usr_msg = 'scp failed'
            dev_msg = usr_msg + ":" + ssh_error
            raise ApiError(usr_msg=usr_msg, dev_msg=dev_msg)

        result = json.loads(result_string)
        result["backend_name"] = "remote_qasm_simulator"
        result["backend_version"] = "0.3.0"
        result["date"] = datetime.datetime.now().isoformat()
        result["job_id"] = job_id
        result_json = {"id": job_id, "qObjectResult": result}

        for each_result in self._results:
            if each_result["id"] == job_id:
                each_result["result"] = result_json
                each_result["status"] = "COMPLETED"

    def run_job(self, job, backend='simulator', shots=1, seed=None):
        """
        Execute a job
        """
        backend_type = self._check_backend(backend)

        if not backend_type:
            raise BadBackendError(backend)

        job_id = uuid.uuid1().hex

        run_job_data = {"id": job_id, "status": "RUNNING", "result": None}
        self._results.append(run_job_data)
        self._executor.submit(self._run_submit, job, job_id)
        return_data = {"id": job_id}

        return return_data

    def get_job(self, id_job):
        """
        Get the information about a job, by its id
        """
        if not id_job:
            return {'status': 'Error',
                    'error': 'Job ID not specified'}

        return_data = None
        for index, each_result in enumerate(self._results):
            if each_result["id"] == id_job:
                return_data = each_result["result"]
                self._results.pop(index)

        return return_data

    def get_status_job(self, id_job):
        """
        Get the status about a job, by its id
        """
        if not id_job:
            return {'status': 'Error',
                    'error': 'Job ID not specified'}

        for each_result in self._results:
            if each_result["id"] == id_job:
                return_data = {"id": id_job, "status": each_result["status"]}

        return return_data

    def available_backends(self):
        """
        Get the backends available to use
        """
        exec_cmd = "cat " + self._conf_path

        try:
            self._client.connect(self._host, username=self._user, key_filename=self._key)
        except paramiko.AuthenticationException:
            usr_msg = 'Authentication failed'
            dev_msg = usr_msg
            raise ApiError(usr_msg=usr_msg, dev_msg=dev_msg)

        stdin, stdout, stderr = self._client.exec_command(exec_cmd)
        config_string = stdout.read()
        ssh_error = stderr.read()
        self._client.close()

        if ssh_error:
            usr_msg = 'Command execution failed'
            dev_msg = usr_msg + ":" + ssh_error
            raise ApiError(usr_msg=usr_msg, dev_msg=dev_msg)

        config = json.loads(config_string)
        config_list = [config]

        return config_list


class ApiError(Exception):
    """
    API Error
    """
    def __init__(self, usr_msg=None, dev_msg=None):
        """
        Args:
            usr_msg (str): Short user facing message describing error.
            dev_msg (str or None): More detailed message to assist
                developer with resolving issue.
        """
        Exception.__init__(self, usr_msg)
        self.usr_msg = usr_msg
        self.dev_msg = dev_msg

    def __repr__(self):
        return repr(self.dev_msg)

    def __str__(self):
        return str(self.usr_msg)


class BadBackendError(ApiError):
    """
    Unavailable backend error.
    """
    def __init__(self, backend):
        """
        Args:
            backend (str): name of backend.
        """
        usr_msg = 'Could not find backend "{0}" available.'.format(backend)
        dev_msg = ('Backend "{0}" does not exist. Please use '
                   'available_backends to see options').format(backend)
        ApiError.__init__(self, usr_msg=usr_msg,
                          dev_msg=dev_msg)


class RegisterSizeError(ApiError):
    """Exception due to exceeding the maximum number of allowed qubits."""
    pass
