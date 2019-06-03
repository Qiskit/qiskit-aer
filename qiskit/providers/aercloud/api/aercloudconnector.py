# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import json
import logging
import re
import time
from urllib import parse

import requests

logger = logging.getLogger(__name__)
CLIENT_APPLICATION = 'qiskit-api-py'


class _Request(object):
    """
    The Request class to manage the methods
    """
    def __init__(self, url, retries=5, timeout_interval=1.0):
        self.client_application = CLIENT_APPLICATION
        self._url = url
        self.errors_not_retry = [401, 403, 413]

        if not isinstance(retries, int):
            raise TypeError('post retries must be positive integer')

        self.retries = retries
        self.timeout_interval = timeout_interval
        self.result = None
        self._max_qubit_error_re = re.compile(
            r".*registers exceed the number of qubits, "
            r"it can\'t be greater than (\d+).*")

    def post(self, path, params='', data=None):
        """
        POST Method Wrapper of the REST API
        """
        self.result = None
        data = data or {}
        headers = {'Content-Type': 'application/json',
                   'x-qx-client-application': self.client_application}

        url = self._url + path + params
        retries = self.retries

        while retries > 0:
            response = requests.post(url, data=data, headers=headers)

            if self._response_good(response):
                if self.result:
                    return self.result
                elif retries < 2:
                    return response.json()
                else:
                    retries -= 1
            else:
                retries -= 1
                time.sleep(self.timeout_interval)

        # timed out
        raise ApiError(usr_msg='Failed to get proper ' +
                       'response from backend.')

    def get(self, path, params=''):
        """
        GET Method Wrapper of the REST API
        """
        self.result = None

        url = self._url + path + params
        retries = self.retries
        headers = {'x-qx-client-application': self.client_application}
        while retries > 0:  # Repeat until no error
            response = requests.get(url, headers=headers)
            if self._response_good(response):
                if self.result:
                    return self.result
                elif retries < 2:
                    return response.json()
                else:
                    retries -= 1
            else:
                retries -= 1
                time.sleep(self.timeout_interval)
        # timed out
        raise ApiError(usr_msg='Failed to get proper ' +
                       'response from backend.')

    def _response_good(self, response):
        """check response

        Args:
            response (requests.Response): HTTP response.

        Returns:
            bool: True if the response is good, else False.

        Raises:
            ApiError: response isn't formatted properly.
        """

        url = parse.urlparse(response.url).path

        if response.status_code != requests.codes.ok:
            logger.warning('Got a %s code response to %s: %s',
                           response.status_code,
                           url,
                           response.text)
            if response.status_code in self.errors_not_retry:
                raise ApiError(usr_msg='Got a {} code response to {}: {}'.format(
                    response.status_code,
                    url,
                    response.text))
            else:
                mobj = self._max_qubit_error_re.match(response.text)
                if mobj:
                    raise RegisterSizeError(
                    'device register size must be <= {}'.format(mobj.group(1)))
                return True
        try:
            if str(response.headers['content-type']).startswith("text/html;"):
                self.result = response.text
                return True
            else:
                self.result = response.json()
        except (json.JSONDecodeError, ValueError):
            usr_msg = 'device server returned unexpected http response'
            dev_msg = usr_msg + ': ' + response.text
            raise ApiError(usr_msg=usr_msg, dev_msg=dev_msg)

        if not isinstance(self.result, (list, dict)):
            msg = ('JSON not a list or dict: url: {0},'
                   'status: {1}, reason: {2}, text: {3}')
            raise ApiError(
                usr_msg=msg.format(url,
                                   response.status_code,
                                   response.reason, response.text))
        if ('error' not in self.result or
                ('status' not in self.result['error'] or
                 self.result['error']['status'] != 400)):
            return True

        logger.warning("Got a 400 code JSON response to %s", url)
        return False

class AerCloudConnector(object):
    """
    Connector for Aer Cloud (On-premise Simulator)
    """
    def __init__(self, url=None):
        self._url = url
        self.req = _Request(self._url)
        self.config = None

    def _check_backend(self, backend):
        """
        Check if the name of a backend is valid to run in QX Platform
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

    def run_job(self, job, backend='simulator', shots=1, seed=None):
        """
        Execute a job
        """

        backend_type = self._check_backend(backend)

        if not backend_type:
            raise BadBackendError(backend)

        if isinstance(job, (list, tuple)):
            qasms = job
            for qasm in qasms:
                qasm['qasm'] = qasm['qasm'].replace('IBMQASM 2.0;', '')
                qasm['qasm'] = qasm['qasm'].replace('OPENQASM 2.0;', '')

            data = {'qasms': qasms,
                    'shots': shots,
                    'backend': {}}

            if seed and len(str(seed)) < 11 and str(seed).isdigit():
                data['seed'] = seed
            elif seed:
                return {"error": "Not seed allowed. Max 10 digits."}

            data['backend']['name'] = backend_type
        elif isinstance(job, dict):
            q_obj = job
            data = {'qObject': q_obj,
                    'backend': {}}

            data['backend']['name'] = backend_type
        else:
            return {"error": "Not a valid data to send"}

        url = "/Jobs"

        job = self.req.post(url, data=json.dumps(data))

        return job

    def get_job(self, id_job):
        """
        Get the information about a job, by its id
        """
        if not id_job:
            return {'status': 'Error',
                    'error': 'Job ID not specified'}

        url = "/Jobs"

        url += '/' + id_job

        job = self.req.get(url)

        if 'qObjectResult' in job:
            # If the job is using Qobj, return the qObjectResult directly,
            # which should contain a valid Result.
            return job
        elif 'qasms' in job:
            # Fallback for pre-Qobj jobs.
            for qasm in job['qasms']:
                if ('result' in qasm) and ('data' in qasm['result']):
                    qasm['data'] = qasm['result']['data']
                    del qasm['result']['data']
                    for key in qasm['result']:
                        qasm['data'][key] = qasm['result'][key]
                    del qasm['result']

        return job

    def get_status_job(self, id_job):
        """
        Get the status about a job, by its id
        """
        if not id_job:
            return {'status': 'Error',
                    'error': 'Job ID not specified'}

        url = "/Jobs"

        url += '/' + id_job + '/status'

        status = self.req.get(url)

        return status

    def available_backends(self):
        """
        Get the backends available to use in the QX Platform
        """
        response = self.req.get('/Backends/v/1')
        if (response is not None) and (isinstance(response, dict)):
            return []

        return response


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
