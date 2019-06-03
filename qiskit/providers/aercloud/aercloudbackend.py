# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""AerCloud module

This module is used for connecting to On-premise Simulator
"""
import logging

from marshmallow import ValidationError
from qiskit import QiskitError
from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.ibmq.utils import update_qobj_config
from qiskit.qobj import QobjConfig
from .api import ApiError
from .aercloudjob import AerCloudJob
from .api import AerCloudConnector
import json

logger = logging.getLogger(__name__)

URL = "http://localhost:5000"


class AerCloudBackend(BaseBackend):
    """
    Backned for Aer Cloud
    """

    def __init__(self, configuration=None, provider=None, kwargs=None):

        if "http_hosts" in kwargs:
            connect_url = kwargs["http_hosts"][0]
        else:
            connect_url = URL

        self._api = AerCloudConnector(connect_url)
        self._provider = provider
        self._url = connect_url

        _raw_config = self._api.available_backends()
        raw_config = _raw_config[0]

        try:
            config = BackendConfiguration.from_dict(raw_config)
        except ValidationError as ex: 
               logger.warning( 
                   'Remote backend "%s" could not be instantiated due to an '
                   'invalid config: %s',
                   raw_config.get('backend_name', 
                   raw_config.get('name', 'unknown')),
                   ex)

        self._backend_name = config.backend_name
        self._config = config
        self._api.config = config
        super().__init__(config)
        return 

    def run(self, qobj, noise_model=None):

        if noise_model:
            qobj = update_qobj_config(qobj, None, noise_model)

            #qobj_config = qobj.config
            #config = qobj_config.as_dict()
            #config["noise_model"] = noise_model
            #qobj.config = QobjConfig.from_dict(config)

        job_class = _job_class_from_backend_support(self)
        job = job_class(self, None, self._api,
                        not self.configuration().simulator, qobj=qobj)
        job.submit()
        return job

    def properties(self):
        return None
        #raise AerCloudBackendValueError("Not Support")

    def status(self):
        raise AerCloudBackendValueError("Not Support")

    def retrieve_job(self, job_id):
        try:
            job_info = self._api.get_status_job(job_id)
            if 'error' in job_info:
                raise AerCloudBackendError('Failed to get job "{}": {}'
                                       .format(job_id, job_info['error']))
        except ApiError as ex:
            raise AerCloudBackendError('Failed to get job "{}":{}'
                                   .format(job_id, str(ex)))
        job_class = _job_class_from_job_response(job_info)

        is_device = not bool(self.configuration().simulator)
        job = job_class(self, job_info.get('id'), self._api, is_device,
                        creation_date=job_info.get('creationDate'),
                        api_status=job_info.get('status'))
        return job

    def __repr__(self):
        credentials_info = ''
        return "<{}('{}') from AerCloud({})>".format(
            self.__class__.__name__, self.name(), credentials_info)


class AerCloudBackendError(QiskitError):
    """
    Backned Error
    """
    pass


class AerCloudBackendValueError(AerCloudBackendError, ValueError):
    """
    Backned Value Error
    """
    pass


def _job_class_from_job_response(job_response):
    is_qobj = job_response.get('kind', None) == 'q-object'
    if not is_qobj: 
        raise AerCloudBackendError('Support only qobj')
    else:
        return AerCloudJob


def _job_class_from_backend_support(backend):
    support_qobj = getattr(backend.configuration(), 'allow_q_object', False)
    if not support_qobj: 
        raise AerCloudBackendError('Support only qobj')
    else:
        return AerCloudJob
