# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for Aer job management."""
import uuid
import copy
from math import ceil
from functools import singledispatch, update_wrapper, wraps
from concurrent.futures import ThreadPoolExecutor
from qiskit.qobj import QasmQobj

from qiskit.providers import JobError

DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def requires_submit(func):
    """
    Decorator to ensure that a submit has been performed before
    calling the method.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        if self._future is None:
            raise JobError("Job not submitted yet!. You have to .submit() first!")
        return func(self, *args, **kwargs)
    return _wrapper


def methdispatch(func):
    """
    Returns a wrapper function that selects which registered function
    to call based on the type of args[2]
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[2].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def _copy_qobj_for_noise(qobj, max_shot_size, qobj_id):

    num_shot_jobs, shot_mod = divmod(qobj.config.shots, max_shot_size)
    qobj_list = []

    if shot_mod > 0:
        setattr(qobj.config, "shots", shot_mod)
        #print("first", qobj)
        qobj_list.append(qobj)

    if num_shot_jobs > 1:
        _qid = qobj_id or str(uuid.uuid4())
        _config = copy.deepcopy(qobj.config)
        setattr(_config, "shots", max_shot_size)
        experiment_list = []
        for experiment in qobj.experiments:
            _id = str(uuid.uuid4())
            for _ in range(num_shot_jobs):
                cpy_exp = copy.deepcopy(experiment)
                #print("hogehogheoh"+str(i))
                setattr(cpy_exp.header, "metadata", {"id": _id})
                experiment_list.append(cpy_exp)
        qobj_list.append(QasmQobj(_qid, _config, experiment_list, qobj.header))

    return qobj_list

def _split_qobj(qobj, max_size, qobj_id):
    # Check if we don't need to split
    if max_size is None or not max_size > 0:
        return qobj

    num_jobs = ceil(len(qobj.experiments) / max_size)
    print("num_jobs", num_jobs)
    if num_jobs == 1:
        return qobj

    qobjs = []
    # Check for parameterizations
    params = getattr(qobj.config, 'parameterizations', None)
    for i in range(num_jobs):
        sub_id = qobj_id or str(uuid.uuid4())
        indices = slice(i * max_size, (i + 1) * max_size)
        sub_exp = qobj.experiments[indices]
        sub_config = qobj.config
        if params is not None:
            sub_config = copy.copy(qobj.config)
            sub_config.parameterizations = params[indices]
        qobjs.append(type(qobj)(sub_id, sub_config, sub_exp, qobj.header))
    return qobjs

def split_qobj(qobj, max_size=None, max_shot_size=None, qobj_id=None):
    """Split a qobj and return a list of qobjs each with a single experiment.

    Args:
        qobj (Qobj): The input qobj object to split
        max_size (int or None): the maximum number of circuits per job. If
            None don't split (Default: None).
        qobj_id (str): Optional, set a fixed qobj ID for all subjob qobjs.

    Returns:
        List: A list of qobjs.
    """
    qobj_list = None
    split_qobj_list = []
    if hasattr(qobj.config, "noise_model"):
        if max_shot_size is not None and max_shot_size > 0:
            _qobj = _copy_qobj_for_noise(qobj, max_shot_size, qobj_id)
            if isinstance(_qobj, list):
                qobj_list = _qobj

    if not qobj_list:
        _qobj = _split_qobj(qobj, max_size, qobj_id)
        if isinstance(_qobj, QasmQobj):
            return _qobj
        else:
            split_qobj_list.append(_qobj)
    else:
        for each_qobj in qobj_list:
            _qobj = _split_qobj(each_qobj, max_size, qobj_id)
            if isinstance(_qobj, QasmQobj):
                split_qobj_list.append([_qobj])
            else:
                split_qobj_list.append(_qobj)

    return split_qobj_list

