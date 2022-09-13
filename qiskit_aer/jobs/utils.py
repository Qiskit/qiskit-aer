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
from qiskit.qobj import QasmQobj, PulseQobj

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

    if shot_mod == 0 and num_shot_jobs == 1:
        return qobj

    if shot_mod > 0:
        qobj.config.shots = shot_mod
        for experiment in qobj.experiments:
            _id = str(uuid.uuid4())
            experiment.header.metadata["id"] = _id
        qobj_list.append(qobj)

    if num_shot_jobs > 1:
        _qid = qobj_id or str(uuid.uuid4())
        _config = copy.copy(qobj.config)
        setattr(_config, "shots", max_shot_size)
        experiment_list = []
        for experiment in qobj.experiments:
            _id = str(uuid.uuid4())
            for _ in range(num_shot_jobs):
                cpy_exp = copy.copy(experiment)
                cpy_exp.header = copy.copy(experiment.header)
                cpy_exp.header.metadata["id"] = _id
                experiment_list.append(cpy_exp)
        qobj_list.append(QasmQobj(_qid, _config, experiment_list, qobj.header))

    return qobj_list


def _split_qobj(qobj, max_size, qobj_id, seed):
    # Check if we don't need to split
    if max_size is None or not max_size > 0:
        return qobj

    num_jobs = ceil(len(qobj.experiments) / max_size)
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
            sub_config.parameterizations = params[indices]
            sub_config = copy.copy(qobj.config)

        if seed > 0:
            if sub_config is qobj.config:
                sub_config = copy.copy(qobj.config)

        qobjs.append(type(qobj)(sub_id, sub_config, sub_exp, qobj.header))

    return qobjs


def _check_custom_instruction(experiments, optypes=None):
    """Return True if circuits contain instructions that cant be split"""
    # Check via optype list if available
    if optypes is not None:
        # Optypes store class names as strings
        return any(
            {"SaveData"}.intersection(optype)
            for optype in optypes
        )

    # Otherwise iterate over instruction names
    return any(
        "save_" in inst.name
        for exp in experiments for inst in exp.instructions
    )


def _set_seed(qobj_list, seed):

    # set seed number to each qobj
    seed_shift = 256

    if seed == 0:
        return

    for _each_qobj_list in qobj_list:
        for _each_qobj in _each_qobj_list:
            _each_qobj.config.seed_simulator = seed
            seed = seed + seed_shift


def split_qobj(qobj, max_size=None, max_shot_size=None, qobj_id=None):
    """Split a qobj and return a list of qobjs each with a single experiment.

    Args:
        qobj (Qobj): The input qobj object to split
        max_size (int or None): the maximum number of circuits per job. If
            None don't split (Default: None).
        max_shot_size (int or None): the maximum number of shots per job. If
            None don't split (Default: None).
        qobj_id (str): Optional, set a fixed qobj ID for all subjob qobjs.

    Raises:
        JobError : If max_job_size > 1 and seed is set.
        JobError : If custom instructions exist.

    Returns:
        List: A list of qobjs.
    """
    optypes = getattr(qobj.config, 'optypes', None)
    split_qobj_list = []
    if (max_shot_size is not None and max_shot_size > 0):
        if _check_custom_instruction(qobj.experiments, optypes):
            raise JobError(
                "`max_shot_size` option cannot be used with circuits"
                " containing save instructions.")

    _seed = getattr(qobj.config, "seed_simulator", 0)
    if hasattr(qobj.config, "noise_model"):
        if _seed and max_size is not None and max_size > 1:
            raise JobError("cannot support max_job_size > 1 for noise simulation, "
                           "when seed_simulator is set.")

        if max_shot_size is not None and max_shot_size > 0:
            _qobj = _copy_qobj_for_noise(qobj, max_shot_size, qobj_id)
            if isinstance(_qobj, list):
                for each_qobj in _qobj:
                    _split = _split_qobj(each_qobj, max_size, qobj_id, _seed)
                    if isinstance(_split, QasmQobj):
                        split_qobj_list.append([_split])
                    else:
                        split_qobj_list.append(_split)
                _set_seed(split_qobj_list, _seed)
                return split_qobj_list

    _qobj = _split_qobj(qobj, max_size, qobj_id, _seed)
    if isinstance(_qobj, (PulseQobj, QasmQobj)):
        return _qobj
    else:
        split_qobj_list.append(_qobj)

    _set_seed(split_qobj_list, _seed)
    return split_qobj_list
