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


def split_qobj(qobj, max_size=None, qobj_id=None):
    """Split a qobj and return a list of qobjs each with a single experiment.

    Args:
        qobj (Qobj): The input qobj object to split
        max_size (int or None): the maximum number of circuits per job. If
            None don't split (Default: None).
        qobj_id (str): Optional, set a fixed qobj ID for all subjob qobjs.

    Returns:
        List: A list of qobjs.
    """
    # Check if we don't need to split
    if max_size is None or not max_size > 0:
        return qobj
    num_jobs = ceil(len(qobj.experiments) / max_size)
    if num_jobs == 1:
        return qobj

    # Check for parameterizations
    params = getattr(qobj.config, 'parameterizations', None)
    qobjs = []
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
