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

"""Utility functions for AerClusterManager."""
import uuid
from typing import Optional, List
from functools import singledispatch, update_wrapper, wraps

from qiskit.qobj import QasmQobj, QasmQobjConfig
from qiskit.providers import JobError


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
    """returns a wrapper function that selects which registered function
    to call based on the type of args[2]"""
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[2].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def split(qobj: QasmQobj, _id: Optional[str] = None) -> List[QasmQobj]:
    """Split a qobj and return a list of qobjs each with a single experiment.

    Args:
        qobj (Qobj): The input qobj object to split
        _id (str): All generated qobjs will have this ID

    Returns:
        A list of qobjs.
    """
    if qobj.type == 'PULSE':
        return None
    else:
        return _split_qasm_qobj(qobj, _id)


def _split_qasm_qobj(qobj: QasmQobj, _id: Optional[str] = None):
    qobjs = []
    if len(qobj.experiments) <= 1:
        return [qobj]
    elif getattr(qobj.config, 'parameterizations', None):
        params = getattr(qobj.config, 'parameterizations', None)
        delattr(qobj.config, 'parameterizations')
        for exp, par in zip(qobj.experiments, params):
            _qid = _id or str(uuid.uuid4())
            _config = QasmQobjConfig(parameterizations=[par], **qobj.config.__dict__)
            qobjs.append(QasmQobj(_qid, _config, [exp], qobj.header))
    else:
        for exp in qobj.experiments:
            _qid = _id or str(uuid.uuid4())
            qobjs.append(QasmQobj(_qid, qobj.config, [exp], qobj.header))
    return qobjs
