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
from platform import node
import uuid
import random
import copy
from typing import Optional, List
from functools import singledispatch, update_wrapper, wraps
from qiskit import circuit

<<<<<<< HEAD
from qiskit.qobj import QasmQobj, PulseQobj, QasmQobjConfig
=======
from qiskit.qobj import QasmQobj, QasmQobjConfig
>>>>>>> Add ClusterBackend and related utilities
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


def split(qobj: QasmQobj, _id: Optional[str] = None, noise: bool = False) -> List[QasmQobj]:
    """Split a qobj and return a list of qobjs each with a single experiment.

    Args:
        qobj (Qobj): The input qobj object to split
        _id (str): All generated qobjs will have this ID

    Returns:
        A list of qobjs.
    """
    if qobj.type == 'PULSE':
        return _split_pulse_qobj(qobj, _id)
    else:
        return _split_qasm_qobj(qobj, _id, noise)


def _split_pulse_qobj(qobj: PulseQobj, _id: Optional[str] = None):
    qobjs = []
    if len(qobj.experiments) <= 1:
        return [qobj]
    for exp in qobj.experiments:
        _qid = _id or str(uuid.uuid4())
        _config = copy.deepcopy(qobj.config)
        qobjs.append([{"qobj":QasmQobj(_qid, _config, [exp], qobj.header)}])
    return qobjs


def _split_qasm_qobj(qobj: QasmQobj, _id: Optional[str] = None, noise: bool = False):
    qobjs = []
    if len(qobj.experiments) <= 1:
        return [{"qobj":qobj}]
    elif getattr(qobj.config, 'parameterizations', None):
        params = getattr(qobj.config, 'parameterizations', None)
        delattr(qobj.config, 'parameterizations')
        for exp, par in zip(qobj.experiments, params):
            _qid = _id or str(uuid.uuid4())
            _config = QasmQobjConfig(parameterizations=[par], **qobj.config.__dict__)
            if noise:
                qobjs.append({"qobj":QasmQobj(_qid, _config, [exp], qobj.header)})
            else:
                qobjs.append([{"qobj":QasmQobj(_qid, _config, [exp], qobj.header)}])
    else:
        for exp in qobj.experiments:
            _config = copy.deepcopy(qobj.config)
            _qid = _id or str(uuid.uuid4())
            if noise:
                qobjs.append({"qobj":QasmQobj(_qid, _config, [exp], qobj.header)})
            else:
                qobjs.append([{"qobj":QasmQobj(_qid, _config, [exp], qobj.header)}])
    #print("split", qobjs))
    return qobjs


def copy_qobj_and_options(qobj, shots, seed, node_num, run_option):
    run_options_list = []
    qobj_list = []

    if seed == 0:
        seed = random.randint(0, 0xffffffff)

    for exp in qobj.experiments:
        _qobjs, _options = _copy_qobj_and_opt(qobj, exp, shots, seed, node_num, run_option)
        #print("options", _options)
        qobj_list.append(_qobjs)
        run_options_list.append(_options)

    return qobj_list, run_options_list

def  _copy_qobj_and_opt(qobj, exp, shots, seed, node_num, option):
    option_list = []
    exp_list = []

    chunk_size, mod = divmod(shots, node_num)
    task_size = [chunk_size] * node_num

    _qid = str(uuid.uuid4())

    for i in range(mod):
        task_size[i] = task_size[i] + 1

    for i in range(node_num):
        if task_size[i] > 0:
            _option = option.copy()
            _option["shots"] = task_size[i]
            _option["seed_simulator"] = seed + i
            option_list.append(_option)
            exp_list.append(exp)

    _qobj = QasmQobj(_qid, qobj.config, exp_list, qobj.header)
    return _qobj, option_list

def copy_circuits_and_options(circuits, shots, seed, node_num, run_option):

    run_options_list = []
    circuits_list = []
    if seed == 0:
        seed = random.randint(0, 0xffffffff)

    if isinstance(circuits, list):
        for circ in circuits:
            _circuits, _options = _copy_circ_and_opt(circ, shots, seed, node_num, run_option)
            circuits_list.append(_circuits)
            run_options_list.append(_options)
    else:
        _circuits, _options = _copy_circ_and_opt(circuits, shots, seed, node_num, run_option)
        circuits_list.append(_circuits)
        run_options_list.append(_options)

    return circuits_list, run_options_list

def _copy_circ_and_opt(circ, shots, seed, node_num, option):
    circuits_list = []
    option_list = []

    chunk_size, mod = divmod(shots, node_num)
    task_size = [chunk_size] * node_num

    for i in range(mod):
        task_size[i] = task_size[i] + 1

    for i in range(node_num):
        if task_size[i] > 0:
            circuits_list.append(circ)
            _option = option.copy()
            _option["shots"] = task_size[i]
            _option["seed_simulator"] = seed + i
            option_list.append(_option)

    return circuits_list, option_list
