import json
import numpy as np


class SimulatorJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        return json.JSONEncoder.default(self, obj)


def qobj2schema(qobj):
    """
    Convert current Qiskit qobj in line with schema spec qobj for simulator testing.
    """
    qobj["type"] = "QASM"
    if "circuits" in qobj:
        qobj["experiments"] = qobj.pop("circuits")
    if "experiments" in qobj:
        for i, experiment in enumerate(qobj["experiments"]):
            # adjust labels
            if "compiled_circuit" in experiment:
                experiment["compiled_circuit"].pop("header", None)
                experiment["instructions"] = experiment.pop("compiled_circuit", {}).pop("operations", [])
                for k, op in enumerate(experiment["instructions"]):
                    if op.get("name") == "measure":
                        op["memory"] = op.pop("clbits")
                        op["register"] = op["memory"]
                        experiment["instructions"][k] = op
            # clear compiled qasm
            experiment.pop("compiled_circuit_qasm", '')
            # clear old header
            if "name" in experiment:
                experiment["header"] = {"name": experiment.pop("name")}
            qobj["experiments"][i] = experiment
    return qobj