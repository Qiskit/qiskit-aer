from qiskit.qobj import Qobj
from qiskit.providers.aer.noise import NoiseModel

def qobj_repr_hook(self):
    """ This is needed for ASV to beauty-printing reports """
    return "Num. qubits: {0}".format(self.config.n_qubits)

Qobj.__repr__ = qobj_repr_hook


def noise_model_repr_hook(self):
    """ This is needed for ASV to beauty-printing reports """
    return self.__class__.__name__.replace("_", " ").capitalize()

NoiseModel.__repr__ = noise_model_repr_hook
