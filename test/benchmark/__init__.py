from qiskit.qobj import Qobj
from qiskit.providers.aer.noise import NoiseModel

def qobj_repr_hook(self):
    """ This is needed for ASV to beauty-printing reports """
    return "Qobj<{0} qubits, {1} {2}, {3} shots>".format(
        self.config.n_qubits,
        len(self.experiments),
        'circuits' if self.type == 'QASM' else 'schedules',
        self.config.shots
    )

Qobj.__repr__ = qobj_repr_hook


def noise_model_repr_hook(self):
    """ This is needed for ASV to beauty-printing reports """
    