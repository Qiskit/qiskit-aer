from qiskit.qobj import Qobj

def repr_hook(self):
    """ This is needed for beauty printing reports from ASV """
    return "Qobj<{0} qubits, {1} {2}, {3} shots>".format(
        self.config.n_qubits,
        len(self.experiments),
        'circuits' if self.type == 'QASM' else 'schedules',
        self.config.shots
    )

Qobj.__repr__ = repr_hook
