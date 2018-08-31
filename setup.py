from skbuild import setup

import sys
import os
#sys.path.append(os.path.abspath('./'))
setup(
    name='aer_qv_wrapper',
    packages=['aer_qv_wrapper'],
    version="0.0.1",
    description="QV Quantum Simulator",
    author="AER Development Team",
    author_email="qiskit@us.ibm.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit simulator quantum"
)
