from distutils.core import setup
from Cython.Build import cythonize

import sys
import os
sys.path.append(os.path.abspath('./src/simulators'))
from simulator_extension import simulator_extension

# Simulator extension
package_name = 'aer.backends.aer_qv_wrapper'
source_files = [os.path.abspath('src/simulators/qubitvector/qv_wrapper.pyx')]
include_dirs = [os.path.abspath('./src')]

simulator = simulator_extension(package_name, source_files, include_dirs=include_dirs)

setup(
    name=package_name,
    packages=[package_name],
    ext_modules=cythonize(simulator)
)
