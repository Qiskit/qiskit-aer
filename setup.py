import os
try:
    from skbuild import setup
    dummy_install = False
except:
    print(""" WARNING
              =======
              scikit-build package is needed to build Aer sources.
              Please, install scikit-build and reinstall Aer:
              pip install -I qiskit-aer """)
    from setuptools import setup
    dummy_install = True
from setuptools import find_packages

requirements = [
    "numpy>=1.13"
]

VERSION_PATH = os.path.join(os.path.dirname(__file__),
                            "qiskit", "providers", "aer", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()


def find_qiskit_aer_packages():
    location = 'qiskit/providers'
    prefix = 'qiskit.providers'
    aer_packages = find_packages(where=location)
    pkg_list = list(
        map(lambda package_name: '{}.{}'.format(prefix, package_name),
            aer_packages)
    )
    return pkg_list


setup(
    name='qiskit-aer',
    version=VERSION,
    packages=find_qiskit_aer_packages() if not dummy_install else [],
    cmake_source_dir='.',
    description="Qiskit Aer - High performance simulators for Qiskit",
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
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=requirements,
    include_package_data=True,
    keywords="qiskit aer simulator quantum addon backend"
)
