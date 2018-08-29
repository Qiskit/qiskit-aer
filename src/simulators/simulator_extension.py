import sys
import os
from distutils.core import Extension
from distutils.spawn import find_executable


def simulator_extension(package_name, source_files, include_dirs=None, blas=True):
    """
    Return an simulator extension for setup cythonize.

    Args:
        package_name (str): Name of Cython package
        source_files (list(str)): Additional source file directories
        include_dirs (list(str)): additinal directories to include
        blas (bool): Include Numpy BLAS libraries

    Returns:
        An Extension for use in cythonize.
    """

    _include_dirs = [os.path.abspath('../common/src/third-party'),
                     os.path.abspath('../common/src/simulator-framework')]
    if isinstance(include_dirs, list):
        _include_dirs += include_dirs

    warnings = ['-pedantic', '-Wall', '-Wextra', '-Wfloat-equal', '-Wundef',
                '-Wcast-align', '-Wwrite-strings', '-Wmissing-declarations',
                '-Wshadow', '-Woverloaded-virtual']

    opt = ['-ffast-math', '-O3', '-march=native']
    if sys.platform != 'win32':
        _extra_compile_args = ['-g', '-std=c++14'] + opt + warnings
    else:
        _extra_compile_args = ['/W1', '/Ox']

    _extra_link_args = []
    _libraries = ['iomp5', 'pthread']
    _library_dirs = []

    # MacOS Specific build instructions
    if sys.platform == 'darwin':
        # Set minimum os version to support C++11 headers
        min_macos_version = '-mmacosx-version-min=10.9'
        _extra_compile_args.append(min_macos_version)
        _extra_link_args.append(min_macos_version)

        # Check for Homebrew libomp and use default apple clang
        if os.path.exists('/usr/local/opt/libomp'):
            # print('Building simulator using Apple clang w/ Homebrew LLVM OpenMP')
            _library_dirs.append('/usr/local/opt/libomp/lib')
            _include_dirs.append('/usr/local/opt/libomp/include')
            _extra_compile_args += ['-Xpreprocessor', '-fopenmp']
            _extra_link_args += ['-Xpreprocessor', '-fopenmp', '-liomp5', '-lpthread']
        else:
            # Check for OpenMP compatible GCC compiler
            for gcc in ['g++-8', 'g++-7', 'g++-6', 'g++-5']:
                path = find_executable(gcc)
                if path is not None:
                    # print('Building using GCC ({})'.format(gcc))
                    # Use most recent GCC compiler
                    os.environ['CC'] = path
                    os.environ['CXX'] = path
                    _extra_compile_args.append('-Wno-cast-function-type')
                    _extra_compile_args.append('-fopenmp')
                    _extra_link_args.append('-fopenmp')
                    break
    # Windows Specific build instructions
    elif sys.platform == 'win32':
        _extra_compile_args.append('/openmp')
    else:
        # Linux
        _extra_compile_args.append('-fopenmp')
        _extra_link_args.append('-fopenmp')

    # Get BLAS library from Numpy
    if blas is True:
        from numpy.__config__ import get_info as np_config
        blas_info = np_config('blas_mkl_info')
        if blas_info == {}:
            blas_info = np_config('blas_opt_info')
        _extra_compile_args += blas_info.get('extra_compile_args', [])
        _extra_link_args += blas_info.get('extra_link_args', [])
        _libraries += blas_info.get('libraries', [])
        _library_dirs += blas_info.get('library_dirs', [])
        _include_dirs += blas_info.get('include_dirs', [])

    # Remove -Wstrict-prototypes from cflags
    import distutils.sysconfig
    cfg_vars = distutils.sysconfig.get_config_vars()
    if "CFLAGS" in cfg_vars:
        cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

    return Extension(package_name,
                     sources=source_files,
                     extra_link_args=_extra_link_args,
                     extra_compile_args=_extra_compile_args,
                     libraries=_libraries,
                     library_dirs=_library_dirs,
                     include_dirs=_include_dirs,
                     language='c++')
