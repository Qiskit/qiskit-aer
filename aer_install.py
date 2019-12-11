import sys, os, fnmatch
from shutil import copy, rmtree


def find(pattern, path):
    """Find files mathcing pattern."""
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# Options
STANDALONE = bool('--standalone' in sys.argv)
DEBUG = bool('--debug' in sys.argv)
TESTS = bool('--test' in sys.argv)
COMPILER = 'g++-8' if '--gcc' in sys.argv else 'g++'
BUILD_ARGS = '-DCMAKE_CXX_COMPILER={}'.format(COMPILER)

# STANDALONE BUILD
if STANDALONE:
    # Remove out dir
    if 'out' in os.listdir('.'):
        print('... Removing out/ ...')
        rmtree('out')
    # Make new out director
    CONFIG = 'Debug' if DEBUG else 'Release'
    os.mkdir('out')
    CMAKE_CMD = 'cd out;cmake .. {};'.format(BUILD_ARGS)
    CMAKE_CMD += 'cmake --build . --config {} -- -j4'.format(CONFIG)
    os.system(CMAKE_CMD)
# BUILD CYTHON
else:
    # Remove old copied files
    print('... Removing copied .so files ...')
    EXE = find('*.so', 'qiskit/providers/aer/backends/')
    LIB = find('*.dylib', 'qiskit/providers/aer/backends/')
    for filename in EXE + LIB:
        os.remove(filename)
        print('... removed: {}'.format(filename))

    # Remove old dist directory
    if 'dist' in os.listdir('.'):
        print('... Removing previous dist/ ...')
        rmtree('dist')

    # Build
    print('... Building ...')
    BUILD_ARGS += ' -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9'
    BUILD_ARGS += ' -DCMAKE_OSX_ARCHITECTURES:STRING=x86_64'
    if DEBUG:
        BUILD_ARGS += ' -DCMAKE_BUILD_TYPE=Debug'
    SETUP_ARGS = '-j4'
    os.system('python setup.py bdist_wheel -- {} -- {}'.format(BUILD_ARGS, SETUP_ARGS))

    # Install
    WHEEL_PATH = find('*.whl', 'dist')
    if WHEEL_PATH:
        print('... Installing ...')
        os.system('pip install -r requirements-dev.txt')
        os.system('pip install {} --upgrade'.format(WHEEL_PATH[0]))

    # Copy executeables for running tests
    # Remove old copied files
    print('... Copying built .so files to testing directory ...')
    BUILD_DIR = os.listdir('_skbuild')
    AER_DIR = 'qiskit/providers/aer/backends/'
    if BUILD_DIR:
        INSTALL_DIR = os.path.join('_skbuild', BUILD_DIR[0], 'cmake-install', AER_DIR)
        EXE = find('*.so', INSTALL_DIR)
        LIB = find('*.dylib', INSTALL_DIR)
        for filename in EXE + LIB:
            copy(filename, AER_DIR)
            print('... Copied: {}'.format(filename))

    # RUN TESTS
    if TESTS:
        os.system('python -m unittest discover -v test.terra')
