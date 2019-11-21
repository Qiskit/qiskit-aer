set(_find_pybind_includes_command "
import sys
import pybind11
sys.stdout.write(pybind11.get_include())
")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "${_find_pybind_includes_command}"
                OUTPUT_VARIABLE _py_output
                RESULT_VARIABLE _py_result)
if(_py_result EQUAL "0")
    message(STATUS "PYCOMM RAW: ${_py_output}")
    set(PYBIND_INCLUDE_DIRS "${_py_output}")
    message(STATUS "PYBIND INCLUDES FOUND: ${PYBIND_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "COULD NOT FIND PYBIND!")
endif()
