INCLUDE = -I../src/
WARNINGS = -pedantic -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wshadow -Woverloaded-virtual
OPT = -O3 -march=native -ffast-math
OUTPUT_DIR = ../out/simulators-cpp

CC = g++

# -march not supported on ppc64le (Power). Use -mcpu instead
ifeq ($(shell uname -m),ppc64le)
	OPT = -O3 -mcpu=native -ffast-math
else
	OPT = -O3 -march=native -ffast-math
endif

OS:=$(shell uname)

# BLAS Library
ifeq ($(OS),Darwin)
	LIB_BLAS = -framework Accelerate
else
	LIB_BLAS = -llapack -lblas
endif

# OpenMP Library
CPP_OMP = -fopenmp
LIB_OMP = -lpthread

# Check for OpenMP support on MacOS
# To use OpenMP on MacOS there are two options that use the Homebrew package manager
# 1. Install LLVM OpenMP (libomp) and compile using the default Apple Clang from XCode
#    >> brew install libomp
LIBOMP_DIR = /usr/local/opt/libomp/
# 2. Install GCC compiler (gcc) and compile using GCC
#    >> brew install gcc
CC_GCC = g++-8
ifeq ($(OS),Darwin)
	ifneq ($(shell ${CC} --version | grep ^Apple),)
		# g++ points to Apple compiler
		ifneq ($(wildcard $(LIBOMP_DIR).*),)
			# Found LLVM OpenMP library
			CPP_OMP = -Xpreprocessor -fopenmp -I$(LIBOMP_DIR)include
			LIB_OMP = -lpthread -lomp -L$(LIBOMP_DIR)lib
		else
			# Didn't find LLVM OpenMP, see if GCC can be found
			ifeq ($(shell $(CC_GCC) --version | grep ^'command not found'),)
				# Found GCC compiler
				CC = $(CC_GCC)
			else
				# Compile without OpenMP
				CPP_OMP =
			endif
		endif
	endif
endif

CPPFLAGS = $(USER_FLAGS) -std=c++11 $(CPP_OMP) $(OPT) $(INCLUDE) $(WARNINGS) 
LIBS = $(LIB_BLAS) $(LIB_OMP)

.SUFFIXES:.cpp .cc .o .c
.cpp.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o  ${OUTPUT_DIR}/$@ $<
.cc.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o  ${OUTPUT_DIR}/$@ $<
.c.o:
	${CC} -c ${CPPFLAGS} ${DEFINES} -o  ${OUTPUT_DIR}/$@ $<


all: directories qasm_sim obs_sim state_sim

directories:
	mkdir -p $(OUTPUT_DIR)

qasm_sim: qasm_simulator.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/qasm_simulator_cpp ${OUTPUT_DIR}/qasm_simulator.o $(LIBS)

obs_sim: observables_simulator.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/observables_simulator_cpp ${OUTPUT_DIR}/observables_simulator.o $(LIBS)

state_sim: statevector_simulator.o
	$(CC) $(CPPFLAGS) $(DEFINES) -o ${OUTPUT_DIR}/statevector_simulator_cpp ${OUTPUT_DIR}/statevector_simulator.o $(LIBS)

clean:
	rm -rf ${OUTPUT_DIR}
