HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
        HIP_PATH=/opt/rocm
endif


HCC_PATH ?= /opt/rocm/hcc
HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)
HIP_INCLUDE = -I. -I${HIP_PATH}/include -I${HCC_PATH}/include
BUILD_DIR ?= build
BIN_DIR ?= ../bin

HIPCC = ${HIP_PATH}/bin/hipcc
CPPFLAGS = -fgpu-rdc -O3
LDFLAGS = -fgpu-rdc --hip-link -lm -lpthread
DEPS = cpen442coin.h sha256.h

ifeq (${HIP_PLATFORM}, nvcc)
    CPPFLAGS += -arch=compute_20
endif

UTILS_SRC = cpen442coin.cpp sha256.cpp
GPUMINER_SRC = $(UTILS_SRC) gpu_miner.cpp
GPUMINER_OBJ = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(GPUMINER_SRC)))
GPUMINER_BIN = ${BIN_DIR}/gpuminer-hip

CPUMINER_SRC = $(UTILS_SRC) cpu_miner.cpp
CPUMINER_OBJ = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(CPUMINER_SRC)))
CPUMINER_BIN = ${BIN_DIR}/cpuminer-hip

.PHONY: all clean run itburn

all: cpu gpu

gpu: ${GPUMINER_BIN}

cpu: ${CPUMINER_BIN}

gpu_debug: ${GPUMINER_OBJ}
	mkdir -p ${BIN_DIR}
	${HIPCC} -g ${LDFLAGS} -o ${GPUMINER_BIN} ${GPUMINER_OBJ}

${GPUMINER_BIN}: ${GPUMINER_OBJ}
	mkdir -p ${BIN_DIR}
	${HIPCC} ${LDFLAGS} -o ${GPUMINER_BIN} ${GPUMINER_OBJ}

${CPUMINER_BIN}: ${CPUMINER_OBJ}
	mkdir -p ${BIN_DIR}
	${HIPCC} ${LDFLAGS} -o ${CPUMINER_BIN} ${CPUMINER_OBJ}

${BUILD_DIR}/cpen442coin.o: cpen442coin.cpp ${BUILD_DIR}/sha256.o $(DEPS) Makefile
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} ${CPPFLAGS} -c -o $@ $<  

${BUILD_DIR}/%.o: %.cpp $(DEPS) Makefile
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} ${CPPFLAGS} -c -o $@ $<  

run: itburn
itburn:
	HCC_LAZYINIT=ON ${GPUMINER_BIN}

clean:
	rm -rf ${BUILD_DIR}
	rm -rf ${BIN_DIR}