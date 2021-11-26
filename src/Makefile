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

GPUMINER_SRC = $(wildcard *_g.cpp)
GPUMINER_OBJ = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(GPUMINER_SRC)))
GPUMINER_BIN = ${BIN_DIR}/GPUMINER-hip

.PHONY: all clean run itburn

all: ${GPUMINER_BIN}

${GPUMINER_BIN}: ${GPUMINER_OBJ}
	${HIPCC} ${LDFLAGS} -o ${GPUMINER_BIN} ${GPUMINER_OBJ}

${BUILD_DIR}/cpen442coin_g.o: cpen442coin_g.cpp ${BUILD_DIR}/sha256_g.o $(DEPS) Makefile
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