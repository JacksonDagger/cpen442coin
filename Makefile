SRC_DIR     = ./src
BUILD_DIR   = $(SRC_DIR)/build
BIN_DIR     = ./bin
DEP_LIST 	= $(wildcard $(SRC_DIR)/*.h)
SRC_LIST 	= $(wildcard $(SRC_DIR)/*.c)
OBJ_LIST 	= $(BUILD_DIR)/$(notdir $(SRC_LIST:.c=.o))

CPP=gcc
CPPFLAGS=-I.
CPPFLAGS+=-I$(SRC_DIR)
LFLAGS = -pthread
DEPS = $(SRC_DIR)/cpen442coin.h $(SRC_DIR)/sha256.h

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
HIPCC_FLAGS = -I$(SRC_DIR) -fgpu-rdc --hip-link

TARGET=hcc

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	$(CPP) -c $(LFLAGS) $< $(CPPFLAGS) -o $@

$(BUILD_DIR)/%.obj: $(SRC_DIR)/%.cpp $(DEPS)
	$(HIPCC) -g $< $(HIPCC_FLAGS) -o $@

gpu_miner: $(BUILD_DIR)/gpu_miner.obj $(BUILD_DIR)/cpen442coin.obj $(BUILD_DIR)/sha256.obj
	$(HIPCC) $(HIPCC_FLAGS) -o $(BIN_DIR)/gpu_miner $(BUILD_DIR)/gpu_miner.obj $(BUILD_DIR)/cpen442coin.obj $(BUILD_DIR)/sha256.obj

cpu_miner: $(BUILD_DIR)/cpu_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	$(CPP) $(LFLAGS) -o $(BIN_DIR)/cpu_miner $(BUILD_DIR)/cpu_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	
single_miner: $(BUILD_DIR)/single_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	$(CPP) $(LFLAGS) -o $(BIN_DIR)/single_miner $(BUILD_DIR)/single_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o

clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(BUILD_DIR)/*.obj
	rm -f $(BIN_DIR)/*