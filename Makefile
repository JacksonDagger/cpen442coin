SRC     = ./src
BUILD_DIR   = $(SRC)/build
BIN_DIR     = ./bin
DEP_LIST 	= $(wildcard $(SRC)/*.h)
SRC_LIST 	= $(wildcard $(SRC)/*.c)
OBJ_LIST 	= $(BUILD_DIR)/$(notdir $(SRC_LIST:.c=.o))

LFLAGS = -pthread
DEPS = $(SRC)/cpen442coin.h $(SRC)/sha256.h

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
HIPCC_FLAGS = -I$(SRC) -I.

TARGET=main

$(BUILD_DIR)/cpen442coin_gpu.o: $(SRC)/cpen442coin.cpp $(DEPS)
	$(HIPCC) -c -g $< $(HIPCC_FLAGS) -o $@

$(BUILD_DIR)/gpu_miner.o: $(SRC)/gpu_miner.cpp $(BUILD_DIR)/cpen442coin_gpu.o $(DEPS)
	$(HIPCC) -L$(SRC) -g $< $(HIPCC_FLAGS) -O3 -o $@

$(BUILD_DIR)/%.o: $(SRC)/%.cpp $(DEPS)
	$(HIPCC) -c $(LFLAGS) $< $(HIPCC_FLAGS) -o $@

gpu_miner: $(BUILD_DIR)/gpu_miner.o $(BUILD_DIR)/cpen442coin_gpu.o $(BUILD_DIR)/sha256.o
	$(HIPCC) $(HIPCC_FLAGS) -o $(BIN_DIR)/gpu_miner $(BUILD_DIR)/gpu_miner.o $(BUILD_DIR)/cpen442coin_gpu.o $(BUILD_DIR)/sha256.o

cpu_miner: $(BUILD_DIR)/cpu_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	$(HIPCC) $(LFLAGS) -o $(BIN_DIR)/cpu_miner $(BUILD_DIR)/cpu_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	
single_miner: $(BUILD_DIR)/single_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	$(HIPCC) $(LFLAGS) -o $(BIN_DIR)/single_miner $(BUILD_DIR)/single_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o

clean:
	rm -f $(BUILD_DIR)/*.o
	rm -f $(BUILD_DIR)/*.o
	rm -f $(BIN_DIR)/*