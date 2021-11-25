SRC_DIR     = ./src
BUILD_DIR   = $(SRC_DIR)/build
BIN_DIR     = ./bin
DEP_LIST 	= $(wildcard $(SRC_DIR)/*.h)
SRC_LIST 	= $(wildcard $(SRC_DIR)/*.c)
OBJ_LIST 	= $(BUILD_DIR)/$(notdir $(SRC_LIST:.c=.o))

CC=gcc
CFLAGS=-I.
CFLAGS+=-I$(SRC_DIR)
CFLAGS+=-I$(BUILD_DIR)
LFLAGS = -pthread
DEPS = $(SRC_DIR)/cpen442coin.h $(SRC_DIR)/sha256.h

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(DEPS)
	$(CC) -c $(LFLAGS) $< $(CFLAGS) -o $@

cpu_miner: $(BUILD_DIR)/cpu_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	$(CC) $(LFLAGS) -o $(BIN_DIR)/cpu_miner $(BUILD_DIR)/cpu_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	
single_miner: $(BUILD_DIR)/single_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	$(CC) $(LFLAGS) -o $(BIN_DIR)/single_miner $(BUILD_DIR)/single_miner.o $(BUILD_DIR)/cpen442coin.o $(BUILD_DIR)/sha256.o
	