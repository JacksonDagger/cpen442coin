SRC_DIR     = ./src
BUILD_DIR   = $(SRC_DIR)/build
BIN_DIR     = ../bin
DEP_LIST 	= $(wildcard $(SRC_DIR)/*.h)
SRC_LIST 	= $(wildcard $(SRC_DIR)/*.c)
OBJ_LIST 	= $(BUILD_DIR)/$(notdir $(SRC_LIST:.c=.o))

CC=gcc
CFLAGS=-I.
LFLAGS = -pthread
DEPS = cpen442coin.h, sha256.h

%.o: %.c $(DEPS)
	$(CC) -c -o $(LFLAGS) $@ $< $(CFLAGS)
cpu_miner: cpu_miner.o cpen442coin.o sha256.o
	$(CC) $(LFLAGS) -o $(BIN_DIR)/cpu_miner cpu_miner.o cpen442coin.o sha256.o