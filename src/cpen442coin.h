/*********************************************************************
* Filename:   cpen442coin.c
* Author:     Jackson Dagger (JacksonDDagger at google mail)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    API for CPEN442 coin mining API
*********************************************************************/

#include "sha256.h"
#include <stdint.h>

#define BLOB_SIZE 8
#define SHA256_STRLEN 2*SHA256_BLOCK_SIZE

#define ID_OF_MINER "e8d2dadb3c5ead451b8943cff5ef909ef0f3de313c4d274d85d2d3c8a5a30c1f"

#define BLOCK_SHIFT 7
#define BLOCKS (1 << BLOCK_SHIFT) // 256

#define XTHREAD_SHIFT 3
#define THREADS_PER_BLOCK_X (1 << XTHREAD_SHIFT) // 16

#define YTHREAD_SHIFT 3
#define THREADS_PER_BLOCK_Y (1 << YTHREAD_SHIFT) // 8

#define BATCH_SHIFT 10
#define GPU_BATCHSIZE (1 << BATCH_SHIFT) // 256

#define WIDTH_SHIFT (BLOCK_SHIFT + YTHREAD_SHIFT) // 11
#define RAND_SHIFT (BATCH_SHIFT + WIDTH_SHIFT + BLOCK_SHIFT + XTHREAD_SHIFT) // 31

#define RUN_SIZE BLOCKS*BLOCKS*GPU_BATCHSIZE*THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y

union blob {
    BYTE bytes[BLOB_SIZE]; 
    uint64_t num;
};

typedef struct thread_data {
    BYTE start_ret_bytes[BLOB_SIZE];
    BYTE preceeding[SHA256_STRLEN];
    BYTE miner_id[SHA256_STRLEN];
    unsigned int inc;
    unsigned int difficulty;
    unsigned long max;
    unsigned int res;
} thead_data;

__device__ __host__ unsigned long check_hash(const BYTE hash[], unsigned int difficulty);
__host__ void find_long_blob(void *arg);
__global__ void  cpen442coin_kernel(uint64_t init, unsigned int difficulty, const BYTE* prec, BYTE* res);
__host__ void print_bytes(BYTE bytes[], unsigned long len);
