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
#define BLOCKS 512
#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define GPU_BATCHSIZE 256
#define BATCH_SHIFT 8

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
