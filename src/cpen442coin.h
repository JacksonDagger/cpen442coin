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

__device__ unsigned long check_hash(const BYTE hash[], unsigned int difficulty);
__device__ void find_long_blob(void *arg);
void print_bytes(BYTE bytes[], unsigned long len);
