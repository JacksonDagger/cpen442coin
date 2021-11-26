/*********************************************************************
* Filename:   single_miner.c
* Author:     Jackson Dagger (JacksonDDagger at google mail)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Mine a single CPEN442 coin for assignment 4 at default difficulty
* and preceding coin hash
*********************************************************************/

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

// hip header file
#include "hip/hip_runtime.h"
#include "cpen442coin.h"

#define ID_OF_MINER "e8d2dadb3c5ead451b8943cff5ef909ef0f3de313c4d274d85d2d3c8a5a30c1f"
#define BLOCKS 512
#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define BATCHSIZE 256
#define BATCH_SHIFT 8

__global__ void 
cpen442coin_kernel(uint64_t init, unsigned int difficulty, const BYTE* prec, BYTE* res, unsigned int *found) 
{ 
    init += (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y) << BATCH_SHIFT;
    union blob start;
    start.num = init;
    struct thread_data arg;
    memcpy(arg.preceeding, prec, SHA256_STRLEN);
    memcpy(arg.miner_id, ID_OF_MINER, SHA256_STRLEN);
    arg.inc = 1;
    arg.difficulty = difficulty*4;
    arg.max = BATCHSIZE;
    arg.res = 0;
    memcpy(arg.start_ret_bytes, start.bytes, BLOB_SIZE);

    find_long_blob((void *) &arg);

    if (arg.res) {
        unsigned int a = atomicAdd(found, 1);
        if (a == 1) {
            memcpy(res, arg.start_ret_bytes, BLOB_SIZE);
        }
    }
}

__host__ int main(int argc, char *argv[])
{
    time_t start_time = time(0);
    srand(time(0));
    char prec_str[SHA256_STRLEN + 1] = "00000000f02eafab71af360b73b2004fb7d47094468cd87cbff4c330e6f55bad";
    int difficulty = 11;

    memcpy(prec_str, argv[1], SHA256_STRLEN);
    char *a = argv[2];
    difficulty = atoi(a);
    uint64_t init = ((uint64_t) rand() << 16);
   
    BYTE prec_bytes[SHA256_STRLEN];
    memcpy(prec_bytes, prec_str, SHA256_STRLEN);

    BYTE ret[BLOB_SIZE];
    //hipMalloc((void**)&ret, BLOB_SIZE * sizeof(BYTE));

    unsigned int found = 0;
    //hipMalloc((void**)&found, sizeof(unsigned int));

    hipLaunchKernelGGL(cpen442coin_kernel, dim3(BLOCKS, BLOCKS), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
        0, 0, init, difficulty, prec_bytes, ret, &found);
    hipDeviceSynchronize();
        
    if (found) {
        printf("success:");
        print_bytes(ret, BLOB_SIZE);
    }
    else {
        double hashrate = ((double)BATCHSIZE*THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y)/(time(0) - start_time);
        printf("hash rate (hps): %.4lf", hashrate);
    }
    return 0;
}