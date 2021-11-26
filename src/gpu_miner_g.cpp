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

int main(int argc, char *argv[])
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

    blob ret;
    ret.num = 0;
    //hipMalloc((void**)&ret, BLOB_SIZE * sizeof(BYTE));

    hipLaunchKernelGGL(cpen442coin_kernel, dim3(BLOCKS, BLOCKS), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
        0, 0, init, difficulty, prec_bytes, ret.bytes);
    hipDeviceSynchronize();
        
    if (ret.num) {
        printf("success:");
        print_bytes(ret.bytes, BLOB_SIZE);
    }
    else {
        double hashrate = ((double)GPU_BATCHSIZE*THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y)/(time(0) - start_time);
        printf("hash rate (hps): %.4lf", hashrate);
    }
    return 0;
}