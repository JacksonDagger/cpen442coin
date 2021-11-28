/*********************************************************************
* Filename:   cpen442coin.c
* Author:     Jackson Dagger (JacksonDDagger at google mail)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Mine for CPEN442 coin 
*********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdint.h>

#include "cpen442coin.h"
#include "sha256.h"
#include "hip/hip_runtime.h"

#define BLOB_LEN        8
#define COIN_PREF1      "CPEN 442 Coin"
#define COIN_PREF2      "2021"
#define COIN_PREF1_LEN  13
#define COIN_PREF2_LEN  4

#define MINER_ID_LEN SHA256_STRLEN

#ifdef GPU_CODE
    #define MEMCPY(dest, src, len) hipMemcpyDtoD(dest, src, len)
    #define LOC __device__
#else
    #define MEMCPY(dest, src, len) memcpy(dest, src, len)
    #define LOC __host__
#endif

__host__ void launch_stream(void *arg)
{
    struct stream_data *sdata=(struct stream_data *)arg;
    blob ret;
    uint64_t init;

    init = ((uint64_t) rand() << RAND_SHIFT);

    BYTE *ret_bytes;
    BYTE *prec_bytes;

    hipMalloc((void**)&ret_bytes, BLOB_SIZE);
    hipMalloc((void**)&prec_bytes, SHA256_STRLEN);
    hipMemcpy(ret_bytes, sdata->ret.bytes, BLOB_SIZE, hipMemcpyHostToDevice);
    hipMemcpy(prec_bytes, sdata->preceeding, SHA256_STRLEN, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(cpen442coin_kernel, dim3(BLOCKS, BLOCKS), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
        0, *(sdata->stream), init, sdata->difficulty, prec_bytes, ret_bytes);

    hipStreamSynchronize(*(sdata->stream));

    hipMemcpy(sdata->ret.bytes, ret_bytes, BLOB_SIZE, hipMemcpyDeviceToHost);

    hipFree(ret_bytes);
    hipFree(prec_bytes);
}

__global__ void 
cpen442coin_kernel(uint64_t init, unsigned int difficulty, const BYTE* prec, BYTE* res) 
{ 
    init += (((THREADS_PER_BLOCK_X * hipBlockIdx_x + hipThreadIdx_x) << WIDTH_SHIFT) + THREADS_PER_BLOCK_Y * hipBlockIdx_y + hipThreadIdx_y) << BATCH_SHIFT;
    union blob test_blob;
    test_blob.num = init;

    SHA256_CTX start_ctx, test_ctx;

    sha256_init(&start_ctx);
    sha256_update(&start_ctx, (BYTE *) COIN_PREF1, COIN_PREF1_LEN);
    sha256_update(&start_ctx, (BYTE *) COIN_PREF2, COIN_PREF2_LEN);
    sha256_update(&start_ctx, prec, SHA256_STRLEN);
    
    BYTE test_hash[SHA256_BLOCK_SIZE];
    BYTE miner_id[MINER_ID_LEN+1] = ID_OF_MINER;

    for (int i = 0; i < GPU_BATCHSIZE; i++)
    {
        test_ctx = start_ctx;
        test_blob.num += 1;

        sha256_update(&test_ctx, test_blob.bytes, BLOB_LEN);
        sha256_update(&test_ctx, miner_id, MINER_ID_LEN);

        sha256_final(&test_ctx, test_hash);
        
        if(!check_hash(test_hash, difficulty))
        {
            for (int j = 0; j < BLOB_SIZE; j++){
                res[j] = test_blob.bytes[j];
            }
            return;
        }
    }
    return;
}

/*
 * Function: find_long_blob
 * ----------------------------
 *   Find a blob that produces a hash that satisifies the given difficutly, previous coin hash and miner_id
 *
 *   blob: the starting blob to increment and where the successful blob will be place
 *   preceeding: the string representing the hash of the prceeding block
 *   miner_id: the string representing the miner_id (assumed to be)
 *   inc: the value to increment the blob by each time (can be used to prevent checking of duplicates in parallel processing)
 *   difficulty: the difficulty level of cpen442coin (ie twice the number of bytes that must be 0 at the start of the hash)
 *
 *   returns: 0 if successful and the number of iterations otherwise
 */
__host__ void find_long_blob(void *arg)
{
    struct thread_data *tdata=(struct thread_data *)arg;
    BYTE start_ret_bytes[BLOB_SIZE];
    BYTE preceeding[SHA256_STRLEN];
    BYTE miner_id[SHA256_STRLEN];

    MEMCPY(start_ret_bytes, tdata->start_ret_bytes, BLOB_SIZE);
    MEMCPY(preceeding, tdata->preceeding, SHA256_STRLEN);
    MEMCPY(miner_id, tdata->miner_id, SHA256_STRLEN);
    unsigned int inc = tdata->inc;
    unsigned int difficulty = tdata->difficulty;
    unsigned long max = tdata -> max;

    unsigned long i;
    union blob test_blob;
    MEMCPY(test_blob.bytes, start_ret_bytes, BLOB_SIZE);

    SHA256_CTX start_ctx, test_ctx;

    // for efficiency, only update hash once
    sha256_init(&start_ctx);
    sha256_update(&start_ctx, (BYTE *) COIN_PREF1, COIN_PREF1_LEN);
    sha256_update(&start_ctx, (BYTE *) COIN_PREF2, COIN_PREF2_LEN);
    sha256_update(&start_ctx, preceeding, SHA256_STRLEN);
    
    BYTE test_hash[SHA256_BLOCK_SIZE];

    if (max > ULONG_MAX/inc) {
        max = ULONG_MAX/inc;
    }

    for (i = 0; i < max; i ++)
    {
        test_ctx = start_ctx;
        test_blob.num += inc;

        sha256_update(&test_ctx, test_blob.bytes, BLOB_LEN);
        sha256_update(&test_ctx, miner_id, MINER_ID_LEN);

        sha256_final(&test_ctx, test_hash);
        
        if(!check_hash(test_hash, difficulty))
        {
            MEMCPY(tdata->start_ret_bytes, test_blob.bytes, BLOB_SIZE);
            tdata->res = 0;
            return;
        }
    }
    tdata->res = i > 0? i : 1;
    return;
}

/*
 * Function: check_hash
 * ----------------------------
 *   Check if given hash output satisfies difficulty requirement
 *
 *   hash: the given SHA256 hash (assumed to be 256 bits long)
 *   difficulty: the difficulty level of cpen442coin (ie twice the number of bytes that must be 0 at the start of the hash) (<= 32)
 *
 *   returns: 0 if successful
 */
__device__ __host__ unsigned long check_hash(const BYTE hash[], unsigned int difficulty)
{
    unsigned long mask = 1;
    mask <<= difficulty;
    mask -= 1;
    unsigned long check_bits = *((unsigned long*) hash); // works on little endian systems only (ie x86)
    return check_bits & mask;
}

/*
 * Function: print_bytes
 * ----------------------------
 *   Print the given BYTE array as lower case hex
 *
 *   bytes: bytes to be printed
 *   len: number of bytes to be printed (on user to check it doesn't exceed the length of bytes)
 */
__host__ void print_bytes(BYTE bytes[], unsigned long len) {
    for (unsigned int i = 0; i < len; i++) {
        printf("%02X", bytes[i]);
    }
}