/*********************************************************************
* Filename:   single_miner.c
* Author:     Jackson Dagger (JacksonDDagger at google mail)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Mine a single CPEN442 coin for assignment 4 at default difficulty
* and preceding coin hahs
*********************************************************************/

#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

#include "cpen442coin.h"

#define DIFFICULTY 32
#define PRECEDING_COIN "a9c1ae3f4fc29d0be9113a42090a5ef9fdef93f5ec4777a008873972e60bb532"
#define BATCHSIZE 16777216 // 2^24 meaning we expect to find a coin once every 256 batches


int main(void)
{
    union blob start;
    start.num = 0;

    int res;
    time_t start_time, batch_start_time, end_time;
    time(&start_time);

    struct thread_data arg;
    memcpy(arg.preceeding, PRECEDING_COIN, SHA256_STRLEN);
    memcpy(arg.miner_id, ID_OF_MINER, SHA256_STRLEN);
    arg.inc = 1;
    arg.difficulty = DIFFICULTY*4;
    arg.max = BATCHSIZE;
    arg.res = 0;

    do {
        start.num += BATCHSIZE;
        printf("blobs checked: %lu, ", start.num - BATCHSIZE);
        time(&batch_start_time);
        memcpy(arg.start_ret_bytes, start.bytes, BLOB_SIZE);
        find_long_blob((void *) &arg);
        time(&end_time);
        printf("batch time(s): %ld, time elapsed(s): %ld\n", end_time - batch_start_time, end_time - start_time);
    }
    while(arg.res);

    printf("success!\n");
    printf("blob: ");
    print_bytes(start.bytes, BLOB_SIZE);
    printf("\n");
}