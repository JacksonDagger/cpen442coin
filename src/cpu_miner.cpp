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

#define NUM_THREADS 16
#define BATCHSIZE 16777216

int main(int argc, char *argv[])
{
    time_t start_time = time(0);
    srand(time(0));
    char prec_str[SHA256_STRLEN + 1] = "00000000f02eafab71af360b73b2004fb7d47094468cd87cbff4c330e6f55bad";
    int difficulty = 9;

    memcpy(prec_str, argv[1], SHA256_STRLEN);
    char *a = argv[2];
    difficulty = atoi(a);

    pthread_t threads[NUM_THREADS];
    struct thread_data thread_args[NUM_THREADS];

    union blob start;
    start.num = ((uint64_t) rand() << 30) - 1;
   
    BYTE prec_bytes[SHA256_STRLEN];
    memcpy(prec_bytes, prec_str, SHA256_STRLEN);
    int check = -1;

    for (int i = 0; i < NUM_THREADS; i++){
        start.num ++;
        memcpy(thread_args[i].start_ret_bytes, start.bytes, BLOB_SIZE);
        memcpy(thread_args[i].preceeding, prec_bytes, SHA256_STRLEN);
        memcpy(thread_args[i].miner_id, ID_OF_MINER, SHA256_STRLEN);
        thread_args[i].inc = NUM_THREADS;
        thread_args[i].difficulty = difficulty*4;
        thread_args[i].max = BATCHSIZE;
        thread_args[i].res = 0;

        pthread_create(&threads[i], NULL, (void* (*)(void*)) find_long_blob, (void *)&(thread_args[i]));
    }

    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
        if (! thread_args[i].res){
            check = i;
            break;
        }
    }

    if (check >= 0) {
        printf("success:");
        print_bytes(thread_args[check].start_ret_bytes, BLOB_SIZE);
    }
    else {
        double hashrate = ((double)BATCHSIZE*NUM_THREADS)/(time(0) - start_time);
        printf("hash rate (hps): %.4lf", hashrate);
    }
    return 0;
}