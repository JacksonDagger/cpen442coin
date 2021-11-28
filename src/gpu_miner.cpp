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
    int difficulty = 10;

    memcpy(prec_str, argv[1], SHA256_STRLEN);
    char *a = argv[2];
    difficulty = atoi(a);
   
    BYTE prec_bytes[SHA256_STRLEN];
    memcpy(prec_bytes, prec_str, SHA256_STRLEN);

    pthread_t threads[NUM_STREAMS];
    struct stream_data sdata[NUM_STREAMS];
    hipStream_t streams[NUM_STREAMS];
    

    for (int i = 0; i < NUM_STREAMS; i++){
        hipStreamCreate(&streams[i]);
        sdata[i].stream = &(streams[i]);
        sdata[i].ret.num = 0;
        memcpy(sdata[i].preceeding, prec_bytes, SHA256_STRLEN);
        sdata[i].difficulty = 4*difficulty;

        pthread_create(&threads[i], NULL, (void* (*)(void*)) launch_stream, (void *)&(sdata[i]));
    }

    for (int i = 0; i < NUM_STREAMS; i++){
        pthread_join(threads[i], NULL);
    }

    hipDeviceSynchronize();
    
    blob final_ret;
    final_ret.num = 0;

    for (int i = 0; i < NUM_STREAMS; i++){
        if (sdata[i].ret.num){
            final_ret.num = sdata[i].ret.num;
            break;
        }
    }

    if (final_ret.num) {
        printf("success:");
        print_bytes(final_ret.bytes, BLOB_SIZE);
    }
    else {
        double hashrate = ((double) RUN_SIZE) / (time(0) - start_time);
        printf("hash rate (hps): %.4lf", hashrate);
    }
    return 0;
}