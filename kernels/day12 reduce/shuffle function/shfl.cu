#include <iostream>
#include <stdio.h>
#include <cooperative_groups.h>

#define SIZE 1024

__global__ void test_shfl_sync(int* x, int N){
    int tid = threadIdx.x;
    int val = tid;

    x[tid] = __shfl_sync(0xffffffff, val, 1);
}


__global__ void test_shfl_up_sync(int* x, int N){
    int tid = threadIdx.x;
    int val = tid ;

    x[tid] = __shfl_up_sync(0xffffffff, val, 1);
}

__global__ void test_shfl_down_sync(int*x, int N){
    int tid = threadIdx.x;
    int val = tid ;

     x[tid] = __shfl_down_sync(0xffffffff, val, 1);
}

__global__ void test_shfl_xor_sync(int* x, int N){
    int tid = threadIdx.x;
    int val = tid;

    x[tid] = __shfl_xor_sync(0xffffffff, val, 1);
}

int main(){
    int* x = (int *)malloc(sizeof(int) * SIZE);
    int* y = (int *)malloc(sizeof(int) * SIZE);

    for(int i = 0; i < SIZE; i++){
        x[i] = i;
        y[i] = 0;
    }

    int *d_x1;
    int *d_x2;
    int *d_x3;
    int *d_x4;
    cudaMalloc(&d_x1, sizeof(int) * SIZE);
    cudaMalloc(&d_x2, sizeof(int) * SIZE);
    cudaMalloc(&d_x3, sizeof(int) * SIZE);
    cudaMalloc(&d_x4, sizeof(int) * SIZE);
    cudaMemcpy(d_x1, x, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x3, x, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x4, x, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    int THREAD_NUM_PER_BLOCK = 64;
    int block_num = (SIZE + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK;

    test_shfl_sync<<<block_num, THREAD_NUM_PER_BLOCK>>>(d_x1, SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(y, d_x1, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 64; i++){
        printf("%d ", y[i]);
    }

    printf("\n");



    test_shfl_up_sync<<<block_num, THREAD_NUM_PER_BLOCK>>>(d_x2, SIZE); 
    cudaDeviceSynchronize();
    cudaMemcpy(y, d_x2, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 64; i++){
        printf("%d ", y[i]);
    }

    printf("\n");

    test_shfl_down_sync<<<block_num, THREAD_NUM_PER_BLOCK>>>(d_x3, SIZE); 
    cudaDeviceSynchronize();
    cudaMemcpy(y, d_x3, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 64; i++){
        printf("%d ", y[i]);
    }

    printf("\n");

    test_shfl_xor_sync<<<block_num, THREAD_NUM_PER_BLOCK>>>(d_x4, SIZE); 

    cudaDeviceSynchronize();
    cudaMemcpy(y, d_x4, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 64; i++){
        printf("%d ", y[i]);
    }

    printf("\n");
}

//输出是：
//1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 
//0 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 32 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 
//1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 31 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 63 
//1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 14 17 16 19 18 21 20 23 22 25 24 27 26 29 28 31 30 33 32 35 34 37 36 39 38 41 40 43 42 45 44 47 46 49 48 51 50 53 52 55 54 57 56 59 58 61 60 63 62 