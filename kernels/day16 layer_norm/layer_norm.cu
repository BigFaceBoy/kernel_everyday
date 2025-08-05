#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ float reduce_smem[NUM_WARPS];

#pragma unroll
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if(lane == 0){
        reduce_smem[warp] = val;
    }
    __syncthreads();

    float sum = lane < NUM_WARPS ? reduce_smem[lane]: 0.0f;

#pragma unroll
    for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    

    sum = __shfl_sync(0xffffffff, sum, 0, 32);
    return sum;
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float* x, float* y, float g, float b, int N, int K){
    int idx= blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float mu;
    __shared__ float sigma;           

    const float epsilon = 1e-5f;
    float x_i = (idx < N * K) ? x[idx] : 0.0f;

    float sum = block_reduce_sum_f32<NUM_THREADS>(x_i);
    if(tid == 0){
        mu = sum / (float)K;
    }
    float v = (x_i - mu) * (x_i - mu);
    float sum_sigma =  block_reduce_sum_f32<NUM_THREADS>(v);
    if(tid == 0){
        sigma = sqrtf(sum_sigma / (float)K + epsilon);
    }

    if(idx < N * K){
        y[idx] = ((x_i - mu) / sigma) * g + b;
    }
}


template <const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float* x, float* y, float g, float b, int N, int K){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    int tid = threadIdx.x;
    __shared__ float mu;
    __shared__ float sigma;
    const float epsilon = 1e-5f;
    
    if(idx + 3 < N * K){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        float reg_sum = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
        float sum = block_reduce_sum_f32<NUM_THREADS>(reg_sum);
        if(tid == 0){
            mu = sum / (float)K;
        }
        float mean_x_pow = (reg_x.x - mu) * (reg_x.x - mu);
        float mean_y_pow = (reg_x.y - mu) * (reg_x.y - mu);
        float mean_z_pow = (reg_x.z - mu) * (reg_x.z - mu);
        float mean_w_pow = (reg_x.w - mu) * (reg_x.w - mu);

        float mean_sum = mean_x_pow + mean_y_pow + mean_z_pow + mean_w_pow;
        float sum_sigma =  block_reduce_sum_f32<NUM_THREADS>(mean_sum);
        if(tid == 0){
            sigma = sqrtf(sum_sigma / (float)K + epsilon);
        }

        float4 reg_y;
        reg_y.x = ((reg_x.x - mu) / sigma) * g + b;
        reg_y.y = ((reg_x.y - mu) / sigma) * g + b;
        reg_y.z = ((reg_x.z - mu) / sigma) * g + b;
        reg_y.w = ((reg_x.w - mu) / sigma) * g + b;

        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
    }
}
