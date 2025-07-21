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
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define WARP_SIZE 32

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32x1_f32_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_x[NUM_THREADS];
    s_x[tid] = idx < N ? x[idx] : 0.0;
    __syncthreads();


    for(int offset = blockDim.x >> 1; offset >= 32; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }

    float sum = s_x[tid];
    for(int offset = 16; offset > 0; offset >>=1){
         sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if(tid == 0){
        atomicAdd(y, sum);
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32xdown_f32_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float sum = (idx < N) ? x[idx] : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum  += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0)
        atomicAdd(y, sum);
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* x, float*y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float sum = (idx < N) ? x[idx] : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum  += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0)
        atomicAdd(y, sum);

}

template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float* x, float*y, int N){
    int tid = threadIdx.x;
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
    float sum = (idx < N) ? (reg_x.x + reg_x.y + reg_x.w + reg_x.z) : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum  += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = sum;
    }
    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0)
        atomicAdd(y, sum);
  
}

//////////////////// fp16 ///////////////////////////////////////////////
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half* x, float* y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float  reduce_smem[NUM_WARPS];
    half sum_f16 = (idx < N) ? x[idx] : __float2half(0.f);

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum_f16  += __shfl_xor_sync(0xffffffff, sum_f16, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = __half2float(sum_f16);
    }
    __syncthreads();

    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0)
        atomicAdd(y, sum);

}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half* x, float* y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float  reduce_smem[NUM_WARPS];
    half sum_f16 = (idx < N) ? x[idx] : __float2half(0.f);

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    float sum_32 = __half2float(sum_f16);
#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum_32  += __shfl_xor_sync(0xffffffff, sum_32, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = sum_32;
    }
    __syncthreads();

    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0)
        atomicAdd(y, sum);

}



template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half* x, float* y, int N){
    //idx
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    int tid = threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float  reduce_smem[NUM_WARPS];
    half2 reg_x = (reinterpret_cast<half2*>(x + idx))[0];
    if(idx < N){
       reg_x = (reinterpret_cast<half2*>(x + idx))[0];
    }

    half sum_f16 = (idx < N) ? (reg_x.x + reg_x.y) : __float2half(0.f);

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    //每个warp单独归约，存储的结果数组
#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum_f16  += __shfl_xor_sync(0xffffffff, sum_f16, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = __half2float(sum_f16);
    }
    __syncthreads();

    //结果数组归约
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    //最终值
    if (tid == 0)
        atomicAdd(y, sum);
    
}


template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half* x, float* y, int N){
    //idx
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    int tid = threadIdx.x;

    //smem
    //lane  warp
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float  reduce_smem[NUM_WARPS];
    half2 reg_x = (reinterpret_cast<half2*>(x + idx))[0];
    if(idx < N){
       reg_x = (reinterpret_cast<half2*>(x + idx))[0];
    }

    float sum_32 = (idx < N) ? __half2float(reg_x.x + reg_x.y) : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    //每个warp单独归约，存储到结果数组

#pragma unroll
    for(int offset = 16; offset > 0 ; offset >>= 1){
        sum_32  += __shfl_xor_sync(0xffffffff, sum_32, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = sum_32;
    }
    __syncthreads();


    //结果数组归约
    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
           sum  += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }
    //最终值
    if (tid == 0)
        atomicAdd(y, sum);
}

#define HALF_ZERO __float2half(0.f)

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half* x, float* y, int N){
    //idx
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    int tid = threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float  reduce_smem[NUM_WARPS];

    half pack_x[8];
    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

        half sum_fp16;
#pragma unroll
        for(int i = 0 ; i < 8; i++){
            sum_fp16 += pack_x[i];
        }

        int warp = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;

#pragma unroll
        for(int offset = 16; offset > 0 ; offset >>= 1){
            sum_fp16  += __shfl_xor_sync(0xffffffff, sum_fp16, offset);
        }

        if(lane == 0){
            reduce_smem[warp] = __half2float(sum_fp16);
        }
        __syncthreads();

        //结果数组归约
        float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
        if (warp == 0){
    #pragma unroll
            for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
                sum  += __shfl_xor_sync(0xffffffff, sum, offset);
            }
        }
        //最终值
        if (tid == 0)
            atomicAdd(y, sum);
    }
    
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half* x, float* y, int N){
    //idx
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    int tid = threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float  reduce_smem[NUM_WARPS];
    half pack_x[8];
    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

        float sum_fp32 = 0.0f;
#pragma unroll
        for(int i = 0 ; i < 8; i++){
            sum_fp32 += __half2float(pack_x[i]);
        }
        
        int warp = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;

#pragma unroll
        for(int offset = 16; offset > 0 ; offset >>= 1){
            sum_fp32  += __shfl_xor_sync(0xffffffff, sum_fp32, offset);
        }

        if(lane == 0){
            reduce_smem[warp] = sum_fp32;
        }
        __syncthreads();

        //结果数组归约
        float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
        if (warp == 0){
    #pragma unroll
            for(int offset = NUM_WARPS >> 1; offset > 0 ; offset >>= 1){
                sum  += __shfl_xor_sync(0xffffffff, sum, offset);
            }
        }
        //最终值
        if (tid == 0)
            atomicAdd(y, sum);
    }
}


//__bfloat162float
//__float2bfloat16
#define BF_ZERO __float2bfloat16(0.0f)

///////////////////////////// bf16 ///////////////////////////////////
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

    __nv_bfloat16 sum_bf16 = idx < N ? x[idx] : BF_ZERO;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    //每个warp单独归约，存储到结果数组reduce_smem
#pragma unroll
    for(int offset = WARP_SIZE >> 1 ; offset > 0; offset >>= 1){
        sum_bf16 = __hadd(sum_bf16, __shfl_xor_sync(0xffffffff, sum_bf16, offset));
    }

    if(lane == 0){
        reduce_smem[warp] = sum_bf16;
    }
    __syncthreads();

    __nv_bfloat16 sum = (lane < NUM_WARPS) ? reduce_smem[lane] : BF_ZERO;

    if(warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
            sum = __hadd(sum, __shfl_xor_sync(0xffffffff, sum, offset));
        }
    }

    if(tid == 0){
        atomicAdd(y, __bfloat162float(sum));
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float sum_f32 = idx < N ? __bfloat162float(x[idx]) : 0.0f;

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    //每个warp单独归约，存储到结果数组reduce_smem
#pragma unroll
    for(int offset = WARP_SIZE >> 1 ; offset > 0; offset >>= 1){
        sum_f32 += __shfl_xor_sync(0xffffffff, sum_f32, offset);
    }

    if(lane == 0){
        reduce_smem[warp] = sum_f32;
    }
    __syncthreads();

    float sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;

    if(warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
    }

    if(tid == 0){
        atomicAdd(y, sum);
    }

}


template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

    __nv_bfloat16 pack_x[8];
    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

        __nv_bfloat16 sum_bf16 = BF_ZERO;
#pragma unroll
        for(int i = 0; i < 8; i++){
            sum_bf16 += pack_x[i];
        }

#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>=1){
            sum_bf16 = __hadd(sum_bf16, __shfl_xor_sync(0xffffffff, sum_bf16, offset));
        }

        if(lane == 0){
            reduce_smem[warp] = sum_bf16;
        }
        __syncthreads();

        __nv_bfloat16 sum = lane < NUM_WARPS ? reduce_smem[lane] : BF_ZERO;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1 ; offset > 0; offset >>=1){
                sum = __hadd(sum, __shfl_xor_sync(0xffffffff, sum, offset));
            }
        }

        if(tid == 0){
            atomicAdd(y, __bfloat162float(sum));
        }    
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    __nv_bfloat16 pack_x[8];
    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

        __nv_bfloat16 sum_bf16 = BF_ZERO;

        float sum_f32 = 0.0f;
#pragma unroll
        for(int i = 0; i < 8; i++){
            sum_f32 +=  __bfloat162float(pack_x[i]);
        }

#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>=1){
            sum_f32 += __shfl_xor_sync(0xffffffff, sum_f32, offset);
        }

        if(lane == 0){
            reduce_smem[warp] = sum_f32;
        }
        __syncthreads();

        float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1 ; offset > 0; offset >>=1){
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
        }

        if(tid == 0){
            atomicAdd(y, sum);
        }    
    }
}



template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16* x, float* y, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

    if(idx + 1 < N){
        __nv_bfloat162 reg_x = reinterpret_cast<__nv_bfloat162*>(x + idx)[0];
        __nv_bfloat16 sum_bf16 = reg_x.x + reg_x.y;

#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>=1){
           sum_bf16 = __hadd(sum_bf16, __shfl_xor_sync(0xffffffff, sum_bf16, offset));
        }
        if(lane == 0){
            reduce_smem[warp] = sum_bf16;
        }
        __syncthreads();

        __nv_bfloat16 sum = lane < NUM_WARPS ? reduce_smem[lane] : BF_ZERO;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1 ; offset > 0; offset >>=1){
                sum = __hadd(sum, __shfl_xor_sync(0xffffffff, sum, offset));
            }
        }

        if(tid == 0){
            atomicAdd(y, __bfloat162float(sum));
        }
    }

}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16* x, float* y, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    if(idx + 1 < N){
        __nv_bfloat162 reg_x = reinterpret_cast<__nv_bfloat162*>(x + idx)[0];
        float sum_f32 = __bfloat162float(reg_x.x + reg_x.y);

#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>=1){
            sum_f32 += __shfl_xor_sync(0xffffffff, sum_f32, offset);
        }
        if(lane == 0){
            reduce_smem[warp] = sum_f32;
        }
        __syncthreads();

        float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1 ; offset > 0; offset >>=1){
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }
        }

        if(tid == 0){
            atomicAdd(y, sum);
        }
    }
}

template <const int NUM_THREADS = 256>
__global void block_all_reduce_sum_fp8_e4m3_f16_kernel(){

}

template <const int NUM_THREADS = 256>
__global void block_all_reduce_sum_fp8_e5m2_f16_kernel(){

}

template <const int NUM_THREADS = 256 / 16>
__global void block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(){

}

template <const int NUM_THREADS = 256 / 16>
__global void block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(){

}

template <const int NUM_THREADS = 256>
__global void block_all_reduce_sum_i8_i32_kernel(){}


template <const int NUM_THREADS = 256 / 16>
__global void block_all_reduce_sum_i8x16_pack_i32_kernel(){}



#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type,          \
                             out_type)                                         \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)>               \
      <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                        reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,     \
                             n_elements, out_type)                             \
  torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(               \
      torch::Tensor x) {                                                       \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto y_th_type =                                                           \
        (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;           \
    auto options =                                                             \
        torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(1024 / (n_elements));                                         \
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /          \
                                                               (n_elements)>   \
          <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),    \
                            reinterpret_cast<out_type *>(y.data_ptr()), N);    \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(1024 / (n_elements));                                       \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /        \
                                                                 (n_elements)> \
            <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),  \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32x1, f32, torch::kFloat32, float, 1, float)
TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, float, 1, float)
TORCH_BINDING_REDUCE(f32x4, f32, torch::kFloat32, float, 4, float)
TORCH_BINDING_REDUCE(f16, f16, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16, f32, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16x2, f16, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x2, f32, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x8_pack, f16, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(f16x8_pack, f32, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(bf16, bf16, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16, f32, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16x2, bf16, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x2, f32, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x8_pack, bf16, torch::kBFloat16, __nv_bfloat16, 8,
                     float)
TORCH_BINDING_REDUCE(bf16x8_pack, f32, torch::kBFloat16, __nv_bfloat16, 8,
                     float)
// TORCH_BINDING_REDUCE(fp8_e4m3, f16, torch::kFloat8_e4m3fn, __nv_fp8_storage_t,
//                      1, float)
// TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16, torch::kFloat8_e4m3fn,
//                      __nv_fp8_storage_t, 16, float)
// TORCH_BINDING_REDUCE(fp8_e5m2, f16, torch::kFloat8_e5m2, __nv_fp8_storage_t, 1,
//                      float)
// TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16, torch::kFloat8_e5m2,
//                      __nv_fp8_storage_t, 16, float)
// TORCH_BINDING_REDUCE(i8, i32, torch::kInt8, int8_t, 1, int32_t)
// TORCH_BINDING_REDUCE(i8x16_pack, i32, torch::kInt8, int8_t, 16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x1_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
//   TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
//   TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
//   TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
//   TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
//   TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
//   TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}