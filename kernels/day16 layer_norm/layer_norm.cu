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
__device__ half block_reduce_sum_f16(half val) {
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];

#pragma unroll
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    if(lane == 0){
        reduce_smem[warp] = val;
    }
    __syncthreads();

    half sum = lane < NUM_WARPS ? reduce_smem[lane]: __float2half(0.0f);

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
    __syncthreads();
    float v = (x_i - mu) * (x_i - mu);
    float sum_sigma =  block_reduce_sum_f32<NUM_THREADS>(v);
    if(tid == 0){
        sigma = sqrtf(sum_sigma / (float)K + epsilon);
    }
    __syncthreads();

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
        __syncthreads();

        float mean_x_pow = (reg_x.x - mu) * (reg_x.x - mu);
        float mean_y_pow = (reg_x.y - mu) * (reg_x.y - mu);
        float mean_z_pow = (reg_x.z - mu) * (reg_x.z - mu);
        float mean_w_pow = (reg_x.w - mu) * (reg_x.w - mu);

        float mean_sum = mean_x_pow + mean_y_pow + mean_z_pow + mean_w_pow;
        float sum_sigma =  block_reduce_sum_f32<NUM_THREADS>(mean_sum);
        if(tid == 0){
            sigma = sqrtf(sum_sigma / (float)K + epsilon);
        }
        __syncthreads();

        float4 reg_y;
        reg_y.x = ((reg_x.x - mu) / sigma) * g + b;
        reg_y.y = ((reg_x.y - mu) / sigma) * g + b;
        reg_y.z = ((reg_x.z - mu) / sigma) * g + b;
        reg_y.w = ((reg_x.w - mu) / sigma) * g + b;

        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
    }
}


template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half* x, half* y, float g, float b, int N, int K){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ half mu;
    __shared__ half sigma;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half K_ = __int2half_rn(K);
    if(idx < N * K){
        half x_i = x[idx];
        half sum_x = block_reduce_sum_f16<NUM_THREADS>(x_i);
        if(tid == 0){
            mu = sum_x / K_;
        }
        __syncthreads();

        half x_i_pow = (x_i - mu) * (x_i - mu);
        half sum_sigma = block_reduce_sum_f16<NUM_THREADS>(x_i_pow);
        if(tid == 0){
            sigma = hsqrt(sum_sigma / K_ + epsilon);
        }
        __syncthreads();

        y[idx] = ((x_i - mu) / sigma) * g_ + b_;
    }
}

template <const int NUM_THREADS = 256/2>
__global__ void layer_norm_f16x2_f16_kernel(half* x, half* y, float g, float b, int N, int K){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;
    __shared__ half mu;
    __shared__ half sigma;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half K_ = __int2half_rn(K);
    if(idx + 1 < N * K){
        half2 reg_x = reinterpret_cast<half2*>(x + idx)[0];
        half sum_reg = reg_x.x + reg_x.y;
        half sum_x = block_reduce_sum_f16<NUM_THREADS>(sum_reg);
        if(tid == 0){
            mu = sum_x / K_;
        }
        __syncthreads();

        half x_i_pow = (reg_x.x - mu) * (reg_x.x - mu) + (reg_x.y - mu) * (reg_x.y - mu);
        half sum_sigma = block_reduce_sum_f16<NUM_THREADS>(x_i_pow);
        if(tid == 0){
            sigma = hsqrt(sum_sigma / K_ + epsilon);
        }
        __syncthreads();

        half2 reg_y;
        reg_y.x = ((reg_x.x - mu) / sigma) * g_ + b_;
        reg_y.y = ((reg_x.y - mu) / sigma) * g_ + b_;
        reinterpret_cast<half2*>(y + idx)[0] = reg_y;
    }
}

template <const int NUM_THREADS = 256/8>
__global__ void layer_norm_f16x8_f16_kernel(half* x, half* y, float g, float b, int N, int K){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;
    __shared__ half mu;
    __shared__ half sigma;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half K_ = __int2half_rn(K);
    if(idx + 7 < N * K){
        half2 reg_x1 = reinterpret_cast<half2*>(x + idx)[0];
        half2 reg_x2 = reinterpret_cast<half2*>(x + idx + 2)[0];
        half2 reg_x3 = reinterpret_cast<half2*>(x + idx + 4)[0];
        half2 reg_x4 = reinterpret_cast<half2*>(x + idx + 6)[0];

        half sum_reg = reg_x1.x + reg_x1.y + 
                       reg_x2.x + reg_x2.y +
                       reg_x3.x + reg_x3.y +
                       reg_x4.x + reg_x4.y;
        half sum_x = block_reduce_sum_f16<NUM_THREADS>(sum_reg);
        if(tid == 0){
            mu = sum_x / K_;
        }
        __syncthreads();

        half x_i_pow = (reg_x1.x - mu) * (reg_x1.x - mu) + (reg_x1.y - mu) * (reg_x1.y - mu) +
                       (reg_x2.x - mu) * (reg_x2.x - mu) + (reg_x2.y - mu) * (reg_x2.y - mu) +
                       (reg_x3.x - mu) * (reg_x3.x - mu) + (reg_x3.y - mu) * (reg_x3.y - mu) +
                       (reg_x4.x - mu) * (reg_x4.x - mu) + (reg_x4.y - mu) * (reg_x4.y - mu);
        half sum_sigma = block_reduce_sum_f16<NUM_THREADS>(x_i_pow);
        if(tid == 0){
            sigma = hsqrt(sum_sigma / K_ + epsilon);
        }
        __syncthreads();

        half2 reg_y1, reg_y2, reg_y3, reg_y4;

        reg_y1.x = ((reg_x1.x - mu) / sigma) * g_ + b_;
        reg_y1.y = ((reg_x1.y - mu) / sigma) * g_ + b_;

        reg_y2.x = ((reg_x2.x - mu) / sigma) * g_ + b_;
        reg_y2.y = ((reg_x2.y - mu) / sigma) * g_ + b_;

        reg_y3.x = ((reg_x3.x - mu) / sigma) * g_ + b_;
        reg_y3.y = ((reg_x3.y - mu) / sigma) * g_ + b_;

        reg_y4.x = ((reg_x4.x - mu) / sigma) * g_ + b_;
        reg_y4.y = ((reg_x4.y - mu) / sigma) * g_ + b_;

        reinterpret_cast<half2*>(y + idx)[0] = reg_y1;
        reinterpret_cast<half2*>(y + idx + 2)[0] = reg_y2;
        reinterpret_cast<half2*>(y + idx + 4)[0] = reg_y3;
        reinterpret_cast<half2*>(y + idx + 6)[0] = reg_y4;
    }
}

template <const int NUM_THREADS = 256/8>
__global__ void layer_norm_f16x8_pack_f16_kernel(half* x, half* y, float g, float b, int N, int K){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;
    __shared__ half mu;
    __shared__ half sigma;
    const half epsilon = __float2half(1e-5f);
    const half g_ = __float2half(g);
    const half b_ = __float2half(b);
    const half K_ = __int2half_rn(K);

    half pack_x[8], pack_y[8];

    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
        half sum_pack = __float2half(0.0f);
        for(int i = 0; i < 8; i++){
            sum_pack += pack_x[i];
        }
        half sum_x = block_reduce_sum_f16<NUM_THREADS>(sum_pack);
        if(tid == 0){
            mu = sum_x / K_;
        }
        __syncthreads();

        half x_i_pow =__float2half(0.0f);
        for(int i = 0; i < 8; i++){
            x_i_pow += ((pack_x[i] - mu) * (pack_x[i] - mu));
        }

        half sum_sigma = block_reduce_sum_f16<NUM_THREADS>(x_i_pow);
        if(tid == 0){
            sigma = hsqrt(sum_sigma / K_ + epsilon);
        }
        __syncthreads();

        
        for(int i = 0; i < 8; i++){
            pack_y[i] = ((pack_x[i] - mu) / sigma) * g_ + b_;
        }

        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

template <const int NUM_THREADS = 256/8>
__global__ void layer_norm_f16x8_pack_f32_kernel(half* x, half* y, float g, float b, int N, int K){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;
    __shared__ float mu;
    __shared__ float sigma;
    const float epsilon = 1e-5f;

    half pack_x[8], pack_y[8];

    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
        float sum_pack = 0.0f;
        for(int i = 0; i < 8; i++){
            sum_pack += __half2float(pack_x[i]);
        }
        float sum_x = block_reduce_sum_f32<NUM_THREADS>(sum_pack);
        if(tid == 0){
            mu = sum_x / (float)K;
        }
        __syncthreads();

        float x_i_pow = 0.0f;
        for(int i = 0; i < 8; i++){
            x_i_pow += ((__half2float(pack_x[i]) - mu) * (__half2float(pack_x[i]) - mu));
        }

        float sum_sigma = block_reduce_sum_f32<NUM_THREADS>(x_i_pow);
        if(tid == 0){
            sigma = sqrt(sum_sigma / (float)K + epsilon);
        }
        __syncthreads();

        
        for(int i = 0; i < 8; i++){
            pack_y[i] = __float2half(((__half2float(pack_x[i]) - mu) / sigma) * g + b);
        }

        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half* x, half* y, float g, float b, int N, int K){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float mu;
    __shared__ float sigma;
    const float epsilon = 1e-5f;
    if(idx < N * K){
        float x_i = __half2float(x[idx]);
        float sum_x = block_reduce_sum_f32<NUM_THREADS>(x_i);
        if(tid == 0){
            mu = sum_x / (float)K;
        }
        __syncthreads();

        float x_i_pow = (x_i - mu) * (x_i - mu);
        float sum_sigma = block_reduce_sum_f32<NUM_THREADS>(x_i_pow);
        if(tid == 0){
            sigma = sqrt(sum_sigma / (float)K + epsilon);
        }
        __syncthreads();

        y[idx] = __float2half(((x_i - mu) / sigma) * g + b);
    }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// fp32
#define LANUCH_LAYER_NORM_F32_KERNEL(K)                                        \
  layer_norm_f32_kernel<(K)><<<grid, block>>>(                                 \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F32_KERNEL(N, K)                                   \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32_KERNEL(64)                                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(128)                                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(256)                                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(512)                                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32_KERNEL(1024)                                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F32x4_KERNEL(K)                                      \
  layer_norm_f32x4_kernel<(K) / 4><<<grid, block>>>(                           \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)                                 \
  dim3 block((K) / 4);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32x4_KERNEL(64) break;                                  \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(128) break;                                 \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(256) break;                                 \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(512) break;                                 \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(1024) break;                                \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(2048) break;                                \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(4096) break;                                \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// fp16
#define LANUCH_LAYER_NORM_F16F16_KERNEL(K)                                     \
  layer_norm_f16_f16_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)                                \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F16_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F16_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16F32_KERNEL(K)                                     \
  layer_norm_f16_f32_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)                                \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F32_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F32_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x2F16_KERNEL(K)                                   \
  layer_norm_f16x2_f16_kernel<(K) / 2>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)                              \
  dim3 block((K) / 2);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(2048) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8F16_KERNEL(K)                                   \
  layer_norm_f16x8_f16_kernel<(K) / 8>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)                              \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(2048) break;                             \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(4096) break;                             \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(8192) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(K)                             \
  layer_norm_f16x8_pack_f16_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)                        \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(K)                             \
  layer_norm_f16x8_pack_f32_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)                        \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

void layer_norm_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F32_KERNEL(N, K)
}

void layer_norm_f32x4(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)
}

void layer_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)
}

void layer_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)
}

void layer_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)
}

void layer_norm_f16x8_pack_f16(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)
}

void layer_norm_f16x8_pack_f32(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)
}

void layer_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int N = x.size(0);
  const int K = x.size(1);
  DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f32)
}