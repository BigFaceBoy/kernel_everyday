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
__device__ float block_reduce_max_f32(float val) {
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

#pragma unroll
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    if(lane == 0){
        reduce_smem[warp] = val;
    }
    __syncthreads();

    float max = lane < NUM_WARPS ? reduce_smem[lane]: -FLT_MAX;

#pragma unroll
    for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
        max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, offset));
    }


    max = __shfl_sync(0xffffffff, max, 0, 32);
    return max;
}


// 每个block 独立计算softmax
template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float exp_val = idx < N ? expf(x[idx]) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    
    if(idx < N)
        y[idx] = exp_val / exp_sum;
}


template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N){
    int tid = threadIdx.x;
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

    if(idx + 3 < N){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];

        float4 reg_exp;
        reg_exp.x = expf(reg_x.x);
        reg_exp.y = expf(reg_x.y);
        reg_exp.z = expf(reg_x.z);
        reg_exp.w = expf(reg_x.w);

        float exp_val = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
        float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

        float4 reg_y;
        reg_y.x = reg_exp.x / exp_sum;
        reg_y.y = reg_exp.y / exp_sum;
        reg_y.z = reg_exp.z / exp_sum;
        reg_y.w = reg_exp.w / exp_sum;

        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
    }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        //max
        float max_val = block_reduce_max_f32<NUM_THREADS>(x[idx]);

        // e^(x-max)
        float exp_val = expf(x[idx] - max_val);

        // sum e^(x-max)
        float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);

        // e^(x-max) / sum
        y[idx] = exp_val / exp_sum;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N){
    //idx
    int idx = 4 *(blockDim.x * blockIdx.x + threadIdx.x);
    
    if(idx + 3 < N){
        //load 4 f32
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        //max
        float max = fmaxf(fmaxf(fmaxf(reg_x.x, reg_x.y), reg_x.z), reg_x.w);
        float max_val = block_reduce_max_f32<NUM_THREADS>(max);
        //e^(x-max)
        float4 reg_exp;
        reg_exp.x = expf(reg_x.x - max_val);
        reg_exp.y = expf(reg_x.y - max_val);
        reg_exp.z = expf(reg_x.z - max_val);
        reg_exp.w = expf(reg_x.w - max_val);
        // sum e^(x-max)
        float sum = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
        float exp_sum = block_reduce_sum_f32<NUM_THREADS>(sum);

        // e^(x-max) / sum
        float4 reg_y;
        reg_y.x = reg_exp.x / exp_sum;
        reg_y.y = reg_exp.y / exp_sum;
        reg_y.z = reg_exp.z / exp_sum;
        reg_y.w = reg_exp.w / exp_sum;

        //store 4 f32
        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
    }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N){
    //idx
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        //fp16->f32
        float val = __half2float(x[idx]);
        //max
        float max_val = block_reduce_max_f32<NUM_THREADS>(val);
        //e^(x-max)
        float exp_val = expf(val -max_val);
        // sum e^(x-max)
        float sum_val = block_reduce_sum_f32<NUM_THREADS>(exp_val);
        // e^(x-max) / sum
        y[idx] = __float2half_rn(exp_val / sum_val);

    }
}


template <const int NUM_THREADS = 256 / 2>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y, int N){
    //idx
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);

    if(idx + 1 < N){
        //load 2 f16
        half2 reg_x = reinterpret_cast<half2*>(x + idx)[0];
        //f16 -> f32
        float2 reg_val;
        reg_val.x = __half2float(reg_x.x);
        reg_val.y = __half2float(reg_x.y);
        //max 
        float max = fmaxf(reg_val.x, reg_val.y);
        float max_val = block_reduce_max_f32<NUM_THREADS>(max);
        //e^(x-max)
        float2 exp_val;
        exp_val.x = expf(reg_val.x - max_val); 
        exp_val.y = expf(reg_val.y - max_val); 
        // sum e^(x-max)
        float sum = exp_val.x + exp_val.y;
        float sum_val = block_reduce_sum_f32<NUM_THREADS>(sum);
        // e^(x-max) / sum
        float2 reg_y;
        reg_y.x = exp_val.x / sum_val;
        reg_y.y = exp_val.y / sum_val;
        //store 2 f32
        reinterpret_cast<half2*>(y + idx)[0] = __float22half2_rn(reg_y);
    }
}


template <const int NUM_THREADS = 256 / 8>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y, int N){
    //idx
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 7 < N){
        //pack  f16->f32
        half pack_x[8];
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

        float pack_val[8];
        float max = __half2float(pack_x[0]);
        for(int i = 0; i < 8; i++){
            pack_val[i] = __half2float(pack_x[i]);
            max = fmaxf(max, pack_val[i]);
        }
        float max_val = block_reduce_max_f32<NUM_THREADS>(max);

        //e^(x-max)  sum e^(x-max)
        float exp_val[8];
        float sum = 0.0f;
        for(int i = 0; i < 8; i++){
            exp_val[i] = expf(pack_val[i] - max_val);
            sum += exp_val[i];
        }
        float sum_val = block_reduce_sum_f32<NUM_THREADS>(sum);
        // e^(x-max) / sum
        half pack_y[8];
        for(int i = 0; i < 8; i++){
            pack_y[i] = __float2half_rn(exp_val[i]  / sum_val);
        }

        //store pack
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

struct MD{
    float m;
    float d;
};
template <const int NUM_THREADS = 256>
__device__ MD block_reduce_md_f32(MD md) {
    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ MD reduce_smem[NUM_WARPS];

    float m0 = md.m;
    float d0 = md.d;
#pragma unroll
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
        float m1 = __shfl_xor_sync(0xffffffff, m0, offset);
        float m_new = fmaxf(m0, m1);
        float d1 = __shfl_xor_sync(0xffffffff, d0, offset);
        float d_new = d0 * expf(m0) / expf(m_new) + d1 * expf(m1) / expf(m_new);
        m0 = m_new;
        d0 = d_new;
    }
    if(lane == 0){
        reduce_smem[warp].m = m0;
        reduce_smem[warp].d = d0;
    }
    __syncthreads();

    m0 = lane < NUM_WARPS ? reduce_smem[lane].m: 0.0f;
    d0 = lane < NUM_WARPS ? reduce_smem[lane].d: 0.0f;
    if(warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
            float m1 = __shfl_xor_sync(0xffffffff, m0, offset);
            float m_new = fmaxf(m0, m1);
            float d1 = __shfl_xor_sync(0xffffffff, d0, offset);
            float d_new = d0 * expf(m0) / expf(m_new) + d1 * expf(m1) / expf(m_new);
            m0 = m_new;
            d0 = d_new;
        }
    }

    MD md_n;
    md_n.m = m0;
    md_n.d = d0;

    return md_n;
}


//需要在一次warp 0for循环中把m_i  d_i 计算出来
template <const int NUM_THREADS = 256 / 8>
__global__ void online_safe_softmax_f32_per_token_kernel(float *x, float *y, int N){
    //idx
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if(idx < N){

        //load f32
        float val = x[idx];

        MD md;
        md.m = val;
        md.d = expf(val - md.m);
        //m  d
        MD result = block_reduce_md_f32<NUM_THREADS>(md);

        y[idx] = expf(val - result.m) / result.d;
    }

}



template <const int NUM_THREADS = 256 / 8>
__global__ void online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 3 < N){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        
        float max_val = fmaxf(fmaxf(fmaxf(reg_x.x, reg_x.y), reg_x.z), reg_x.w);
        float4 reg_exp;
        reg_exp.x =  expf(reg_x.x - max_val);
        reg_exp.y =  expf(reg_x.y - max_val);
        reg_exp.z =  expf(reg_x.z - max_val);
        reg_exp.w =  expf(reg_x.w - max_val);

        float sum = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;

        MD md;
        md.m = max_val;
        md.d = sum;

        MD result = block_reduce_md_f32<NUM_THREADS>(md);

        float4 reg_y;
        reg_y.x = expf(reg_x.x - result.m)/ result.d;
        reg_y.y = expf(reg_x.y - result.m)/ result.d;
        reg_y.z = expf(reg_x.z - result.m)/ result.d;
        reg_y.w = expf(reg_x.w - result.m)/ result.d;

        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
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

// grid memory fence
#define TORCH_BINDING_SOFTMAX(packed_type, th_type, element_type, n_elements)  \
  void softmax_##packed_type(torch::Tensor x, torch::Tensor y) {               \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype((th_type)).device(torch::kCUDA, 0);       \
    const int N = x.size(0);                                                   \
    CHECK_TORCH_TENSOR_SHAPE(x, y)                                             \
    auto total = torch::zeros({1}, options);                                   \
    dim3 block(256);                                                           \
    dim3 grid(((N + 256 - 1) / 256) / (n_elements));                           \
    softmax_##packed_type##_kernel<256><<<grid, block>>>(                      \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()),                        \
        reinterpret_cast<element_type *>(total.data_ptr()), N);                \
  }

// softmax per token
#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                                 \
  softmax_f32_per_token_kernel<(H)>                                            \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                            \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                                    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                                    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                                   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                                   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                                   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                                  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                               \
  softmax_f32x4_per_token_kernel<(H) / 4>                                      \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                          \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                           \
  case 64:                                                                     \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                           \
  case 128:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                          \
  case 256:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                          \
  case 512:                                                                    \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                          \
  case 1024:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                         \
  case 2048:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                         \
  case 4096:                                                                   \
    LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                         \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                            \
  safe_softmax_f32_per_token_kernel<(H)>                                       \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                       \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                               \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                               \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                              \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                              \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                              \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                             \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                          \
  online_safe_softmax_f32_per_token_kernel<(H)>                                \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                     \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                             \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                             \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                            \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                            \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                            \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                           \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

// online softmax per token
#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)                   \
  online_safe_softmax_f32x4_pack_per_token_kernel<(H / 4)>                     \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)              \
  dim3 block((H / 4));                                                         \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 128:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128)                     \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256)                     \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512)                     \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024)                    \
    break;                                                                     \
  case 2048:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048)                    \
    break;                                                                     \
  case 4096:                                                                   \
    LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096)                    \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 128/256/.../4096;");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                          \
  safe_softmax_f32x4_per_token_kernel<(H) / 4>                                 \
      <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),               \
                        reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                     \
  const int NT = (H) / 4;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                      \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                      \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;                    \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;                    \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;                    \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*4");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)                        \
  safe_softmax_f16_f32_per_token_kernel<(H)>                                   \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)                   \
  dim3 block((H));                                                             \
  dim3 grid((S));                                                              \
  switch ((H)) {                                                               \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32)                           \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64)                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128)                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256)                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512)                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024)                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)                      \
  safe_softmax_f16x2_f32_per_token_kernel<(H) / 2>                             \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)                 \
  const int NT = (H) / 2;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32) break;                  \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64) break;                  \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128) break;                 \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256) break;                 \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512) break;                 \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024) break;                \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048) break;                \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)                 \
  safe_softmax_f16x8_pack_f32_per_token_kernel<(H) / 8>                        \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)            \
  const int NT = (H) / 8;                                                      \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (H) {                                                                 \
  case 32:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32) break;             \
  case 64:                                                                     \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64) break;             \
  case 128:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128) break;            \
  case 256:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256) break;            \
  case 512:                                                                    \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512) break;            \
  case 1024:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024) break;           \
  case 2048:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048) break;           \
  case 4096:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096) break;           \
  case 8192:                                                                   \
    LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192) break;           \
  default:                                                                     \
    throw std::runtime_error("only support H: 64/128/.../1024*8");             \
    break;                                                                     \
  }

// per token fp32
void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

// per token fp16
void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0); // seqlens
  const int H = x.size(1); // head size/kv_len
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x,
                                              torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int S = x.size(0);
  const int H = x.size(1);
  const int N = S * H;
  DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}

// grid memory fence fp32
// TORCH_BINDING_SOFTMAX(f32,   torch::kFloat32, float, 1)
// TORCH_BINDING_SOFTMAX(f32x4, torch::kFloat32, float, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32)
  // TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token)
}