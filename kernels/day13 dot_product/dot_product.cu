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
__global__ void dot_prod_f32_f32_kernel(float* x, float* y, float* out, int N){
    //idx
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float mul = idx < N ? x[idx] * y[idx] : 0.0f;
    __syncthreads();
#pragma unroll
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>=1){
        mul += __shfl_xor_sync(0xffffffff, mul, offset);
    }


    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if(lane == 0){
        reduce_smem[warp] = mul;
    }
    __syncthreads();

    float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
    if(warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0; offset >>=1){
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        } 
    }
    if(tid == 0){
        atomicAdd(out, sum);
    }
}


template <const int NUM_THREADS = 256/4>
__global__ void dot_prod_f32x4_f32_kernel(float* x, float* y, float* out, int N){
    int tid = threadIdx.x;
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if(idx + 3 < N){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        float4 reg_y = reinterpret_cast<float4*>(y + idx)[0];

        float mul = reg_x.x * reg_y.x + reg_x.y * reg_y.y + reg_x.z * reg_y.z + reg_x.w * reg_y.w;
#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
            mul += __shfl_xor_sync(0xffffffff, mul, offset);
        }

        if(lane == 0){
            reduce_smem[warp] = mul;
        }

        __syncthreads();

        float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }  
        }

        if(tid == 0){
            atomicAdd(out, sum);
        }
    }
}

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f16_f32_kernel(half* x, half* y, float* out, int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    float mul = idx < N ? __half2float(x[idx]) * __half2float(y[idx]) : 0.0f;
#pragma unroll
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
        mul += __shfl_xor_sync(0xffffffff, mul, offset);
    }
    
    if(lane == 0){
        reduce_smem[warp] = mul;
    }
    __syncthreads();

    float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
    if(warp == 0){
#pragma unroll
        for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }  
    }

    if(tid == 0){
        atomicAdd(out, sum);
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void dot_prod_f16x2_f32_kernel(half* x, half* y, float* out, int N){
    int tid = threadIdx.x;
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if(idx + 1 < N){
        half2 reg_x = reinterpret_cast<half2*>(x + idx)[0];
        half2 reg_y = reinterpret_cast<half2*>(y + idx)[0];

        float mul = __half2float(reg_x.x * reg_y.x + reg_x.y * reg_y.y);
#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
            mul += __shfl_xor_sync(0xffffffff, mul, offset);
        }

        if(lane == 0){
            reduce_smem[warp] = mul;
        }

        __syncthreads();

        float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }  
        }

        if(tid == 0){
            atomicAdd(out, sum);
        }
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half* x, half* y, float* out, int N){
    int tid = threadIdx.x;
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    half pack_x[8], pack_y[8];

    if(idx + 7 < N){
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
        LDST128BITS(pack_y[0]) = LDST128BITS(y[idx]);

        float mul = 0.0f;
        for(int i = 0; i < 8; i++){
            mul += __half2float(pack_x[i] * pack_y[i]);
        }

#pragma unroll
        for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1){
            mul += __shfl_xor_sync(0xffffffff, mul, offset);
        }

        if(lane == 0){
            reduce_smem[warp] = mul;
        }

        __syncthreads();

        float sum = lane < NUM_WARPS ? reduce_smem[lane] : 0.0f;
        if(warp == 0){
#pragma unroll
            for(int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1){
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            }  
        }

        if(tid == 0){
            atomicAdd(out, sum);
        }
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

#define LANUCH_DOT_PROD_KERNEL(NT, packed_type, acc_type, element_type)        \
  dot_prod_##packed_type##_##acc_type##_kernel<(NT)>                           \
      <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),        \
                        reinterpret_cast<element_type *>(b.data_ptr()),        \
                        prod.data_ptr<float>(), N);

#define DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_DOT_PROD_KERNEL(32, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_DOT_PROD_KERNEL(64, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_DOT_PROD_KERNEL(128, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_DOT_PROD_KERNEL(256, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_DOT_PROD_KERNEL(512, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_DOT_PROD_KERNEL(1024, packed_type, acc_type, element_type)          \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_DOT_PROD(packed_type, acc_type, th_type, element_type,   \
                               n_elements)                                     \
  torch::Tensor dot_prod_##packed_type##_##acc_type(torch::Tensor a,           \
                                                    torch::Tensor b) {         \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0); \
    auto prod = torch::zeros({1}, options);                                    \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256);                                                         \
      dim3 grid(((N + 256 - 1) / 256) / (n_elements));                         \
      dot_prod_##packed_type##_##acc_type##_kernel<256>                        \
          <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),    \
                            reinterpret_cast<element_type *>(b.data_ptr()),    \
                            prod.data_ptr<float>(), N);                        \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256);                                                       \
        dim3 grid(((N + 256 - 1) / 256) / (n_elements));                       \
        dot_prod_##packed_type##_##acc_type##_kernel<256>                      \
            <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),  \
                              reinterpret_cast<element_type *>(b.data_ptr()),  \
                              prod.data_ptr<float>(), N);                      \
      }                                                                        \
    }                                                                          \
    return prod;                                                               \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_DOT_PROD(f32, f32, torch::kFloat32, float, 1)
TORCH_BINDING_DOT_PROD(f32x4, f32, torch::kFloat32, float, 4)
TORCH_BINDING_DOT_PROD(f16, f32, torch::kHalf, half, 1)
TORCH_BINDING_DOT_PROD(f16x2, f32, torch::kHalf, half, 2)
TORCH_BINDING_DOT_PROD(f16x8_pack, f32, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x8_pack_f32)
}