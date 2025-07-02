#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include "../common.h"


#define lambda 0.5f
#define half_lambda __float2half(0.5f)

__device__ __forceinline__ float hardshrink(float x){
    if(x < -lambda){
        return x;
    }else if(x > lambda){
        return x;
    }else{
        return 0;
    }
}

__device__ __forceinline__ half hardshrink_half(half x){
    if(x < -half_lambda){
        return x;
    }else if(x > half_lambda){
        return x;
    }else{
        return 0;
    }
}

__global__ void hardshrink_f32_kernel(float*x, float* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = hardshrink(x[idx]);
    }
}
__global__ void hardshrink_f32x4_kernel(float*x, float* y, int N){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 1 < N){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        float4 reg_y;
        reg_y.x = hardshrink(reg_x.x);
        reg_y.y = hardshrink(reg_x.y);
        reg_y.z = hardshrink(reg_x.z);
        reg_y.w = hardshrink(reg_x.w);

        reinterpret_cast<float4*>(y + idx)[0] = reg_y;
    }
}

__global__ void hardshrink_f16_kernel(half* x, half* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = hardshrink_half(x[idx]);
    }
}

__global__ void hardshrink_f16x2_kernel(half* x, half* y, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 1 < N){
        half2 reg_x = reinterpret_cast<half2*>(x + idx)[0];
        half2 reg_y;
        reg_y.x =  hardshrink_half(reg_x.x);
        reg_y.y =  hardshrink_half(reg_x.y);

        reinterpret_cast<half2*>(y + idx)[0] = reg_y;
    }
}

__global__ void hardshrink_f16x8_kernel(half* x, half* y, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx +7 < N){
        half2 reg_x1 = reinterpret_cast<half2*>(x + idx + 0)[0];
        half2 reg_x2 = reinterpret_cast<half2*>(x + idx + 2)[0];
        half2 reg_x3 = reinterpret_cast<half2*>(x + idx + 4)[0];
        half2 reg_x4 = reinterpret_cast<half2*>(x + idx + 6)[0];

        half2 reg_y1, reg_y2, reg_y3,reg_y4;
        reg_y1.x = hardshrink_half(reg_x1.x);
        reg_y1.y = hardshrink_half(reg_x1.y);
        reg_y2.x = hardshrink_half(reg_x2.x);
        reg_y2.y = hardshrink_half(reg_x2.y);
        reg_y3.x = hardshrink_half(reg_x3.x);
        reg_y3.y = hardshrink_half(reg_x3.y);
        reg_y4.x = hardshrink_half(reg_x4.x);
        reg_y4.y = hardshrink_half(reg_x4.y);

        reinterpret_cast<half2*>(y + idx + 0)[0] = reg_y1;
        reinterpret_cast<half2*>(y + idx + 2)[0] = reg_y2;
        reinterpret_cast<half2*>(y + idx + 4)[0] = reg_y3;
        reinterpret_cast<half2*>(y + idx + 6)[0] = reg_y4;
    }
}

__global__ void hardshrink_f16x8_pack_kernel(half* x, half* y, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for(int i = 0; i < 8; i++){
        pack_y[i] = hardshrink_half(pack_x[i]);
    }

    if(idx + 7 < N){
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

// 定义 CHECK_TORCH_TENSOR_DTYPE 宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("Tensor dtype must be " #th_type);                \
  }

// 定义 TORCH_BINDING_COMMON_EXTENSION 宏
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define TORCH_BINDING_HARDSHRINK(packed_type, th_type, element_type,           \
                                 n_elements)                                   \
  void hardshrink_##packed_type(torch::Tensor x, torch::Tensor y) {            \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      hardshrink_##packed_type##_kernel<<<grid, block>>>(                      \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        hardshrink_##packed_type##_kernel<<<grid, block>>>(                    \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        hardshrink_##packed_type##_kernel<<<grid, block>>>(                    \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_HARDSHRINK(f32, torch::kFloat32, float, 1)
TORCH_BINDING_HARDSHRINK(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_HARDSHRINK(f16, torch::kHalf, half, 1)
TORCH_BINDING_HARDSHRINK(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_HARDSHRINK(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_HARDSHRINK(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hardshrink_f32)
  TORCH_BINDING_COMMON_EXTENSION(hardshrink_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16)
  TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(hardshrink_f16x8_pack)
}