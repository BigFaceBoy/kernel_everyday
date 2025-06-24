#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#include "../common.h"


__global__ void relu_f32_kernel(float* x, float* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = fmaxf(0.0f, x[idx]);

    }
}
__global__ void  relu_f32x4_kernel(float* x, float* y, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 3 < N){
        float4 reg_x = (reinterpret_cast<float4*>(x + idx))[0];
        float4 reg_y;

        reg_y.x = fmaxf(0.0f, reg_x.x);
        reg_y.y = fmaxf(0.0f, reg_x.y);
        reg_y.z = fmaxf(0.0f, reg_x.z);
        reg_y.w = fmaxf(0.0f, reg_x.w);
        (reinterpret_cast<float4*>(y + idx))[0] = reg_y;
    }

}
__global__ void  relu_f16_kernel(half *x, half *y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = __hmax(__float2half(0.0f), x[idx]);
    }
}
__global__ void  relu_f16x2_kernel(half *x, half *y, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 1 < N){
        half2 reg_x = (reinterpret_cast<half2*>(x + idx))[0];
        half2 reg_y;
        reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
        reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
        (reinterpret_cast<half2*>(y + idx))[0] = reg_y;
    }
}
__global__ void  relu_f16x8_kernel(half *x, half *y, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 7 < N){
        half2 reg_x1 = (reinterpret_cast<half2*>(x + idx + 0))[0];
        half2 reg_x2 = (reinterpret_cast<half2*>(x + idx + 2))[0];
        half2 reg_x3 = (reinterpret_cast<half2*>(x + idx + 4))[0];
        half2 reg_x4 = (reinterpret_cast<half2*>(x + idx + 6))[0];
    
        half2 reg_y1;
        half2 reg_y2;
        half2 reg_y3;
        half2 reg_y4;

        reg_y1.x = __hmax(__float2half(0.0f), reg_x1.x);
        reg_y1.y = __hmax(__float2half(0.0f), reg_x1.y);
        reg_y2.x = __hmax(__float2half(0.0f), reg_x2.x);
        reg_y2.y = __hmax(__float2half(0.0f), reg_x2.y);
        reg_y3.x = __hmax(__float2half(0.0f), reg_x3.x);
        reg_y3.y = __hmax(__float2half(0.0f), reg_x3.y);
        reg_y4.x = __hmax(__float2half(0.0f), reg_x4.x);
        reg_y4.y = __hmax(__float2half(0.0f), reg_x4.y);
        reinterpret_cast<half2*>(y + idx)[0] = reg_y1;
        reinterpret_cast<half2*>(y + idx + 2)[0] = reg_y2;
        reinterpret_cast<half2*>(y + idx + 4)[0] = reg_y3;
        reinterpret_cast<half2*>(y + idx + 6)[0] = reg_y4;
    }
}
__global__ void  relu_f16x8_pack_kernel(half *x, half *y, int N){
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]); 

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_y[i] = __hmax(__float2half(0.0f),pack_x[i]);
    }

    if ((idx + 7) < N) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

// --------------------- PyTorch bindings for custom kernel
// -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)     \
  void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                  \
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
      relu_##packed_type##_kernel<<<grid, block>>>(                            \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        relu_##packed_type##_kernel<<<grid, block>>>(                          \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_RELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_RELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_RELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_RELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_RELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(relu_f32)
  TORCH_BINDING_COMMON_EXTENSION(relu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(relu_f16x8_pack)
}