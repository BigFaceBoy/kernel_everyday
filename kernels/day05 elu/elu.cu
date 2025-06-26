#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include "../common.h"

#define alpha 1.0f

__global__ void elu_f32_kernel(float* x, float* y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = x[idx] >= 0 ? x[idx] : alpha * (expf(x[idx]) - 1.0f);
    }
}

__global__ void elu_f32x4_kernel(float* x, float* y, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 3 < N){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        float4 reg_y;
        reg_y.x = reg_x.x >= 0 ? reg_x.x : alpha * (expf(reg_x.x) - 1.0f);
        reg_y.y = reg_x.y >= 0 ? reg_x.y : alpha * (expf(reg_x.y) - 1.0f);
        reg_y.z = reg_x.z >= 0 ? reg_x.z : alpha * (expf(reg_x.z) - 1.0f);
        reg_y.w = reg_x.w >= 0 ? reg_x.w : alpha * (expf(reg_x.w) - 1.0f);
        (reinterpret_cast<float4*>(y + idx))[0] = reg_y;
    }
}
// const half half_1 = __float2half(1.0f);

// __global__ void elu_f16_kernel(half *x, half *y, int N){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < N){
//         y[idx] = x[idx] >= 0 ? x[idx] : alpha * (hexp(x[idx]) - half_1);
//     }
// }

// __global__ void elu_f16x2_kernel(half *x, half *y, int N){
//     int idx =  2 * (blockIdx.x * blockDim.x + threadIdx.x);
//     if(idx + 1 < N){
//         half2 reg_x = reinterpret_cast<half2*>(x)[idx];
//         half2 reg_y;

//         reg_y.x = reg_x.x >= 0 ? reg_x.x : alpha *  (hexp(reg_x.x) - half_1);
//         reg_y.y = reg_x.y >= 0 ? reg_x.y : alpha *  (hexp(reg_x.y) - half_1);

//         reinterpret_cast<half2*>(y)[idx] = reg_y;
//     }
// }
// 看了`示例`的代码才意识到half的所有计算都有专门的运算函数。

// ELU 计算函数
// -------------------------------------- FP32
// --------------------------------------
__device__ __forceinline__ float elu(float x) {
  return x > 0.f ? x : alpha * (expf(x) - 1.f);
}

// -------------------------------------- FP16
// --------------------------------------
__device__ __forceinline__ half elu_half(half x) {
  return __hgt(x, __float2half(0.f))
             ? x
             : __hmul(__float2half(alpha), __hsub(hexp(x), __float2half(1.f)));
}


__global__ void elu_f16_kernel(half *x, half *y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] =  elu_half(x[idx]);
    }
}


__global__ void elu_f16x2_kernel(half *x, half *y, int N){
    int idx =  2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 1 < N){
        //half2 reg_x = reinterpret_cast<half2*>(x)[idx]; //这种方式会崩溃
        half2 reg_x = (reinterpret_cast<half2*>(x + idx))[0];
        half2 reg_y;

        reg_y.x = elu_half(reg_x.x);
        reg_y.y = elu_half(reg_x.y);
        //reinterpret_cast<half2*>(y)[idx] = reg_y;

        (reinterpret_cast<half2*>(y + idx))[0] = reg_y;
    }
}

__global__ void elu_f16x8_kernel(half *x, half *y, int N){
    int idx =  8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 7 < N){
        half2 reg_x1 = (reinterpret_cast<half2*>(x + idx))[0];
        half2 reg_x2 = (reinterpret_cast<half2*>(x + idx + 2))[0];
        half2 reg_x3 = (reinterpret_cast<half2*>(x + idx + 4))[0];
        half2 reg_x4 = (reinterpret_cast<half2*>(x + idx + 6))[0];

        half2 reg_y1,reg_y2,reg_y3,reg_y4;
        reg_y1.x = elu_half(reg_x1.x);
        reg_y1.y = elu_half(reg_x1.y);
        reg_y2.x = elu_half(reg_x2.x);
        reg_y2.y = elu_half(reg_x2.y);
        reg_y3.x = elu_half(reg_x3.x);
        reg_y3.y = elu_half(reg_x3.y);
        reg_y4.x = elu_half(reg_x4.x);
        reg_y4.y = elu_half(reg_x4.y);

        reinterpret_cast<half2*>(y + idx)[0] = reg_y1;
        reinterpret_cast<half2*>(y + idx + 2)[0] = reg_y2;
        reinterpret_cast<half2*>(y + idx + 4)[0] = reg_y3;
        reinterpret_cast<half2*>(y + idx + 6)[0] = reg_y4;
    }
}

__global__ void elu_f16x8_pack_kernel(half *x, half *y, int N){
    int idx =  8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];

    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
    for(int i = 0; i < 8 ; i++){
        pack_y[i] = elu_half(pack_x[i]);
    }

    if(idx + 7 < N){
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}



// PyTorch 绑定代码
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

#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_elements)      \
  void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                   \
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
      elu_##packed_type##_kernel<<<grid, block>>>(                             \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELU(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELU(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELU(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELU(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELU(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELU(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elu_f32)
  TORCH_BINDING_COMMON_EXTENSION(elu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8_pack)
}