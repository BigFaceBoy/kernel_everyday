
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>


// c = a + b
// -------------------------------------- FP32
// -------------------------------------- ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = a[idx] + b[idx];
    }
}


// c = a + b fp32x4 向量化版本
// fp32x4 向量化是指每个线程处理4个连续的f32元素，可通过cuda 内置的 float4 来取值。
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32x4_kernel_bug(float *a, float *b, float *c, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        float4 reg_a = (reinterpret_cast<float4*>(a + idx))[0];
        float4 reg_b = (reinterpret_cast<float4*>(b + idx))[0];
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        c[idx] = reinterpret_cast<float*>(&reg_c)[0];
    }
}
// 这是我写完的第一个版本，对比后意识到我最后只写回x分量，y、z、w被丢弃了。
// 于是修改后为：
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        float4 reg_a = (reinterpret_cast<float4*>(a + idx))[0];
        float4 reg_b = (reinterpret_cast<float4*>(b + idx))[0];
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        (reinterpret_cast<float4*>(c + idx))[0] = reg_c;
    }
}

// 我又想到为什么不用指针？
__global__ void elementwise_add_f32x4_ptr_kernel(float *a, float *b, float *c, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        float4* reg_a = reinterpret_cast<float4*>(a + idx);
        float4* reg_b = reinterpret_cast<float4*>(b + idx);
        float4* reg_c = reinterpret_cast<float4*>(c + idx);
        reg_c->x = reg_a->x + reg_b->x;
        reg_c->y = reg_a->y + reg_b->y;
        reg_c->z = reg_a->z + reg_b->z;
        reg_c->w = reg_a->w + reg_b->w;
    }
}



#include <cuda_fp16.h>
// fp16版本
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

//fp16x2 向量化版本
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        half2 reg_a = (reinterpret_cast<half2*>(a + idx))[0];
        half2 reg_b = (reinterpret_cast<half2*>(b + idx))[0];
        half2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        (reinterpret_cast<half2*>(c + idx))[0] = reg_c;
    }
}

//fp16x8 向量化版本
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);

    half2 reg_a0 = (reinterpret_cast<half2*>(a + idx + 0))[0];
    half2 reg_a1 = (reinterpret_cast<half2*>(a + idx + 2))[0];
    half2 reg_a2 = (reinterpret_cast<half2*>(a + idx + 4))[0];
    half2 reg_a3 = (reinterpret_cast<half2*>(a + idx + 6))[0];

    half2 reg_b0 = (reinterpret_cast<half2*>(b + idx + 0))[0];
    half2 reg_b1 = (reinterpret_cast<half2*>(b + idx + 2))[0];
    half2 reg_b2 = (reinterpret_cast<half2*>(b + idx + 4))[0];
    half2 reg_b3 = (reinterpret_cast<half2*>(b + idx + 6))[0];

    half2 reg_c0;
    half2 reg_c1;
    half2 reg_c2;
    half2 reg_c3;
    reg_c0.x = __hadd(reg_a0.x, reg_b0.x);
    reg_c0.y = __hadd(reg_a0.y, reg_b0.y);

    reg_c1.x = __hadd(reg_a1.x, reg_b1.x);
    reg_c1.y = __hadd(reg_a1.y, reg_b1.y);

    reg_c2.x = __hadd(reg_a2.x, reg_b2.x);
    reg_c2.y = __hadd(reg_a2.y, reg_b2.y);

    reg_c3.x = __hadd(reg_a3.x, reg_b3.x);
    reg_c3.y = __hadd(reg_a3.y, reg_b3.y);

    if ((idx + 0) < N) {
        (reinterpret_cast<half2*>(c + idx + 0))[0] = reg_c0;
    }
    if ((idx + 2) < N) {
        (reinterpret_cast<half2*>(c + idx + 2))[0] = reg_c1;
    }
    if ((idx + 4) < N) {
        (reinterpret_cast<half2*>(c + idx + 4))[0] = reg_c2;
    }
    if ((idx + 6) < N) {
        (reinterpret_cast<half2*>(c + idx + 6))[0] = reg_c3;
    }
}

//fp16向量化版本,pack

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c,int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    // temporary register(memory), .local space in ptx, addressable
    half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
    // reinterpret as float4 and load 128 bits in 1 memory issue.
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
    // __hadd2 for half2 x 4
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    // reinterpret as float4 and store 128 bits in 1 memory issue.
    if ((idx + 7) < N) {
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
}

#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N){
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        half pack_a[2],pack_b[2],pack_c[2];
        LDST32BITS(pack_a[0]) = LDST32BITS(a[idx]); // load 32 bits
        LDST32BITS(pack_b[0]) = LDST32BITS(b[idx]); // load 32 bits

        HALF2(pack_c[0]) = __hadd2(HALF2(pack_a[0]), HALF2(pack_b[0]));        
        // reinterpret as float4 and store 128 bits in 1 memory issue.
        if ((idx + 7) < N) {
            LDST32BITS(c[idx]) = LDST32BITS(pack_c[0]);
        }
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

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f32x4_ptr, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x2_pack, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4_ptr)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2_pack)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
