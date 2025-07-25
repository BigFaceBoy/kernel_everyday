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

#define BASE 10000
#define BLOCK_SIZE 256
__global__ void rope_f32_kernel(float* x, float* y, int seq_len,  int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float q1 = x[2 * idx];
    float q2 = x[2 * idx + 1];

    int token_pos = idx / N;
    int token_idx = idx % N;

    float theta = powf(BASE, -2 * (token_idx-1) / N);
    float cos_v = cosf(token_pos * theta);
    float sin_v = sinf(token_pos * theta);

    y[2 * idx] = q1 * cos_v - q2 * sin_v;
    y[2 * idx + 1] = q1 * sin_v + q2 * cos_v;
}

__global__ void rope_f32x4_pack_kernel(float* x, float* y, int seq_len, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 3 < N){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];
        int token_pos = idx / N;
        int token_idx = idx % N;
        float theta1 = powf(BASE, -2 * (token_idx - 1) / N);
        float theta2 = powf(BASE, -2 * (token_idx + 1 -1) / N);
        
        float cos_v1 = cosf(token_pos * theta1);
        float sin_v1 = sinf(token_pos * theta1);

        float cos_v2 = cosf(token_pos * theta2);
        float sin_v2 = sinf(token_pos * theta2);

        float4 reg_y;
        reg_y.x = reg_x.x * cos_v1 - reg_x.y * sin_v1;
        reg_y.y = reg_x.x * sin_v1 + reg_x.y * cos_v1;

        reg_y.z = reg_x.z * cos_v2 - reg_x.w * sin_v2;
        reg_y.w = reg_x.z * sin_v2 + reg_x.w * cos_v2;

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

void rope_f32(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size / 2);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                   seq_len, N);
}


void rope_f32x4_pack(torch::Tensor x, torch::Tensor out) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
  int seq_len = x.size(0);
  int hidden_size = x.size(1);
  int N = (int)(hidden_size / 4);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);
  rope_f32x4_pack_kernel<<<grid, block>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(), seq_len, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rope_f32)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32x4_pack)
}