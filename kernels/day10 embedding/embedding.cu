
#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include "../common.h"


__global__ void embedding_f32_my_kernel(const int *indices, float *embedding_table,
                                     float *output, int num_indices, int emb_size) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_indices){
        int emb_idx = indices[idx];
        float* emb_ptr = embedding_table + emb_idx * emb_size;
        for(int i = 0; i < emb_size; i++){
            output[idx * emb_size + i] = emb_ptr[i];
        }
    }
}


// dim3 block(emb_size / 1);
__global__ void embedding_f32_kernel(const int *indices, float *embedding_table, float *output, int num_indices, int emb_size) {
    int offset = indices[blockIdx.x] * emb_size;

    output[blockIdx.x * emb_size + threadIdx.x] = embedding_table[offset + threadIdx.x];
}


__global__ void embedding_f32x4_kernel(const int *indices, float *embedding_table, float *output, int num_indices, int emb_size) {
    int tid =  threadIdx.x * 4;

    int offset = indices[blockIdx.x] * emb_size;

    output[blockIdx.x * emb_size + tid] = embedding_table[offset + tid];
    output[blockIdx.x * emb_size + tid + 1] = embedding_table[offset + tid + 1];
    output[blockIdx.x * emb_size + tid + 2] = embedding_table[offset + tid + 2];
    output[blockIdx.x * emb_size + tid + 3] = embedding_table[offset + tid + 3];
}

__global__ void embedding_f32x4_pack_kernel(const int *indices, float *embedding_table, float *output, int num_indices, int emb_size) {
    int tid =  threadIdx.x * 4;
    int offset = indices[blockIdx.x] * emb_size;

    LDST128BITS(output[blockIdx.x * emb_size + tid]) = LDST128BITS(embedding_table[offset + tid]);
}

__global__ void embedding_f16_kernel(const int *indices, half *embedding_table, half *output, int num_indices, int emb_size) {
    int offset = indices[blockIdx.x] * emb_size;

    output[blockIdx.x * emb_size + threadIdx.x] = embedding_table[offset + threadIdx.x];
}

__global__ void embedding_f16x8_kernel(const int *indices, half *embedding_table, half *output, int num_indices, int emb_size) {
    int tid =  threadIdx.x * 8;

    int offset = indices[blockIdx.x] * emb_size;

    output[blockIdx.x * emb_size + tid] = embedding_table[offset + tid];
    output[blockIdx.x * emb_size + tid + 1] = embedding_table[offset + tid + 1];
    output[blockIdx.x * emb_size + tid + 2] = embedding_table[offset + tid + 2];
    output[blockIdx.x * emb_size + tid + 3] = embedding_table[offset + tid + 3];
    output[blockIdx.x * emb_size + tid + 4] = embedding_table[offset + tid + 4];
    output[blockIdx.x * emb_size + tid + 5] = embedding_table[offset + tid + 5];
    output[blockIdx.x * emb_size + tid + 6] = embedding_table[offset + tid + 6];
    output[blockIdx.x * emb_size + tid + 7] = embedding_table[offset + tid + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int *indices, half *embedding_table, half *output, int num_indices, int emb_size) {
    int tid =  threadIdx.x * 8;
    int offset = indices[blockIdx.x] * emb_size;

    LDST128BITS(output[blockIdx.x * emb_size + tid]) = LDST128BITS(embedding_table[offset + tid]);
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
                               torch::Tensor o) {                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
    CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
    CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
                                                                               \
    const int N = a.size(0);                                                   \
    const int emb_size = weight.size(1);                                       \
    dim3 block(emb_size / n_elements);                                         \
    dim3 grid(N);                                                              \
    embedding_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<int *>(a.data_ptr()),                                 \
        reinterpret_cast<element_type *>(weight.data_ptr()),                   \
        reinterpret_cast<element_type *>(o.data_ptr()), N, emb_size);          \
  }

TORCH_BINDING_EMBEDDING(f32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f32x4_pack, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f16, torch::kHalf, half, 1)
TORCH_BINDING_EMBEDDING(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_EMBEDDING(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4_pack);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8_pack);
}
