#include <algorithm>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include "../common.h"
#define WARP_SIZE 256
#define WARP_SIZE_S 16

__global__ void mat_transpose_f32_col2row_kernel(float* x, float* y, int row, int col){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int yi_row = idx / row;
    int yi_col = idx % row;
    if(idx < row * col){
        y[idx] = x[yi_col * col + yi_row];
    }
}


__global__ void mat_transpose_f32_row2col_kernel(float* x, float* y, int row, int col){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int xi_row = idx / col;
    int xi_col = idx % col;
    if(idx < row * col){
        y[xi_col * row + xi_row] = x[idx];
    }
}
//////////////////////////////////////////////////////////////

//这个kernel 需要调用方 grid 的维度是 ((M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row)) x ((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col))

// __global__ void mat_transpose_f32_col2row2d_kernel(float* x, float* y, int row, int col){
//     int ix = blockDim.x * blockIdx.x + threadIdx.x;
//     int iy = blockDim.y * blockIdx.y + threadIdx.y;
//     if(ix < row && iy < col){0
//         y[iy * row + ix] = x[ix * col + iy];
//     }
// }

__global__ void mat_transpose_f32_col2row2d_kernel(float* x, float* y, int row, int col){
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if(ix < col && iy < row){
        y[ix * row + iy] = x[iy * col + ix];
    }
}

__global__ void mat_transpose_f32_row2col2d_kernel(float* x, float* y, int row, int col){
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if(ix < col && iy < row){
        y[ix * row + iy] = x[iy * col + ix];
    }
}

///////////////////////////////////////////////////////////////

__global__ void mat_transpose_f32x4_col2row_kernel(float* x, float* y, int row, int col){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    int y1_row = idx / row;
    int y1_col = idx % row;

    if(idx + 3 < row * col){
        float4 reg_x;

        reg_x.x = x[      y1_col * col + y1_row];
        reg_x.y = x[(y1_col + 1) * col + y1_row];
        reg_x.z = x[(y1_col + 2) * col + y1_row];
        reg_x.w = x[(y1_col + 3) * col + y1_row];

        reinterpret_cast<float4*>(y + idx)[0] = reg_x;
    }
}

__global__ void mat_transpose_f32x4_row2col_kernel(float* x, float* y, int row, int col){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    int xi_row = idx / col;
    int xi_col = idx % col;
    if(idx + 3 < row * col){
        float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];

        y[xi_col * row + xi_row]       = reg_x.x;
        y[(xi_col + 1) * row + xi_row] = reg_x.y;
        y[(xi_col + 2) * row + xi_row] = reg_x.z;
        y[(xi_col + 3) * row + xi_row] = reg_x.w;
    }
}
//////////////////////////////////////////////////////////////////

__global__ void mat_transpose_f32_diagonal_kernel(float* x, float* y, int row, int col){
    // int ny = blockDim.y * blockIdx.y + threadIdx.y;
    // int nx = blockDim.x * blockIdx.x + threadIdx.x;

    // if(ny < row && nx < col){
    //     y[ny * col + nx] = x[nx * row + ny];
    // }
}


__global__ void mat_transpose_f32_shared_col2row2d_kernel(float* x, float* y, int row, int col){
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ float s_x[WARP_SIZE_S][WARP_SIZE_S];

    if(ix < col && iy < row){
        s_x[threadIdx.y][threadIdx.x] =  x[iy * col + ix];
        __syncthreads();

        y[ix * row + iy] = s_x[threadIdx.y][threadIdx.x];
    }
}


__global__ void mat_transpose_f32_shared_row2col2d_kernel(float* x, float* y, int row, int col){
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ float s_x[WARP_SIZE_S][WARP_SIZE_S];

    if(ix < col && iy < row){
        s_x[threadIdx.y][threadIdx.x] = x[iy * col + ix];
        __syncthreads();

        y[ix * row + iy]  = s_x[threadIdx.y][threadIdx.x];
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

#define TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)        \
  void mat_transpose_##tag(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(WARP_SIZE);                                                     \
    dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));                 \
    mat_transpose_##tag##_kernel<<<grid, block>>>(                             \
        reinterpret_cast<element_type *>(x.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), M, N);                 \
  }

#define TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type,              \
                                      n_element_row, n_element_col)            \
  void mat_transpose_##tag##2d(torch::Tensor x, torch::Tensor y) {             \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int M = x.size(0);                                                   \
    const int N = x.size(1);                                                   \
    dim3 block(WARP_SIZE_S, WARP_SIZE_S);                                      \
    dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col),           \
              (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));          \
    mat_transpose_##tag##2d_kernel <<< grid,                                 \
        block >>> (reinterpret_cast<element_type *>(x.data_ptr()),             \
                   reinterpret_cast<element_type *>(y.data_ptr()), M, N);      \
  }

// 1d index
TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32_row2col, torch::kFloat32, float, 1)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_col2row, torch::kFloat32, float, 4)
TORCH_BINDING_MAT_TRANSPOSE(f32x4_row2col, torch::kFloat32, float, 4)
// 2d index. easier for diagonal
TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1,1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1,1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_shared_col2row, torch::kFloat32, float, 1, 1)
TORCH_BINDING_MAT_TRANSPOSE2D(f32_shared_row2col, torch::kFloat32, float, 1, 1)

// diagonal index method.


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 1d index
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_col2row)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32x4_row2col)
  // 2d index. easier for diagonal
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_row2col2d)

  // shared memory optimize
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_shared_col2row2d)
  TORCH_BINDING_COMMON_EXTENSION(mat_transpose_f32_shared_row2col2d)


}


