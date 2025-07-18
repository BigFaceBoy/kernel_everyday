# 一、Reduce kernel 总览
- warp_reduce_fp32/fp16/bf16_kernel
- block_reduce_fp32_kernel
- block_all_reduce_sum_f32_f32_kernel
- block_all_reduce_sum_f32x4_f32_kernel(float4向量化版本)
- block_all_reduce_sum_f16_f16_kernel(fp16版本，使用fp16 acc)
- block_all_reduce_sum_f16_f32_kernel(fp16版本，使用fp32 acc)
- block_all_reduce_sum_f16x2_f16_kernel(fp16向量化版本，使用fp16 acc)
- block_all_reduce_sum_f16x2_f32_kernel(fp16向量化版本，使用fp32 acc)
- block_all_reduce_sum_f16x8_pack_f16_kernel(fp16向量化版本，使用fp16 acc, pack)
- block_all_reduce_sum_f16x8_pack_f32_kernel(fp16向量化版本，使用fp32 acc, pack)
- block_all_reduce_sum_bf16_bf16_kernel(bf16版本，使用bf16 acc)
- block_all_reduce_sum_bf16_f32_kernel(bf16版本，使用fp32 acc)
- block_all_reduce_sum_bf16x8_pack_bf16_kernel(bf16版本，使用bf16 acc, pack)
- block_all_reduce_sum_bf16x8_pack_f32_kernel(bf16版本，使用fp32 acc, pack)
- block_all_reduce_sum_bf16x2_bf16_kernel(bf16向量化版本，使用bf16 acc)
- block_all_reduce_sum_bf16x2_f32_kernel(bf16向量化版本，使用fp32 acc)
- block_all_reduce_sum_fp8_e4m3_f16_kernel(fp8_e4m3版本，使用fp16 acc)
- block_all_reduce_sum_fp8_e5m2_f16_kernel(fp8_e5m2版本，使用fp16 acc)
- block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(fp8_e4m3版本，使用fp16 acc, pack)
- block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(fp8_e5m2版本，使用fp16 acc, pack)
- block_all_reduce_sum_i8_i32_kernel(i8版本，使用i32 acc)
- block_all_reduce_sum_i8x16_pack_i32_kernel(i8版本，使用i32 acc, pack)

建议先食用 [GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)
# 二、cpu实现
```C++
float reduce_sum(float* x, int N){
    float sum = 0.0;
    for(int i = 0; i < N; i++){
        sum += x[i];
    }
    return sum;
}
```
# 三、CUDA实现
那这个在 CUDA 中如何实现？不可能在CUDA中写for循环累加。考虑到cuda中每个线程都在完成相似的操作，那么 CUDA 中 reduce 的实现如下图所示：

![](../../.assets/reduce_cuda.png)

## 3.1 reduce_fp32_kernel
```C++
__global__ void reduce_fp32_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    float* b_x = x + blockDim.x * blockIdx.x;
    for(int offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(tid < offset){
            b_x[tid] += b_x[tid + offset]
        }
        __syncthreads();
    }
    if(tid == 0){
        y[blockIdx.x] = b_x[0];
    }
}
```
## 3.2 reduce_fp32_shared_kernel
每个线程都是操作global memory(在x中加载数据和写入数据), 那么如果每个block将对应的数据加载到shared memory，就能够将提升性能。
```C++
__global__ void reduce_fp32_shared_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_x[blockDim.x];
    s_x[tid] = idx < N ? x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset]
        }
        __syncthreads();
    }
    if(tid == 0){
        y[blockIdx.x] = s_x[0];
    }
}
```
这两种方式都是最后将y拷贝到cpu内存，然后做一次累加，得到最终值。那是否有方法在GPU上计算出最终结果？
## 3.3 使用原子函数
```C++
__global__ void reduce_fp32_shared_atomic_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_x[blockDim.x];
    s_x[tid] = idx < N ? x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(y, s_x[0]);
    }
}
```

## 3.4 使用warp
线程束(warp) 是 SM 中基本的执行单元。一个线程束由32个连续线程组成
### 3.4.1 __syncwarp
当所涉及的线程都在一个线程束内时，可以将线程块同步函数 __syncthreads 换成一个更加廉价的线程束同步函数 __syncwarp。
```C++
__global__ void reduce_fp32_shared_atomic_warp_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_x[blockDim.x];
    s_x[tid] = idx < N ? x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset >= 32; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }
    for(int offset = 16; offset > 0; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncwarp();
    }
    if(tid == 0){
        atomicAdd(y, s_x[0]);
    }
}
```
### 3.4.2 线程束洗牌函数
线程束洗牌函数（Warp Shuffle Functions）是一组高效的线程束内数据交换指令，允许同一线程束（Warp）中的线程直接读取其他线程的寄存器值，无需通过共享内存，从而显著提升并行计算的效率‌。需指定线程mask以控制参与线程。

- `T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)`  // 从指定线程（srcLane）广播变量值到当前线程束的所有线程，实现数据复制‌
- `T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` //从当前线程ID减去delta的线程中获取数据，数据上移
- `T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize)` //当前线程ID加上delta的线程中获取数据，数据下移
- `T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize)` // source lane id = 当前线程ID xor lanemask, 从 source lane id 拷贝数据 

```C++
__global__ void reduce_fp32_shared_atomic_warp_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_x[blockDim.x];
    s_x[tid] = idx < N ? x[idx] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset >= 32; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }

    float x_val = s_x[tid]; 
    for(int offset = 16; offset > 0; offset >>=1){
        if(tid < offset){
            x_val += __shfl_down_sync(0xffffffff, x_val, offset);
        }
        __syncwarp();
    }
    if(tid == 0){
        atomicAdd(y, x_val);
    }
}
```

## 3.5 提高线程利用率
在前边的例子中， 都使用大小为 128 的线程块，所以当 offset 等于 64 时，只用了 1/2 的线程进行计算，其余线程闲置。当 offset 等于 32 时，只用了 1/4 的线程进行计算，其余线程闲置。最终，当 offset 等于 1 时，只用了 1/128 的线程进行计算，其余线程闲置。归约过程一共用了 `log2 128 = 7` 步， 故归约过程中线程的平均利用率只有 `(1/2 + 1/4 + ...)/7 ≈ 1/7 `。


```C++
__global__ void reduce_fp32_shared_atomic_warp_kernel(float* x, float* y, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_x[blockDim.x];
    float x_stride_sum = 0.0;
    int stride = blockDim.x * gridDim.x;

    for(int n = idx; n < N>; n += stride){
        x_stride_sum += x[n];
    }
    s_x[tid] = x_stride_sum;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset >= 32; offset >>=1){
        if(tid < offset){
            s_x[tid] += s_x[tid + offset];
        }
        __syncthreads();
    }

    float x_val = s_x[tid];
    for(int offset = 16; offset > 0; offset >>=1){
        if(tid < offset){
            x_val += __shfl_down_sync(0xffffffff, x_val, offset);
        }
        __syncwarp();
    }
    if(tid == 0){
        y[blockIdx.x] = x_val;
    }
}
```

有了上述的优化历程，再开始本例中的kernel。问题是：既然[GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980) 中已经优化到这个地步了，那 [LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main)中的实现是有其他优化点还是实现方式的不同？  
[LeetCUDA](https://github.com/xlite-dev/LeetCUDA/tree/main) 中的实现如下：
```C++
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enough for warp operaion.
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0)
    atomicAdd(y, sum);
}
```
对比了一下 [GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980) 中的最终实现版本，差异在：  
1、`优化指南`中是将每个block内的数据折半相加，直至到32个数据时使用线程束洗牌函数归约， 而`LeetCUDA` 中是先将block中的每个warp使用线程束洗牌函数归约，结果存放在shared memory中，最后通过线程束洗牌函数归约得到最终值。
2、线程束洗牌函数不同， `优化指南`中使用的是`__shfl_down_sync` , `LeetCUDA`中使用的是`__shfl_xor_sync`,当然在本kernel中两者对最后结果的等价的。
3、有向量化(fp32x4 等kernel)


# benchmark
--------------------------------------------------------------------------------
                                        S=1024, K=1024
               out_f32f32: 403.78421021   , time:0.02674603ms
             out_f32x4f32: 403.78402710   , time:0.02051067ms
             out_f32x1f32: 403.78430176   , time:0.02745390ms
            out_f32f32_th: 403.78451538   , time:0.01807666ms
--------------------------------------------------------------------------------
               out_f16f16: 403.16845703   , time:0.02321887ms
               out_f16f32: 403.76617432   , time:0.02320075ms
             out_f16x2f32: 404.09667969   , time:0.02067375ms
             out_f16x2f16: 404.09960938   , time:0.02042890ms
         out_f16x8packf16: 403.93554688   , time:0.02032518ms
         out_f16x8packf32: nan            , time:0.01997018ms
            out_f16f16_th: 403.75000000   , time:0.01757550ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=1024, K=2048
               out_f32f32: -1409.65051270 , time:0.02715993ms
             out_f32x4f32: -1409.65112305 , time:0.02273393ms
             out_f32x1f32: -1409.64953613 , time:0.03524160ms
            out_f32f32_th: -1409.64941406 , time:0.01886106ms
--------------------------------------------------------------------------------
               out_f16f16: -1409.23046875 , time:0.02762413ms
               out_f16f32: -1409.43078613 , time:0.02892160ms
             out_f16x2f32: -1409.45495605 , time:0.02327156ms
             out_f16x2f16: -1408.71728516 , time:0.02314711ms
         out_f16x8packf16: -1411.34521484 , time:0.02039695ms
         out_f16x8packf32: nan            , time:0.02007794ms
            out_f16f16_th: -1409.00000000 , time:0.01808333ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=1024, K=4096
               out_f32f32: 808.68267822   , time:0.03665614ms
             out_f32x4f32: 808.67974854   , time:0.02438688ms
             out_f32x1f32: 808.68182373   , time:0.05115914ms
            out_f32f32_th: 808.68218994   , time:0.02080822ms
--------------------------------------------------------------------------------
               out_f16f16: 808.74218750   , time:0.03572226ms
               out_f16f32: 808.66247559   , time:0.03568268ms
             out_f16x2f32: 808.96777344   , time:0.02543139ms
             out_f16x2f16: 809.03356934   , time:0.02559137ms
         out_f16x8packf16: 808.19238281   , time:0.02105045ms
         out_f16x8packf32: nan            , time:0.02027702ms
            out_f16f16_th: 808.50000000   , time:0.01954341ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=2048, K=1024
               out_f32f32: -106.36501312  , time:0.02740073ms
             out_f32x4f32: -106.36377716  , time:0.02209520ms
             out_f32x1f32: -106.36486053  , time:0.03538084ms
            out_f32f32_th: -106.36396790  , time:0.01890349ms
--------------------------------------------------------------------------------
               out_f16f16: -106.02441406  , time:0.02750850ms
               out_f16f32: -106.12662506  , time:0.02723241ms
             out_f16x2f32: -105.90848541  , time:0.02220464ms
             out_f16x2f16: -105.99291992  , time:0.02208066ms
         out_f16x8packf16: -105.49755859  , time:0.02214789ms
         out_f16x8packf32: nan            , time:0.02151418ms
            out_f16f16_th: -106.12500000  , time:0.01807237ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=2048, K=2048
               out_f32f32: -948.21441650  , time:0.03551388ms
             out_f32x4f32: -948.21386719  , time:0.02272415ms
             out_f32x1f32: -948.21484375  , time:0.05091476ms
            out_f32f32_th: -948.21441650  , time:0.02094674ms
--------------------------------------------------------------------------------
               out_f16f16: -947.12207031  , time:0.03565693ms
               out_f16f32: -948.01202393  , time:0.03547144ms
             out_f16x2f32: -948.46746826  , time:0.02757668ms
             out_f16x2f16: -948.88940430  , time:0.02922010ms
         out_f16x8packf16: -948.23925781  , time:0.02244878ms
         out_f16x8packf32: nan            , time:0.02315736ms
            out_f16f16_th: -948.00000000  , time:0.02022290ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=2048, K=4096
               out_f32f32: -308.35345459  , time:0.05188918ms
             out_f32x4f32: -308.34963989  , time:0.02943039ms
             out_f32x1f32: -308.34988403  , time:0.08285236ms
            out_f32f32_th: -308.34863281  , time:0.02401638ms
--------------------------------------------------------------------------------
               out_f16f16: -305.36987305  , time:0.05304980ms
               out_f16f32: -307.38061523  , time:0.05280757ms
             out_f16x2f32: -306.92733765  , time:0.03219366ms
             out_f16x2f16: -308.18774414  , time:0.03193617ms
         out_f16x8packf16: -308.12158203  , time:0.02302074ms
         out_f16x8packf32: nan            , time:0.02172160ms
            out_f16f16_th: -307.50000000  , time:0.02097893ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=4096, K=1024
               out_f32f32: -475.87634277  , time:0.03563356ms
             out_f32x4f32: -475.87622070  , time:0.02684474ms
             out_f32x1f32: -475.87698364  , time:0.05155921ms
            out_f32f32_th: -475.87628174  , time:0.02279806ms
--------------------------------------------------------------------------------
               out_f16f16: -478.54150391  , time:0.03573251ms
               out_f16f32: -477.21914673  , time:0.03572917ms
             out_f16x2f32: -477.56417847  , time:0.02555990ms
             out_f16x2f16: -476.08032227  , time:0.02552319ms
         out_f16x8packf16: -477.94140625  , time:0.02516150ms
         out_f16x8packf32: nan            , time:0.02470064ms
            out_f16f16_th: -477.25000000  , time:0.02062368ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=4096, K=2048
               out_f32f32: -5693.60302734 , time:0.05206823ms
             out_f32x4f32: -5693.60595703 , time:0.02623868ms
             out_f32x1f32: -5693.59912109 , time:0.08280158ms
            out_f32f32_th: -5693.61132812 , time:0.02377677ms
--------------------------------------------------------------------------------
               out_f16f16: -5694.51562500 , time:0.05289817ms
               out_f16f32: -5693.15527344 , time:0.05483222ms
             out_f16x2f32: -5692.87792969 , time:0.03617525ms
             out_f16x2f16: -5694.16015625 , time:0.03575277ms
         out_f16x8packf16: -5689.65039062 , time:0.02533484ms
         out_f16x8packf32: nan            , time:0.02474189ms
            out_f16f16_th: -5692.00000000 , time:0.02113128ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=4096, K=4096
               out_f32f32: 3660.71655273  , time:0.08469939ms
             out_f32x4f32: 3660.71313477  , time:0.03958273ms
             out_f32x1f32: 3660.71679688  , time:0.14618325ms
            out_f32f32_th: 3660.71289062  , time:0.03160763ms
--------------------------------------------------------------------------------
               out_f16f16: 3659.74682617  , time:0.08665848ms
               out_f16f32: 3662.92968750  , time:0.08489537ms
             out_f16x2f32: 3661.68212891  , time:0.04416823ms
             out_f16x2f16: 3662.26855469  , time:0.04490137ms
         out_f16x8packf16: 3657.91259766  , time:0.02707434ms
         out_f16x8packf32: nan            , time:0.02482104ms
            out_f16f16_th: 3662.00000000  , time:0.02419090ms
--------------------------------------------------------------------------------

