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
               out_f32f32: 1622.18884277  , time:0.02352047ms
             out_f32x4f32: 1622.18872070  , time:0.02175117ms
             out_f32x1f32: 1622.18908691  , time:0.02810860ms
            out_f32f32_th: 1622.18859863  , time:0.01894164ms
--------------------------------------------------------------------------------
               out_f16f16: 1622.06250000  , time:0.02377939ms
               out_f16f32: 1622.32250977  , time:0.02316666ms
             out_f16x2f32: 1622.41235352  , time:0.02144313ms
             out_f16x2f16: 1622.31909180  , time:0.02145529ms
         out_f16x8packf16: 1622.14501953  , time:0.02040458ms
         out_f16x8packf32: 1622.32214355  , time:0.02042961ms
            out_f16f16_th: 1622.00000000  , time:0.01775622ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=1024, K=2048
               out_f32f32: 2654.11376953  , time:0.02831626ms
             out_f32x4f32: 2654.11547852  , time:0.02119422ms
             out_f32x1f32: 2654.11450195  , time:0.03675127ms
            out_f32f32_th: 2654.11279297  , time:0.02017212ms
--------------------------------------------------------------------------------
               out_f16f16: 2653.75292969  , time:0.02809143ms
               out_f16f32: 2654.01000977  , time:0.02736878ms
             out_f16x2f32: 2653.83862305  , time:0.02340651ms
             out_f16x2f16: 2654.27001953  , time:0.02427077ms
         out_f16x8packf16: 2655.09228516  , time:0.02142334ms
         out_f16x8packf32: 2654.01220703  , time:0.02053761ms
            out_f16f16_th: 2654.00000000  , time:0.01827097ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=1024, K=4096
               out_f32f32: -1016.95642090 , time:0.03604388ms
             out_f32x4f32: -1016.95617676 , time:0.02626753ms
             out_f32x1f32: -1016.95422363 , time:0.05178690ms
            out_f32f32_th: -1016.95642090 , time:0.02149892ms
--------------------------------------------------------------------------------
               out_f16f16: -1016.98828125 , time:0.03583813ms
               out_f16f32: -1016.90466309 , time:0.03650331ms
             out_f16x2f32: -1017.17071533 , time:0.02616405ms
             out_f16x2f16: -1018.54443359 , time:0.02558661ms
         out_f16x8packf16: -1015.61523438 , time:0.02118969ms
         out_f16x8packf32: -1016.90484619 , time:0.02176833ms
            out_f16f16_th: -1017.00000000 , time:0.02022934ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=2048, K=1024
               out_f32f32: -2538.28881836 , time:0.02847338ms
             out_f32x4f32: -2538.29003906 , time:0.02283549ms
             out_f32x1f32: -2538.29052734 , time:0.03626251ms
            out_f32f32_th: -2538.29077148 , time:0.01901746ms
--------------------------------------------------------------------------------
               out_f16f16: -2538.58398438 , time:0.02740693ms
               out_f16f32: -2538.30126953 , time:0.02832890ms
             out_f16x2f32: -2538.57934570 , time:0.02368498ms
             out_f16x2f16: -2538.89135742 , time:0.02245760ms
         out_f16x8packf16: -2537.50781250 , time:0.02209902ms
         out_f16x8packf32: -2538.30200195 , time:0.02394938ms
            out_f16f16_th: -2538.00000000 , time:0.02069783ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=2048, K=2048
               out_f32f32: 1001.14300537  , time:0.03610706ms
             out_f32x4f32: 1001.14190674  , time:0.02387071ms
             out_f32x1f32: 1001.14398193  , time:0.05213118ms
            out_f32f32_th: 1001.14208984  , time:0.02293611ms
--------------------------------------------------------------------------------
               out_f16f16: 1001.28637695  , time:0.04191136ms
               out_f16f32: 1000.80615234  , time:0.04369569ms
             out_f16x2f32: 1001.46301270  , time:0.02833986ms
             out_f16x2f16: 1000.84704590  , time:0.02974534ms
         out_f16x8packf16: 999.92333984   , time:0.02197075ms
         out_f16x8packf32: 1000.80688477  , time:0.02204370ms
            out_f16f16_th: 1001.00000000  , time:0.02021861ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=2048, K=4096
               out_f32f32: 922.26263428   , time:0.05232525ms
             out_f32x4f32: 922.26776123   , time:0.02982879ms
             out_f32x1f32: 922.26348877   , time:0.08408046ms
            out_f32f32_th: 922.26550293   , time:0.02390933ms
--------------------------------------------------------------------------------
               out_f16f16: 922.59619141   , time:0.05257082ms
               out_f16f32: 922.28894043   , time:0.05258346ms
             out_f16x2f32: 922.88012695   , time:0.03251362ms
             out_f16x2f16: 923.22045898   , time:0.03296566ms
         out_f16x8packf16: 923.64160156   , time:0.02303600ms
         out_f16x8packf32: 922.28997803   , time:0.02285099ms
            out_f16f16_th: 922.50000000   , time:0.02119231ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=4096, K=1024
               out_f32f32: 353.59225464   , time:0.03558731ms
             out_f32x4f32: 353.59426880   , time:0.02538800ms
             out_f32x1f32: 353.59429932   , time:0.05184770ms
            out_f32f32_th: 353.59350586   , time:0.02085924ms
--------------------------------------------------------------------------------
               out_f16f16: 354.50793457   , time:0.03601122ms
               out_f16f32: 354.29022217   , time:0.03620005ms
             out_f16x2f32: 354.62377930   , time:0.02551746ms
             out_f16x2f16: 355.22436523   , time:0.02555418ms
         out_f16x8packf16: 354.41406250   , time:0.02737236ms
         out_f16x8packf32: 354.28854370   , time:0.02615929ms
            out_f16f16_th: 354.25000000   , time:0.01942444ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=4096, K=2048
               out_f32f32: -438.36926270  , time:0.05236554ms
             out_f32x4f32: -438.36831665  , time:0.02723885ms
             out_f32x1f32: -438.36834717  , time:0.08537865ms
            out_f32f32_th: -438.36633301  , time:0.02545786ms
--------------------------------------------------------------------------------
               out_f16f16: -440.68676758  , time:0.05244541ms
               out_f16f32: -439.05776978  , time:0.05210233ms
             out_f16x2f32: -439.02236938  , time:0.03659534ms
             out_f16x2f16: -437.96826172  , time:0.03658748ms
         out_f16x8packf16: -440.27880859  , time:0.02588820ms
         out_f16x8packf32: -439.05575562  , time:0.02563500ms
            out_f16f16_th: -439.00000000  , time:0.02147698ms
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                        S=4096, K=4096
               out_f32f32: 1849.91979980  , time:0.08502650ms
             out_f32x4f32: 1849.90747070  , time:0.03985000ms
             out_f32x1f32: 1849.91430664  , time:0.14872646ms
            out_f32f32_th: 1849.91040039  , time:0.03025508ms
--------------------------------------------------------------------------------
               out_f16f16: 1850.81494141  , time:0.08579016ms
               out_f16f32: 1850.61877441  , time:0.08506775ms
             out_f16x2f32: 1850.47998047  , time:0.04550099ms
             out_f16x2f16: 1849.80664062  , time:0.04464459ms
         out_f16x8packf16: 1849.84814453  , time:0.02655435ms
         out_f16x8packf32: 1850.62487793  , time:0.02671409ms
            out_f16f16_th: 1851.00000000  , time:0.02449846ms
--------------------------------------------------------------------------------
