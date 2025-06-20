直方图统计：计算每个数据出现的次数  

# histogram_i32_kernel
给出kernel定义：`__global__ void  histogram_i32_kernel(int *a, int *y,  int N)`
a 为需要统计的数据，y为存储的频率。那么很容易写出：
```
__global__ void  histogram_i32_kernel(int *a, int *y,  int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        int v = a[idx];
        y[v] += 1;
    }
}
```
考虑到GPU是并发执行多个线程，多个v出现时这种写法显然不行，那么就要用到CUDA提供的原子操作 atomicAdd(address, val).
```
__global__ void  histogram_i32_kernel(int *a, int *y,  int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        atomicAdd( &(y[a[idx]]), 1);
    }
}
```

# histogram_i32x4_kernel
有了 `elementwise`中对向量化的练习，这个kernel就很容易实现了。
```
__global__ void  histogram_i32x4_kernel(int *a, int *y,  int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        int4 reg_a = (reinterpret_cast<int4*>(a + idx))[0];
        atomicAdd( &(y[reg_a.x]), 1);
        atomicAdd( &(y[reg_a.y]), 1);
        atomicAdd( &(y[reg_a.z]), 1);
        atomicAdd( &(y[reg_a.w]), 1);
    }
}
```