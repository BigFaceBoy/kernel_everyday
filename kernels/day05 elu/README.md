# elu 计算公式
$$f(x)=\begin{cases}
x &\text{if x >=  0} \\
\alpha(e^x - 1) &\text{if x < 0} \\
\end{cases}$$  
其中，$\alpha$是一个超参数，通常取值为1。


# f16_kernel
一开始我的实现是：
```C++
#define alpha 1.0f
const half half_1 = __float2half(1.0f);

__global__ void elu_f16_kernel(half *x, half *y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        y[idx] = x[idx] >= 0 ? x[idx] : alpha * (hexp(x[idx]) - half_1);
    }
}

__global__ void elu_f16x2_kernel(half *x, half *y, int N){
    int idx =  2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx + 1 < N){
        half2 reg_x = reinterpret_cast<half2*>(x)[idx];
        half2 reg_y;

        reg_y.x = reg_x.x >= 0 ? reg_x.x : alpha *  (hexp(reg_x.x) - half_1);
        reg_y.y = reg_x.y >= 0 ? reg_x.y : alpha *  (hexp(reg_x.y) - half_1);

        reinterpret_cast<half2*>(y)[idx] = reg_y;
    }
}
```
看了`示例`的代码才意识到 half 的所有计算都有专门的运算函数。于是按 `示例` 做了调整。
```
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
```

# benchmark
-------------------------------------------------------------------------------------
                                        S=1024, K=1024
           out_f32: ['0.02439803  ', '-0.88201714 '], time:0.01034117ms
         out_f32x4: ['0.02439803  ', '-0.88201714 '], time:0.00943661ms
        out_f32_th: ['0.02439803  ', '-0.88201714 '], time:0.06954718ms
-------------------------------------------------------------------------------------
           out_f16: ['0.0243988   ', '-0.88183594 '], time:0.01043034ms
         out_f16x2: ['0.0243988   ', '-0.88183594 '], time:0.00900793ms
         out_f16x8: ['0.0243988   ', '-0.88183594 '], time:0.00878644ms
     out_f16x8pack: ['0.0243988   ', '-0.88183594 '], time:0.00900698ms
        out_f16_th: ['0.0243988   ', '-0.88183594 '], time:0.06552386ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=1024, K=2048
           out_f32: ['-0.26625383 ', '-0.89974725 '], time:0.01228237ms
         out_f32x4: ['-0.26625383 ', '-0.89974725 '], time:0.01095343ms
        out_f32_th: ['-0.26625383 ', '-0.89974725 '], time:0.07885909ms
-------------------------------------------------------------------------------------
           out_f16: ['-0.26611328 ', '-0.89990234 '], time:0.01234651ms
         out_f16x2: ['-0.26611328 ', '-0.89990234 '], time:0.01124597ms
         out_f16x8: ['-0.26611328 ', '-0.89990234 '], time:0.00975752ms
     out_f16x8pack: ['-0.26611328 ', '-0.89990234 '], time:0.00975633ms
        out_f16_th: ['-0.26611328 ', '-0.89990234 '], time:0.07081008ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=1024, K=4096
           out_f32: ['0.43156666  ', '-0.2943188  '], time:0.01659083ms
         out_f32x4: ['0.43156666  ', '-0.2943188  '], time:0.01419020ms
        out_f32_th: ['0.43156666  ', '-0.29431874 '], time:0.09580398ms
-------------------------------------------------------------------------------------
           out_f16: ['0.43164062  ', '-0.29443359 '], time:0.01665783ms
         out_f16x2: ['0.43164062  ', '-0.29443359 '], time:0.01648188ms
         out_f16x8: ['0.43164062  ', '-0.29443359 '], time:0.01163602ms
     out_f16x8pack: ['0.43164062  ', '-0.29443359 '], time:0.01150799ms
        out_f16_th: ['0.43164062  ', '-0.29443359 '], time:0.07743192ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=2048, K=1024
           out_f32: ['-0.6480642  ', '0.23097877  '], time:0.01274109ms
         out_f32x4: ['-0.6480642  ', '0.23097877  '], time:0.01112485ms
        out_f32_th: ['-0.6480642  ', '0.23097877  '], time:0.07794285ms
-------------------------------------------------------------------------------------
           out_f16: ['-0.64794922 ', '0.23095703  '], time:0.01324439ms
         out_f16x2: ['-0.64794922 ', '0.23095703  '], time:0.01243973ms
         out_f16x8: ['-0.64794922 ', '0.23095703  '], time:0.00986743ms
     out_f16x8pack: ['-0.64794922 ', '0.23095703  '], time:0.00976467ms
        out_f16_th: ['-0.64794922 ', '0.23095703  '], time:0.06970835ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=2048, K=2048
           out_f32: ['-0.1370312  ', '-0.55019927 '], time:0.01657438ms
         out_f32x4: ['-0.1370312  ', '-0.55019927 '], time:0.01426935ms
        out_f32_th: ['-0.1370312  ', '-0.55019927 '], time:0.09923792ms
-------------------------------------------------------------------------------------
           out_f16: ['-0.13720703 ', '-0.54980469 '], time:0.01671338ms
         out_f16x2: ['-0.13720703 ', '-0.54980469 '], time:0.01457596ms
         out_f16x8: ['-0.13720703 ', '-0.54980469 '], time:0.01162219ms
     out_f16x8pack: ['-0.13720703 ', '-0.54980469 '], time:0.01132774ms
        out_f16_th: ['-0.13720703 ', '-0.54980469 '], time:0.07795954ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=2048, K=4096
           out_f32: ['1.23178411  ', '0.68323904  '], time:0.02553725ms
         out_f32x4: ['1.23178411  ', '0.68323904  '], time:0.02089000ms
        out_f32_th: ['1.23178411  ', '0.68323904  '], time:0.28684878ms
-------------------------------------------------------------------------------------
           out_f16: ['1.23144531  ', '0.68310547  '], time:0.02536631ms
         out_f16x2: ['1.23144531  ', '0.68310547  '], time:0.02505541ms
         out_f16x8: ['1.23144531  ', '0.68310547  '], time:0.01502752ms
     out_f16x8pack: ['1.23144531  ', '0.68310547  '], time:0.01474953ms
        out_f16_th: ['1.23144531  ', '0.68310547  '], time:0.09794211ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=4096, K=1024
           out_f32: ['2.46254206  ', '0.02073942  '], time:0.01787829ms
         out_f32x4: ['2.46254206  ', '0.02073942  '], time:0.01433063ms
        out_f32_th: ['2.46254206  ', '0.02073942  '], time:0.10575986ms
-------------------------------------------------------------------------------------
           out_f16: ['2.46289062  ', '0.02073669  '], time:0.01886225ms
         out_f16x2: ['2.46289062  ', '0.02073669  '], time:0.01297355ms
         out_f16x8: ['2.46289062  ', '0.02073669  '], time:0.01164198ms
     out_f16x8pack: ['2.46289062  ', '0.02073669  '], time:0.01145458ms
        out_f16_th: ['2.46289062  ', '0.02073669  '], time:0.07827806ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=4096, K=2048
           out_f32: ['-0.31856847 ', '0.22570813  '], time:0.02531433ms
         out_f32x4: ['-0.31856847 ', '0.22570813  '], time:0.02105856ms
        out_f32_th: ['-0.31856847 ', '0.22570813  '], time:0.28442907ms
-------------------------------------------------------------------------------------
           out_f16: ['-0.31835938 ', '0.22570801  '], time:0.02533841ms
         out_f16x2: ['-0.31835938 ', '0.22570801  '], time:0.02142453ms
         out_f16x8: ['-0.31835938 ', '0.22570801  '], time:0.01493716ms
     out_f16x8pack: ['-0.31835938 ', '0.22570801  '], time:0.01466966ms
        out_f16_th: ['-0.31835938 ', '0.22570801  '], time:0.09697843ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        S=4096, K=4096
           out_f32: ['0.65886843  ', '0.5854739   '], time:0.19540477ms
         out_f32x4: ['0.65886843  ', '0.5854739   '], time:0.19368672ms
        out_f32_th: ['0.65886843  ', '0.5854739   '], time:0.94934630ms
-------------------------------------------------------------------------------------
           out_f16: ['0.65869141  ', '0.58544922  '], time:0.04269123ms
         out_f16x2: ['0.65869141  ', '0.58544922  '], time:0.04240012ms
         out_f16x8: ['0.65869141  ', '0.58544922  '], time:0.02185416ms
     out_f16x8pack: ['0.65869141  ', '0.58544922  '], time:0.02131605ms
        out_f16_th: ['0.65869141  ', '0.58544922  '], time:0.33389688ms
-------------------------------------------------------------------------------------