# mat_transpose
包含以下kernel：
- mat_transpose_f32_col2row_kernel
- mat_transpose_f32_row2col_kernel
- mat_transpose_f32x4_col2row_kernel(float4向量化版本)
- mat_transpose_f32x4_row2col_kernel(float4向量化版本)
- mat_transpose_f32_diagonal(对角轴应用于S=K)
- mat_transpose_f32x4_shared_col2row_kernel(float4向量化版本，共享内存)
- mat_transpose_f32x4_shared_row2col_kernel(float4向量化版本，共享内存)
- mat_transpose_f32x4_shared_bcf_col2row_kernel(float4向量化版本，共享内存，去bank conflict)
- mat_transpose_f32x4_shared_bcf_row2col_kernel(float4向量化版本，共享内存，去bank conflict)
- mat_transpose_cute_row2col_reg
- mat_transpose_cute_col2row_reg
- mat_transpose_cute_col_smem
- mat_transpose_cute_row_smem
- mat_transpose_cute_col_smem_swizzled (bank conflict free)
- mat_transpose_cute_row_smem_swizzled
- mat_transpose_cute_row_cvectorized
- mat_transpose_cute_row_cvectorized_swizzled
- mat_transpose_cute_row_rvectorized
- mat_transpose_cute_row_rvectorized_swizzled


$$
x=\begin{pmatrix}
    1 & 2\\
    3 & 4\\
    5 & 6
\end{pmatrix}
\to y=\begin{pmatrix}
    1 & 3 & 5\\
    2 & 4 & 6
\end{pmatrix}
$$

## col2row 和 row2col
x: row x col  
y: col x row 

前提：  矩阵在内存中是连续存储的，那么矩阵 x 在内存中的就是 `1,2,3,4,5,6`, 矩阵y在内存中就是 `1,3,5,2,4,6`,  **col2row** 就是把 x 中的 `1,3,5` 放到 y 中， x就是跳跃的读取， y是连续的写入； 而 **row2col** 就是 x 是连续的读取，y 是跳跃的写入。
 
`int idx = blockIdx.x * blockDim.x + threadIdx.x;`
假设对 y 是连续的写入(col2row)，则对于任意位置 i，其在 y 中的行、列的计算为：  
```
i_row = idx / row;
i_col = idx % row;
```
`y[i_row][i_col]` 对应` x[i_col][i_row]`, 那么 `y[i] = x[i_col * col + i_row];`  

假设对 x 是连续的读取(row2col)，则对于任意位置 i，其在 x 中的行、列的计算为：  
```
i_row = idx / col;   
i_col = idx % col;  
```
`x[i_row][i_col]` 对应 `y[i_col][i_row]`, 那么 `y[i_col * row + i_row] = x[i];`

可以这么理解：col2row 和 row2col 区别在于 `idx = blockIdx.x * blockDim.x + threadIdx.x` 是针对谁的索引。

但是 GPU 中是每个线程去计算一个元素，其 `"连续"` 体现在什么地方？
## 合并访问
关于全局内存的访问模式，有合并（coalesced）与非合并（uncoalesced）之分。合并访问指的是一个线程束对全局内存的一次访问请求（读或者写）导致最少数量的数据传输，否则称访问是非合并的。
也就是说如果一个线程束内的线程访问的是连续且对齐的内存，则需要的内存传输最少。
所以 row2col 是合并访存读取， col2row是合并访存写入。

## col2row2d 和 row2col2d
```
//col2row  idx针对的是输出y
int ix = blockDim.x * blockIdx.x + threadIdx.x;
int iy = blockDim.y * blockIdx.y + threadIdx.y;

if(iy < col & ix < row){
    y[iy * row + ix] = x[ix * col + iy];
}

//row2col idx针对的输入x
int ix = blockDim.x * blockIdx.x + threadIdx.x;
int iy = blockDim.y * blockIdx.y + threadIdx.y;

if(ix < col & iy < row> )
y[ix * row + iy] = x[iy * col + ix];
```
## fp32x4
从之前(day1~day10)的kernel可知道，fp32x4向量化就是一次性读取4个输入数据，在计算完成后一次性写入到输出中。
### fp32x4 col2row
输出连续写入4个元素，默认行列能被4整除，则这四个元素转置前在x中是同一列。
```
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
int y1_row = idx / row;
int y1_col = idx % col; // y[y1_row * row + y1_col] 

float4 reg_x;

reg_x.x = x[      y1_col * col + y1_row];
reg_x.y = x[(y1_col + 1) * col + y1_row];
reg_x.z = x[(y1_col + 2) * col + y1_row];
reg_x.w = x[(y1_col + 3) * col + y1_row];

reinterpret_cast<float4 *>(y + idx)[0] = reg_x;
```
### fp32x4 row2col
输入取连续的4个元素，默认行列能被4整除，则这四个元素转置后在y中是同一列。
```
int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
float4 reg_x = reinterpret_cast<float4*>(x + idx)[0];

int xi_row = idx / col;
int xi_col = idx % col;


y[xi_col * row + xi_row]       = reg_x.x;
y[(xi_col + 1) * row + xi_row] = reg_x.y;
y[(xi_col + 2) * row + xi_row] = reg_x.z;
y[(xi_col + 3) * row + xi_row] = reg_x.w;
```

## shared memory kernel
shared memory节省了全局内存的读取次数，那么我们需要先将数据加载到shared memory，再进行计算。
假设block size 是 32x32, 定义`__shared__ float s_x[32][32];`， 则 s_x 的索引方式是 `s_x[threadIdx.y][threadIdx.x]`。  
我们第一步是将部分 x 加载到 s_x 中.
```
//col2row
int ix = blockDim.x * blockIdx.x + threadIdx.x;
int iy = blockDim.y * blockIdx.y + threadIdx.y;
__shared__ float s_x[32][32];

s_x[threadIdx.y][threadIdx.x] =  x[ix * col + iy];
__syncthreads();

y[iy * row + ix]  = s_x[threadIdx.y][threadIdx.x];


//row2col
int ix = blockDim.x * blockIdx.x + threadIdx.x;
int iy = blockDim.y * blockIdx.y + threadIdx.y;
__shared__ float s_x[32][32];

s_x[threadIdx.y][threadIdx.x] = x[iy * col + ix];
__syncthreads();

y[ix * row + iy] = s_x[threadIdx.y][threadIdx.x];
```

## bank conflict
shared memory 划分为大小相等、可以被同时访问的模块，称为 bank。一般为32个bank，对应 warp 中的 thread.  
假设有一个内存读/写请求，有 n 个地址， 这些地址落在 n 个不同的内存 bank 中，就能获得比单个模块高n倍的带宽。
然而，如果内存请求的两个地址位于同一个bank中，就会出现bank conflict, 访问就必须串行化。  
注意是同一个bank的两个地址。如果访问的同一个bank中的同一个地址，这样是没有bank conflict.




## benchmark
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.09649754ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.01936078ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.03108358ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01538706ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01545000ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.01785493ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.03134322ms
         out_f32_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01191688ms
         out_f32_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01193929ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.01694202ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.03088951ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.05555034ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.02645469ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.02640271ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.02948093ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.05497646ms
         out_f32_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.01543832ms
         out_f32_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.01547408ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.01948643ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.05403399ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.10207295ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.04809332ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.04801416ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.05430889ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.10067511ms
         out_f32_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.02295613ms
         out_f32_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.02293801ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=1024, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.05765462ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.09880042ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.19306684ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.08934116ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.09172702ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.10134172ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.19170690ms
         out_f32_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.03974152ms
         out_f32_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.03962731ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.01605368ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.03104711ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.05669069ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02370143ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02365661ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.03116417ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.05631566ms
         out_f32_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.01530194ms
         out_f32_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.01538134ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.01957560ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.05375528ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.10584569ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.04738379ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.04725027ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.05447102ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.10448337ms
         out_f32_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.02245903ms
         out_f32_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.02240801ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.05933261ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.10649276ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.19870663ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.08967733ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.09101534ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.10898924ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.19498253ms
         out_f32_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.03762817ms
         out_f32_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.03768969ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=2048, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.21445203ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.27488661ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.39232707ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.23497057ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.23621964ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.26685596ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.39303446ms
         out_f32_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.22533202ms
         out_f32_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.22537804ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.01951075ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.05420041ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.10492706ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.04720879ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.04721260ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.05454493ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.10485578ms
         out_f32_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.02303529ms
         out_f32_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.02295041ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.06028557ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.10358071ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.20091105ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.08855963ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.08855748ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.10481381ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.20011520ms
         out_f32_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.03754783ms
         out_f32_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.03748107ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.21568704ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.37528300ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.39656687ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.22059298ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.22185540ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.30088687ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.39897323ms
         out_f32_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.21113777ms
         out_f32_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.21245027ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.42162991ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:0.71340036ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:0.75776911ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.48649573ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.48694038ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:0.63712382ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:0.76208186ms
         out_f32_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.45714211ms
         out_f32_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.45435333ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=1024
                       out_original: [0.0, 1.0, 1024.0], validate False, time:0.06053543ms
                    out_f32_col2row: [0.0, 1024.0, 1.0], validate True , time:0.09960365ms
                    out_f32_row2col: [0.0, 1024.0, 1.0], validate True , time:0.20161510ms
                out_f32_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.09139347ms
                out_f32_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.09147215ms
                  out_f32x4_col2row: [0.0, 1024.0, 1.0], validate True , time:0.10045195ms
                  out_f32x4_row2col: [0.0, 1024.0, 1.0], validate True , time:0.19914699ms
         out_f32_shared_col2row(2d): [0.0, 1024.0, 1.0], validate True , time:0.03908300ms
         out_f32_shared_row2col(2d): [0.0, 1024.0, 1.0], validate True , time:0.03901172ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=2048
                       out_original: [0.0, 1.0, 2048.0], validate False, time:0.21493340ms
                    out_f32_col2row: [0.0, 2048.0, 1.0], validate True , time:0.47788787ms
                    out_f32_row2col: [0.0, 2048.0, 1.0], validate True , time:0.40239453ms
                out_f32_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.22269773ms
                out_f32_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.22056675ms
                  out_f32x4_col2row: [0.0, 2048.0, 1.0], validate True , time:0.38777661ms
                  out_f32x4_row2col: [0.0, 2048.0, 1.0], validate True , time:0.40359497ms
         out_f32_shared_col2row(2d): [0.0, 2048.0, 1.0], validate True , time:0.21252942ms
         out_f32_shared_row2col(2d): [0.0, 2048.0, 1.0], validate True , time:0.21182489ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096
                       out_original: [0.0, 1.0, 4096.0], validate False, time:0.42221594ms
                    out_f32_col2row: [0.0, 4096.0, 1.0], validate True , time:0.99105573ms
                    out_f32_row2col: [0.0, 4096.0, 1.0], validate True , time:0.77326846ms
                out_f32_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.47339463ms
                out_f32_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.47317171ms
                  out_f32x4_col2row: [0.0, 4096.0, 1.0], validate True , time:0.79385543ms
                  out_f32x4_row2col: [0.0, 4096.0, 1.0], validate True , time:0.77381611ms
         out_f32_shared_col2row(2d): [0.0, 4096.0, 1.0], validate True , time:0.45013690ms
         out_f32_shared_row2col(2d): [0.0, 4096.0, 1.0], validate True , time:0.45168400ms
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192
                       out_original: [0.0, 1.0, 8192.0], validate False, time:0.82729101ms
                    out_f32_col2row: [0.0, 8192.0, 1.0], validate True , time:1.93113256ms
                    out_f32_row2col: [0.0, 8192.0, 1.0], validate True , time:1.51194859ms
                out_f32_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:1.02180648ms
                out_f32_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:1.02312326ms
                  out_f32x4_col2row: [0.0, 8192.0, 1.0], validate True , time:1.55509400ms
                  out_f32x4_row2col: [0.0, 8192.0, 1.0], validate True , time:1.51401544ms
         out_f32_shared_col2row(2d): [0.0, 8192.0, 1.0], validate True , time:0.95149851ms
         out_f32_shared_row2col(2d): [0.0, 8192.0, 1.0], validate True , time:0.95202160ms
----------------------------------------------------------------------------------------------------------------------------------


..... To be continued