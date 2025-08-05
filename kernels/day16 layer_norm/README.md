# LayerNorm
- layer_norm_f32_kernel
- layer_norm_f32x4_kernel
- layer_norm_f16_f16_kernel
- layer_norm_f16x2_f16_kernel
- layer_norm_f16x8_f16_kernel
- layer_norm_f16x8_pack_f16_kernel
- layer_norm_f16x8_pack_f32_kernel
- layer_norm_f16_f32_kernel

计算公式：
$$
\mu = \frac{1}{N}\sum^N_{i=1}x_i\\
\sigma^2 = \frac{1}{N}\sum^N_{i=1}(x_i - \mu)^2 \\
layer\_norm(x_i) = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} * g + b
$$