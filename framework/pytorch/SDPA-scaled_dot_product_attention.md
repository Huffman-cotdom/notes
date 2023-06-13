![各 kernal 优化效果对比](assents/Pasted%20image%2020230511213316.png)
```python
torch.nn.functional.scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) → Tensor:
"""
Args:
query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
attn_mask (optional Tensor): Attention mask; shape :math:`(N, ..., L, S)`. Two types of masks are supported.
    A boolean mask where a value of True indicates that the element *should* take part in attention.
    A float mask of the same type as query, key, value that is added to the attention score.
    dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
is_causal (bool): If true, assumes causal attention masking and errors if both attn_mask and is_causal
are set.
    scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
to :math:`\frac{1}{\sqrt{E}}`.
Returns:
output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.
"""
pass
```
SDPA 实现了 attention 模块最核心的部分（缩放的点乘注意力），这个函数等价于以下代码：

```python
scale_factor = 1 / math.sqrt(Q.size(-1)) if scale is None else scale
attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
attn_weight = torch.softmax((Q @ K.transpose(-2, -1) * scale_factor) + attn_mask, dim=-1)
attn_weight = torch.dropout(attn_weight, dropout_p)
return attn_weight @ V
```

SDPA 之所以能带来性能的加速，主要是它背后已经实现了优化的 kernels，目前 SDPA 支持三种 kernels：
-   **sdpa_flash：**[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
-   **sdpa_mem_eff:**[Memory-Efficient Attention](https://github.com/facebookresearch/xformers)
-   **sdpa_math：**A PyTorch implementation defined in C++

其中 sdpa_flash 支持在 SM80+架构的 GPUs 上使用 FP16 精度训练和推理，而 sdpa_mem_eff 支持在大部分 GPUs 上采用 FP16 和 FP32 精度训练和推理。如果上述两个 kernel 都不支持的话，那么就只能采用 sdpa_math 了，它是直接基于 C++的通用实现。默认情况下，这三个 kernel 都是开启的，当你调用 SDPA 时，它将根据你的输入选择一个最优的 kernel 来进行执行。

大部分情况下，我们不需要关注背后具体所选择的 kernel，因为它背后已经做了最优的选择。但是如果你想显式控制所使用的 kernel，那么可以采用[torch.backends.cuda.sdp_kernel()](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel)来关闭具体的 kernels，它是一个上下文管理器，比如我们要关闭 sdpa_math，那么可以这样调用：
```python
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
with torch.backends.cuda.sdp_kernel(enable_math=False):
    F.scaled_dot_product_attention(query, key, value)
```

由于 sdpa_math 被关闭，那么此时系统只能从 sdpa_flash 和 sdpa_mem_eff 这个两个 kernel 进行选择了。当你关闭两个 kernel，那么就等同于直接选择使用剩下的那个 kernel 来进行实现了，比如下面的代码就相当于显式采用 sdpa_mem_eff 这个 kernel 了：

```python
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
    F.scaled_dot_product_attention(query, key, value)
```

加速对比：
```python
import torch
import torch.utils.benchmark as benchmark
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.nn.functional as F

# Lets define a helpful benchmarking function:
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 32
embed_dimension = 32

dtype = torch.float16
device = "cuda"

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# Lets explore the speed of each of the 3 implementations

# Helpful arg mapper
backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

with sdp_kernel(**backend_map[SDPBackend.MATH]):
    print(f"The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
    try:
        print(f"The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
    try:
        print(f"The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")
```

在 V100 机器上的运行结果如下所示：

> The default implementation runs in 6569.854 microseconds  
> The math implementation runs in 16091.686 microseconds  
> <timeit-src>:6: UserWarning: Memory efficient kernel not used because: (Triggered internally at ../aten/src/ATen/native/transformers/cuda/sdp_utils.h:527.)  
> <timeit-src>:6: UserWarning: Memory Efficient attention has been runtime disabled. (Triggered internally at ../aten/src/ATen/native/transformers/cuda/sdp_utils.h:338.)  
> <timeit-src>:6: UserWarning: Flash attention kernel not used because: (Triggered internally at ../aten/src/ATen/native/transformers/cuda/sdp_utils.h:529.)  
> <timeit-src>:6: UserWarning: Flash attention only supports sm75 and sm8x gpu architectures. Attempting to run on a sm 7.0 gpu. (Triggered internally at ../aten/src/ATen/native/transformers/cuda/sdp_utils.h:352.)  
> FlashAttention is not supported. See warnings for reasons.  
> The memory efficient implementation runs in 6595.339 microseconds

V100 卡属于 sm 7.0，不支持 Flash attention，但是我们可以看到默认采用的 kernel 是 sdpd_mem_eff，它相比 sdpd_math，速度提升非常明显（6ms vs 16ms）。当我们把机器换成 A100 后，运行结果如下所示：

> The default implementation runs in 2831.521 microseconds  
> The math implementation runs in 7001.696 microseconds  
> The flash attention implementation runs in 2829.635 microseconds  
> The memory efficient implementation runs in 3011.410 microseconds

A100 卡上是支持 Flash attention，而且默认的实现方式是**sdpa_flash**，此时运行时间最短，A100 比 V100 快了 2 倍多。