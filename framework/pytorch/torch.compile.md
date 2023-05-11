`torch.compile` 100% 向后兼容，支撑`torch.compile`是新技术——TorchDynamo、AOTAutograd、PrimTorch 和 TorchInductor。

-   **TorchDynamo**使用 Python Frame Evaluation Hooks 安全地捕获 PyTorch 程序。解析 Python 字节码，可以 trace model

-   **AOTAutograd**重载 PyTorch 的 autograd 引擎作为跟踪 autodiff，用于生成提前向后跟踪。
   
-   **PrimTorch**将约 2000 多个 PyTorch 运算符规范化为一组约 250 个原始运算符的封闭集，开发人员可以将其作为目标来构建完整的 PyTorch 后端。这大大降低了编写 PyTorch 功能或后端的障碍。

-   **TorchInductor**是一种深度学习编译器，可为多个加速器和后端生成快速代码。对于 NVIDIA 和 AMD GPU，它使用 OpenAI Triton 作为关键构建块。pytorch 模型 lower 为 IR 然后生成高性能的 CUDA 或者 CPU 代码

TorchDynamo、AOTAutograd、PrimTorch 和 TorchInductor 是用 Python 编写的，支持动态形状（即能够发送不同大小的张量而无需重新编译）

`torch.compile` 在 NVIDIA A100 GPU 上的训练速度提高了 43%。在 Float32 精度下，它的平均运行速度提高了 21%，而在 AMP 精度下，它的运行速度平均提高了 51%。默认后端 TorchInductor 支持 CPU 以及 NVIDIA Volta 和 Ampere GPU。它（目前）不支持其他 GPU、xPU 或较旧的 NVIDIA GPU。
![A100 加速效果](assents/Pasted%20image%2020230511204820.png)

## 使用

```python
compiled_model = torch.compile(model)
```

```python
def torch.compile(model: Callable,
  *,
  mode: Optional[str] = "default",
  dynamic: bool = False,
  fullgraph:bool = False,
  backend: Union[str, Callable] = "inductor",
  # advanced backend options go here as kwargs
  **kwargs
) -> torch._dynamo.NNOptimizedModule
```
-   **mode** 指定编译器在编译时应该优化什么。
    -   默认模式是一种预设，它会尝试在不花费太长时间编译或使用额外内存的情况下高效编译。
    -   其他模式，如`reduce-overhead`更多地减少框架开销，但会消耗少量额外内存。`max-autotune`编译很长时间，试图为您提供它可以生成的最快代码。
-   **dynamic** 指定是否启用动态形状的代码。某些编译器优化不能应用于动态整形程序。明确表示您想要一个具有动态形状还是静态形状的编译程序将有助于编译器为您提供更好的优化代码。
-   **fullgraph** 类似于 Numba 的`nopython`. 它将整个程序编译成一个图形，或者给出一个错误来解释为什么它不能这样做。大多数用户不需要使用此模式。如果您非常注重性能，那么您可以尝试使用它。
-   **background**指定要使用的编译器后端。默认情况下，使用 TorchInductor，但还有一些其他可用的。

```python
import torch
import torchvision.models as models

model = models.resnet18().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
compiled_model = torch.compile(model)

x = torch.randn(16, 3, 224, 224).cuda()
optimizer.zero_grad()
out = compiled_model(x)
out.sum().backward()
optimizer.step()
```
第一次运行时`compiled_model(x)`，它会编译模型。因此，运行需要更长的时间。随后的运行速度很快。
`reduce-overhead`的意思是适合小模型，而`max-autotune`则是相当于 trt 或者 tvm 那样对整个模型进行编译优化

```python
# API NOT FINAL
# default: optimizes for large models, low compile-time
#          and no extra memory usage
torch.compile(model)

# reduce-overhead: optimizes to reduce the framework overhead
#                and uses some extra memory. Helps speed up small models
torch.compile(model, mode="reduce-overhead")

# max-autotune: optimizes to produce the fastest model,
#               but takes a very long time to compile
torch.compile(model, mode="max-autotune")
```

### 保存模型
```python
torch.save(optimized_model.state_dict(), "foo.pt")
# both these lines of code do the same thing
torch.save(model.state_dict(), "foo.pt")
```

- 无法直接直接保存优化好的模型
```python
torch.save(optimized_model, "foo.pt") # Error
torch.save(model, "foo.pt")           # Works
```

### 推理和导出
用 torch.compile 生成编译模型后，在实际模型服务之前运行一些预热步骤。这有助于缓解初始服务期间的延迟峰值。
`torch.export`该模式会为需要保证和可预测延迟的环境仔细导出整个模型和守卫基础设施。`torch.export`将需要更改您的程序，特别是如果您有数据相关的控制流
```python
# API Not Final
exported_model = torch._dynamo.export(model, input)
torch.save(exported_model, "foo.pt")
```

### 动态形状加速效果
一个关键要求是支持动态形状，并允许模型采用不同大小的张量，而无需在每次形状更改时重新编译。
![动态形状加速效果](assents/Pasted%20image%2020230511210830.png)

### 分布式
> 目前应该都还用不了

torch.distributed 的两个主要分布式包装器在编译模式下运行良好。
`DistributedDataParallel`(DDP) 和 (FSDP) 都`FullyShardedDataParallel`在 indeuctor 模式下工作，并且相对于 eager 模式提供了改进的性能和内存利用率。
![DDP AMP 加速](assents/Pasted%20image%2020230511211301.png)
![FSDP AMP 加速](assents/Pasted%20image%2020230511211359.png)
![FSDP 内存占用](assents/Pasted%20image%2020230511211434.png)

OTAutograd 以提前的方式动态捕获 autograd 逻辑，以 FX 图形格式生成前向和后向运算符的图形。
![](assents/Pasted%20image%2020230511211810.png)