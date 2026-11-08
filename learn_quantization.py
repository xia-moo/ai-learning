import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

torch.manual_seed(42)

# 一个尚未训练的小模型，仅用于观察数值变化
model_fp32 = nn.Sequential(
    nn.Linear(4, 3)
).eval()

# 转换其中的 Linear，保留原始模型
model_int8 = quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8,
    inplace=False
)

X = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

with torch.no_grad():
    output_fp32 = model_fp32(X)
    output_int8 = model_int8(X)

print("原始权重类型：", model_fp32[0].weight.dtype)
print("量化权重类型：", model_int8[0].weight().dtype)

print("原始输出：", output_fp32)
print("量化后输出：", output_int8)

difference = (output_fp32 - output_int8).abs().max().item()
print("最大绝对差异：", difference)
print("量化模型输出类型：", output_int8.dtype)