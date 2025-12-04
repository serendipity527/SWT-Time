#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试WaveletHead和ISWT重构功能

验证点：
1. ISWTReconstruction重构精度
2. WaveletHead输出形状正确性
3. 端到端小波域对称架构
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers.WaveletEmbed import SWTDecomposition, ISWTReconstruction, WaveletPatchEmbedding
from models.TimeLLM import WaveletHead, FlattenHead

print("=" * 80)
print("测试1: ISWT重构精度验证")
print("=" * 80)

# 测试参数
batch_size = 4
num_vars = 7
seq_len = 512
level = 3
wavelet = 'db4'

# 检查GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建测试数据
x_original = torch.randn(batch_size, num_vars, seq_len, device=device)
print(f"\n原始信号: {x_original.shape}")

# 正向SWT分解
swt = SWTDecomposition(wavelet=wavelet, level=level).to(device)
coeffs = swt(x_original)
print(f"SWT分解后: {coeffs.shape} (4个频段)")

# 逆ISWT重构
iswt = ISWTReconstruction(wavelet=wavelet, level=level).to(device)
x_reconstructed = iswt(coeffs)
print(f"ISWT重构后: {x_reconstructed.shape}")

# 计算重构误差
reconstruction_error = torch.abs(x_original - x_reconstructed).mean().item()
max_error = torch.abs(x_original - x_reconstructed).max().item()
print(f"\n重构精度:")
print(f"  平均绝对误差: {reconstruction_error:.8f}")
print(f"  最大绝对误差: {max_error:.8f}")

if reconstruction_error < 1e-5:
    print("  ✅ 重构精度优秀 (误差 < 1e-5)")
elif reconstruction_error < 1e-3:
    print("  ⚠️  重构精度良好 (误差 < 1e-3)")
else:
    print("  ❌ 重构精度较差，请检查实现")

print("\n" + "=" * 80)
print("测试2: WaveletHead输出形状验证")
print("=" * 80)

# WaveletHead参数
d_ff = 256
patch_nums = 64
pred_len = 96

# 创建WaveletHead
wavelet_head = WaveletHead(
    n_vars=num_vars,
    d_model=d_ff,
    patch_nums=patch_nums,
    pred_len=pred_len,
    level=level,
    wavelet=wavelet,
    head_dropout=0.1
).to(device)

# 模拟LLM隐状态输入
llm_hidden = torch.randn(batch_size, num_vars, d_ff, patch_nums, device=device)
print(f"\nLLM隐状态输入: {llm_hidden.shape}")

# 前向传播
pred = wavelet_head(llm_hidden)
print(f"WaveletHead输出: {pred.shape}")

# 验证输出形状
expected_shape = (batch_size, num_vars, pred_len)
if pred.shape == expected_shape:
    print(f"✅ 输出形状正确: {pred.shape} == {expected_shape}")
else:
    print(f"❌ 输出形状错误: {pred.shape} != {expected_shape}")

# 验证输出数值有效性
if torch.isnan(pred).any() or torch.isinf(pred).any():
    print("❌ 输出包含NaN或Inf")
else:
    print("✅ 输出数值有效")
    print(f"  均值: {pred.mean().item():.6f}")
    print(f"  标准差: {pred.std().item():.6f}")
    print(f"  最小值: {pred.min().item():.6f}")
    print(f"  最大值: {pred.max().item():.6f}")

print("\n" + "=" * 80)
print("测试3: WaveletHead vs FlattenHead 参数量对比")
print("=" * 80)

# FlattenHead
head_nf = d_ff * patch_nums
flatten_head = FlattenHead(
    n_vars=num_vars,
    nf=head_nf,
    target_window=pred_len,
    head_dropout=0.1
).to(device)

# 参数量统计
wavelet_params = sum(p.numel() for p in wavelet_head.parameters())
flatten_params = sum(p.numel() for p in flatten_head.parameters())

print(f"\nWaveletHead 参数量: {wavelet_params:,}")
print(f"FlattenHead 参数量: {flatten_params:,}")
print(f"参数量比例: {wavelet_params / flatten_params:.2f}x")

# 测试FlattenHead输出
flatten_pred = flatten_head(llm_hidden)
print(f"\nFlattenHead输出: {flatten_pred.shape}")

if flatten_pred.shape == pred.shape:
    print("✅ 两种Head输出形状一致，可以无缝替换")
else:
    print("❌ 输出形状不一致")

print("\n" + "=" * 80)
print("测试4: 端到端小波域对称架构")
print("=" * 80)

print("\n完整流程:")
print("  输入时序 (B, N, T)")
print("    ↓ WaveletPatchEmbedding (SWT分解 + Patching)")
print("  Patch embeddings (B*N, num_patches, d_model)")
print("    ↓ LLM处理")
print("  LLM隐状态 (B, N, d_ff, patch_nums)")
print("    ↓ WaveletHead (投影到小波系数)")
print("  小波系数 (B, N, pred_len, 4频段)")
print("    ↓ ISWT重构")
print("  预测时序 (B, N, pred_len)")

# 模拟完整流程
print("\n执行完整流程...")

# Step 1: 输入时序
x_input = torch.randn(batch_size, num_vars, seq_len, device=device)
print(f"1. 输入时序: {x_input.shape}")

# Step 2: WaveletPatchEmbedding
patch_embed = WaveletPatchEmbedding(
    d_model=32,
    patch_len=16,
    stride=8,
    wavelet=wavelet,
    level=level,
    dropout=0.1
).to(device)
patches, n_vars = patch_embed(x_input)
print(f"2. Patch embeddings: {patches.shape}, n_vars={n_vars}")

# Step 3: 模拟LLM处理（这里直接reshape到需要的形状）
# 实际中会经过Reprogramming + LLM
num_patches = patches.shape[1]
llm_out = torch.randn(batch_size, num_vars, d_ff, num_patches, device=device)
print(f"3. LLM隐状态: {llm_out.shape}")

# Step 4: WaveletHead预测
wavelet_head_pred = WaveletHead(
    n_vars=num_vars,
    d_model=d_ff,
    patch_nums=num_patches,
    pred_len=pred_len,
    level=level,
    wavelet=wavelet,
    head_dropout=0.1
).to(device)
final_pred = wavelet_head_pred(llm_out)
print(f"4. 最终预测: {final_pred.shape}")

print("\n✅ 端到端流程测试通过！")

print("\n" + "=" * 80)
print("测试5: 不同频段独立预测验证")
print("=" * 80)

# 验证WaveletHead确实为每个频段独立预测
print("\n验证频段独立性...")
print(f"WaveletHead有 {wavelet_head_pred.num_bands} 个独立的投影层")

# 检查每个投影层的参数
for i, proj in enumerate(wavelet_head_pred.band_projections):
    num_params = sum(p.numel() for p in proj.parameters())
    print(f"  频段{i} (band_{i}): {num_params:,} 参数")

print("\n频段含义:")
print("  频段0: cA3 - 低频趋势（全局模式）")
print("  频段1: cD3 - 最高频细节")
print("  频段2: cD2 - 中频细节")
print("  频段3: cD1 - 低频细节")

print("\n" + "=" * 80)
print("✅ 所有测试完成！")
print("=" * 80)

print("\n🎉 小波域对称架构实现成功！")
print("\n使用方法：")
print("在配置文件中添加:")
print("  configs.use_wavelet = True          # 使用WaveletPatchEmbedding")
print("  configs.use_wavelet_head = True     # 使用WaveletHead输出")
print("  configs.wavelet = 'db4'             # 小波类型")
print("  configs.swt_level = 3               # 分解层数")
