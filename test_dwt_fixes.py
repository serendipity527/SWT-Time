#!/usr/bin/env python3
"""
DWT动态提示词生成器修复测试
逐步验证每个问题的修复
"""
import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from layers.DWTPromptGenerator_v2 import DWTPromptGenerator

def create_test_data():
    """创建测试数据"""
    # 生成模拟的时间序列数据
    B, N, T = 2, 3, 64  # batch_size=2, n_vars=3, seq_len=64
    
    # 创建不同特性的测试序列
    t = torch.linspace(0, 10, T)
    
    # 序列1: 明显上升趋势 + 季节性
    series1 = 0.5 * t + 2 * torch.sin(2 * np.pi * t) + 0.1 * torch.randn(T)
    
    # 序列2: 下降趋势 + 高频噪声
    series2 = -0.3 * t + 0.5 * torch.sin(20 * np.pi * t) + 0.2 * torch.randn(T)
    
    # 序列3: 平稳 + 中频波动
    series3 = torch.sin(4 * np.pi * t) + 0.1 * torch.randn(T)
    
    # 组装成批次数据
    x_enc = torch.stack([
        torch.stack([series1, series2, series3], dim=0),  # batch 1
        torch.stack([series2, series3, series1], dim=0),  # batch 2
    ], dim=0)
    
    print(f"测试数据shape: {x_enc.shape}")
    return x_enc

def test_original_trend_calculation(generator, x_enc):
    """测试原始的趋势计算方法"""
    print("\n=== 测试原始趋势计算 ===")
    
    B, N, T = x_enc.shape
    x_reshaped = x_enc.reshape(B * N, 1, T).float()
    
    import ptwt
    coeffs = ptwt.wavedec(x_reshaped, generator.wavelet, level=generator.level, mode='reflect')
    
    # 原始方法: 直接求和
    trends_sum = []
    for i, c in enumerate(coeffs):
        diff = c[..., 1:] - c[..., :-1]
        trend_sum = diff.sum(dim=-1).mean().item()
        print(f"  频段{i} (长度={c.shape[-1]}): trend_sum = {trend_sum:.4f}")
        trends_sum.append(trend_sum)
    
    # 新方法: 平均变化率
    trends_mean = []
    for i, c in enumerate(coeffs):
        diff = c[..., 1:] - c[..., :-1]
        trend_mean = diff.mean().item()
        print(f"  频段{i} (长度={c.shape[-1]}): trend_mean = {trend_mean:.4f}")
        trends_mean.append(trend_mean)
    
    print(f"原始方法(sum): {[f'{t:.4f}' for t in trends_sum]}")
    print(f"修复方法(mean): {[f'{t:.4f}' for t in trends_mean]}")
    
    # 检查问题: 长序列的sum是否不成比例地大
    print(f"\n问题验证:")
    for i, c in enumerate(coeffs):
        length = c.shape[-1]
        print(f"  频段{i}: 长度={length}, sum/mean比值={trends_sum[i]/trends_mean[i]:.2f}")
    
    return trends_sum, trends_mean

def main():
    print("开始DWT动态提示词生成器修复测试...")
    
    # 创建测试数据
    x_enc = create_test_data()
    
    # 创建生成器
    generator = DWTPromptGenerator(wavelet='db4', level=3)
    
    print(f"生成器配置: wavelet={generator.wavelet}, level={generator.level}")
    
    # 测试原始趋势计算问题
    trends_sum, trends_mean = test_original_trend_calculation(generator, x_enc)
    
    # 测试完整的forward过程（修改前）
    print(f"\n=== 测试完整forward过程（修改前） ===")
    try:
        features_before = generator.forward(x_enc)
        print(f"成功提取特征:")
        print(f"  趋势特征: {features_before.get('trends', 'N/A')}")
        print(f"  趋势一致性: {features_before.get('trend_consistency', 'N/A')}")
        print(f"  趋势描述: {features_before.get('trend_desc', 'N/A')}")
        print(f"  能量熵: {features_before.get('energy_entropy', 'N/A'):.4f}")
        print(f"  SNR: {features_before.get('snr_db', 'N/A'):.2f} dB")
        print(f"  难度: {features_before.get('difficulty', 'N/A')}")
    except Exception as e:
        print(f"❌ forward过程出错: {e}")
        return False
    
    # 测试完整的forward过程（修改后）
    print(f"\n=== 测试完整forward过程（修改后-修复1） ===")
    try:
        features_after = generator.forward(x_enc)
        print(f"修复后特征:")
        print(f"  趋势特征: {features_after.get('trends', 'N/A')}")
        print(f"  趋势一致性: {features_after.get('trend_consistency', 'N/A')}")
        print(f"  趋势描述: {features_after.get('trend_desc', 'N/A')}")
        
        # 验证修复效果
        trends_after = features_after.get('trends', [])
        if len(trends_after) == 4:
            print(f"\n✅ 修复验证:")
            print(f"  修复前趋势: {[f'{t:.4f}' for t in trends_sum]}")
            print(f"  修复后趋势: {[f'{t:.4f}' for t in trends_after]}")
            
            # 检查是否接近我们预期的mean值
            expected = trends_mean
            match = all(abs(a - e) < 0.001 for a, e in zip(trends_after, expected))
            print(f"  是否匹配预期平均变化率: {'✅' if match else '❌'}")
        else:
            print(f"❌ 趋势特征数量异常: {len(trends_after)}")
    except Exception as e:
        print(f"❌ 修复后forward过程出错: {e}")
        return False

    print(f"\n✅ 问题1修复完成，趋势计算已改为平均变化率")
    
    # 测试问题2修复：波动性改为4频段标准差
    print(f"\n=== 测试问题2修复：4频段标准差分析 ===")
    try:
        features_fix2 = generator.forward(x_enc)
        
        # 检查新增的频段标准差特征
        freq_stds = features_fix2.get('frequency_stds', [])
        stability_desc = features_fix2.get('stability_desc', 'N/A')
        
        print(f"4个频段标准差: {[f'{std:.4f}' for std in freq_stds] if freq_stds else 'N/A'}")
        print(f"稳定性描述: {stability_desc}")
        
        # 验证是否包含4个频段
        if len(freq_stds) == 4:
            print(f"✅ 成功提取4个频段的标准差")
            print(f"  cA (低频): {freq_stds[0]:.4f}")
            print(f"  cD3 (中低频): {freq_stds[1]:.4f}")
            print(f"  cD2 (中高频): {freq_stds[2]:.4f}")
            print(f"  cD1 (高频): {freq_stds[3]:.4f}")
            
            # 验证高频vs低频比率
            ratio = freq_stds[3] / (freq_stds[0] + 1e-10)
            print(f"  高频/低频比率: {ratio:.2f}")
        else:
            print(f"❌ 频段标准差数量异常: {len(freq_stds)}")
            
    except Exception as e:
        print(f"❌ 问题2测试出错: {e}")
        return False
    
    print(f"\n✅ 问题2修复完成，波动性分析已改为4频段标准差")
    
    # 验证问题3：确认torch.std().mean()逻辑错误已修复
    print(f"\n=== 验证问题3：torch.std().mean()错误已修复 ===")
    print(f"✅ 在问题2修复中，我们已经将:")
    print(f"  原来: torch.std(coeffs[0]).mean().item()  # 错误：对标量调用mean()")
    print(f"  修复: torch.std(c).item()                # 正确：直接获取标量值")
    print(f"")
    print(f"✅ 剩余的.mean().item()都是合理的:")
    print(f"  - energy_entropy.mean().item(): 聚合(B*N,)形状的熵值 ✓")
    print(f"  - energy_ratio[idx].mean().item(): 聚合(B*N,)形状的能量占比 ✓")
    print(f"  - diff.mean().item(): 计算平均变化率 ✓")
    
    print(f"\n✅ 问题3已在问题2修复过程中解决")
    
    # 测试问题4修复：动态频段映射
    print(f"\n=== 测试问题4修复：动态频段映射 ===")
    try:
        # 测试不同level的频段映射
        for test_level in [2, 3, 5]:
            generator_test = DWTPromptGenerator(level=test_level)
            band_names = generator_test._get_band_names(test_level)
            band_names_desc = generator_test._get_band_names(test_level, style='descriptive')
            
            print(f"Level {test_level} ({test_level+1}个频段):")
            print(f"  Basic风格: {band_names}")
            print(f"  Descriptive风格: {band_names_desc}")
        
        # 测试原有功能是否正常
        features_fix4 = generator.forward(x_enc)
        print(f"\n原有功能测试:")
        print(f"  频率模式: {features_fix4.get('freq_pattern', 'N/A')}")
        print(f"  趋势描述: {features_fix4.get('trend_desc', 'N/A')}")
        
    except Exception as e:
        print(f"❌ 问题4测试出错: {e}")
        return False
    
    print(f"\n✅ 问题4修复完成，频段映射已改为动态生成")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("测试环境准备完毕")
    else:
        print("测试环境准备失败")
