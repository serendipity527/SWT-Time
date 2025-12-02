"""
测试DWT Prompt生成器

注意: 使用 DWTPromptGenerator_performance_up 进行测试
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.DWTPromptGenerator_performance_up import DWTPromptGenerator


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 80)
    print("测试1: 基本功能测试")
    print("=" * 80)
    
    # 创建生成器
    generator = DWTPromptGenerator(
        wavelet='db4',
        level=3,
        compression_level='balanced'
    )
    
    # 创建测试数据
    batch_size = 2
    n_vars = 7
    seq_len = 512
    
    x_enc = torch.randn(batch_size, n_vars, seq_len)
    
    print(f"输入形状: {x_enc.shape}")
    
    # 提取特征
    features = generator(x_enc)
    
    print("\n提取的特征:")
    print(f"  - 频域模式: {features['freq_pattern']}")
    print(f"  - 能量熵: {features['energy_entropy']:.4f}")
    print(f"  - 能量分布: {features['energy_ratio']}")
    print(f"  - 主导频段: {features['dominant_band']} (能量: {features['dominant_energy']:.2%})")
    print(f"  - 趋势描述: {features['trend_desc']}")
    print(f"  - 趋势一致性: {features['trend_consistency']:.2f}")
    print(f"  - 信号质量: {features['signal_quality']}")
    print(f"  - SNR: {features['snr_db']:.2f} dB")
    print(f"  - 预测难度: {features['difficulty']}")
    
    print("\n✅ 基本功能测试通过!")


def test_prompt_generation():
    """测试prompt生成"""
    print("\n" + "=" * 80)
    print("测试2: Prompt生成测试")
    print("=" * 80)
    
    # 测试三种压缩级别
    compression_levels = ['minimal', 'balanced', 'detailed']
    
    x_enc = torch.randn(1, 7, 512)
    
    # 模拟lags
    lags = torch.tensor([24, 48, 96, 168, 336])
    
    base_info = {
        'min': -1.234,
        'max': 2.567,
        'median': 0.345,
        'lags': lags.numpy(),
        'description': 'The Electricity Transformer Temperature (ETT) dataset',
        'seq_len': 512,
        'pred_len': 96
    }
    
    for compression in compression_levels:
        print(f"\n{'='*60}")
        print(f"压缩级别: {compression.upper()}")
        print(f"{'='*60}")
        
        generator = DWTPromptGenerator(
            wavelet='db4',
            level=3,
            compression_level=compression
        )
        
        features = generator(x_enc)
        prompt_text = generator.build_prompt_text(features, base_info)
        
        print(prompt_text)
        
        # 统计token数（粗略估计，按空格分割）
        token_count = len(prompt_text.split())
        print(f"\n估计Token数: ~{token_count}")
    
    print("\n✅ Prompt生成测试通过!")


def test_different_patterns():
    """测试不同模式的序列"""
    print("\n" + "=" * 80)
    print("测试3: 不同模式序列测试")
    print("=" * 80)
    
    generator = DWTPromptGenerator(compression_level='balanced')
    
    # 测试场景
    test_cases = [
        ("平稳趋势", torch.randn(1, 7, 512) * 0.1 + torch.linspace(0, 10, 512).view(1, 1, 512)),
        ("强周期性", torch.sin(torch.linspace(0, 20*3.14159, 512)).repeat(1, 7, 1) + torch.randn(1, 7, 512) * 0.1),
        ("高噪声", torch.randn(1, 7, 512) * 2),
        ("多尺度混合", torch.sin(torch.linspace(0, 10*3.14159, 512)).repeat(1, 7, 1) + 
                      torch.sin(torch.linspace(0, 50*3.14159, 512)).repeat(1, 7, 1) * 0.5 +
                      torch.randn(1, 7, 512) * 0.3)
    ]
    
    for name, x_enc in test_cases:
        print(f"\n{'-'*60}")
        print(f"场景: {name}")
        print(f"{'-'*60}")
        
        features = generator(x_enc)
        
        print(f"频域模式: {features['freq_pattern']}")
        print(f"趋势描述: {features['trend_desc']}")
        print(f"信号质量: {features['signal_quality']} (SNR: {features['snr_db']:.1f} dB)")
        print(f"预测难度: {features['difficulty']}")
        # 能量分布是numpy数组，需要先转换为标量
        energy = features['energy_ratio']
        print(f"能量分布: cA={energy[0].item():.1%}, cD3={energy[1].item():.1%}, "
              f"cD2={energy[2].item():.1%}, cD1={energy[3].item():.1%}")
    
    print("\n✅ 不同模式测试通过!")


def test_batch_processing():
    """测试批处理性能"""
    print("\n" + "=" * 80)
    print("测试4: 批处理性能测试")
    print("=" * 80)
    
    import time
    
    generator = DWTPromptGenerator(compression_level='balanced')
    
    batch_sizes = [1, 4, 8, 16, 32]
    n_vars = 7
    seq_len = 512
    
    print(f"\n序列长度: {seq_len}, 变量数: {n_vars}")
    print(f"{'Batch Size':<12} {'时间(ms)':<12} {'样本/秒':<12}")
    print("-" * 40)
    
    for batch_size in batch_sizes:
        x_enc = torch.randn(batch_size, n_vars, seq_len)
        
        # 预热
        _ = generator(x_enc)
        
        # 测试
        start = time.time()
        for _ in range(10):
            _ = generator(x_enc)
        elapsed = (time.time() - start) / 10 * 1000  # ms
        
        throughput = batch_size / (elapsed / 1000)
        
        print(f"{batch_size:<12} {elapsed:<12.2f} {throughput:<12.1f}")
    
    print("\n✅ 批处理性能测试通过!")


def test_gpu_compatibility():
    """测试GPU兼容性"""
    print("\n" + "=" * 80)
    print("测试5: GPU兼容性测试")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU测试")
        return
    
    generator = DWTPromptGenerator(compression_level='balanced').cuda()
    
    x_enc = torch.randn(4, 7, 512).cuda()
    
    print(f"输入设备: {x_enc.device}")
    
    features = generator(x_enc)
    
    print(f"特征提取成功!")
    print(f"  - 频域模式: {features['freq_pattern']}")
    print(f"  - SNR: {features['snr_db']:.2f} dB")
    
    print("\n✅ GPU兼容性测试通过!")


if __name__ == "__main__":
    print("DWT Prompt Generator 测试套件")
    print("=" * 80)
    
    try:
        test_basic_functionality()
        test_prompt_generation()
        test_different_patterns()
        test_batch_processing()
        test_gpu_compatibility()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
