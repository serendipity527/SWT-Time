"""
TimeLLM + DWT Prompt 集成测试
测试DWT Prompt生成器与TimeLLM模型的完整集成
"""

import torch
import sys
import os
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.TimeLLM import Model


class MockConfigs:
    """模拟配置对象，用于测试"""
    def __init__(self, use_dwt_prompt=False, prompt_compression='balanced'):
        # 基础配置
        self.task_name = 'long_term_forecast'
        self.pred_len = 96
        self.seq_len = 512
        self.label_len = 48
        self.d_model = 16
        self.d_ff = 32
        self.n_heads = 8
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.dropout = 0.1
        self.llm_layers = 6
        self.llm_dim = 768
        self.patch_len = 16
        self.stride = 8
        
        # SWT配置
        self.use_swt = False
        self.swt_wavelet = 'db4'
        self.swt_level = 3
        self.use_all_coeffs = True
        
        # DWT Prompt配置
        self.use_dwt_prompt = use_dwt_prompt
        self.dwt_prompt_level = 3
        self.prompt_compression = prompt_compression
        
        # LLM配置
        self.llm_model = 'GPT2'  # 使用GPT2更快
        
        # Prompt配置
        self.prompt_domain = False
        self.content = 'Test dataset'


def test_model_initialization():
    """测试1: 模型初始化"""
    print("=" * 80)
    print("测试1: TimeLLM模型初始化测试")
    print("=" * 80)
    
    # 测试原版配置
    print("\n[1.1] 测试原版配置（use_dwt_prompt=False）")
    configs_baseline = MockConfigs(use_dwt_prompt=False)
    try:
        model_baseline = Model(configs_baseline)
        assert model_baseline.dwt_prompt_generator is None
        print("✅ 原版配置初始化成功，dwt_prompt_generator=None")
    except Exception as e:
        print(f"❌ 原版配置初始化失败: {e}")
        raise
    
    # 测试DWT配置
    print("\n[1.2] 测试DWT配置（use_dwt_prompt=True）")
    configs_dwt = MockConfigs(use_dwt_prompt=True, prompt_compression='balanced')
    try:
        model_dwt = Model(configs_dwt)
        assert model_dwt.dwt_prompt_generator is not None
        assert model_dwt.use_dwt_prompt == True
        assert model_dwt.prompt_compression == 'balanced'
        print("✅ DWT配置初始化成功，dwt_prompt_generator已创建")
    except Exception as e:
        print(f"❌ DWT配置初始化失败: {e}")
        raise
    
    # 测试不同压缩级别
    print("\n[1.3] 测试不同压缩级别")
    for compression in ['minimal', 'balanced', 'detailed']:
        configs = MockConfigs(use_dwt_prompt=True, prompt_compression=compression)
        model = Model(configs)
        assert model.prompt_compression == compression
        print(f"✅ {compression} 级别初始化成功")
    
    print("\n✅ 模型初始化测试通过!")


def test_forward_pass():
    """测试2: Forward Pass"""
    print("\n" + "=" * 80)
    print("测试2: Forward Pass测试")
    print("=" * 80)
    
    # 准备测试数据
    batch_size = 2
    seq_len = 512
    pred_len = 96
    label_len = 48
    enc_in = 7
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.zeros(batch_size, seq_len, 4)  # 时间特征
    x_dec = torch.randn(batch_size, label_len + pred_len, enc_in)
    x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4)
    
    print(f"\n输入数据形状:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    
    # 测试原版forward
    print("\n[2.1] 测试原版Forward Pass")
    configs_baseline = MockConfigs(use_dwt_prompt=False)
    model_baseline = Model(configs_baseline)
    model_baseline.eval()
    
    try:
        with torch.no_grad():
            output_baseline = model_baseline(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"✅ 原版Forward通过，输出形状: {output_baseline.shape}")
        assert output_baseline.shape == (batch_size, pred_len, enc_in)
    except Exception as e:
        print(f"❌ 原版Forward失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 测试DWT Forward
    print("\n[2.2] 测试DWT Forward Pass")
    configs_dwt = MockConfigs(use_dwt_prompt=True, prompt_compression='balanced')
    model_dwt = Model(configs_dwt)
    model_dwt.eval()
    
    try:
        with torch.no_grad():
            output_dwt = model_dwt(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"✅ DWT Forward通过，输出形状: {output_dwt.shape}")
        assert output_dwt.shape == (batch_size, pred_len, enc_in)
    except Exception as e:
        print(f"❌ DWT Forward失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n✅ Forward Pass测试通过!")


def test_prompt_generation():
    """测试3: Prompt生成对比"""
    print("\n" + "=" * 80)
    print("测试3: Prompt生成对比测试")
    print("=" * 80)
    
    # 准备数据
    batch_size = 2
    seq_len = 512
    enc_in = 7
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    
    # 创建模型
    configs_baseline = MockConfigs(use_dwt_prompt=False)
    configs_dwt = MockConfigs(use_dwt_prompt=True, prompt_compression='balanced')
    
    model_baseline = Model(configs_baseline)
    model_dwt = Model(configs_dwt)
    
    print("\n[3.1] 原版Prompt生成")
    # 手动调用forecast方法的prompt生成部分来捕获prompt
    model_baseline.eval()
    
    # 模拟forecast中的prompt生成逻辑
    x_enc_normalized = model_baseline.normalize_layers(x_enc, 'norm')
    B, T, N = x_enc_normalized.size()
    x_enc_reshaped = x_enc_normalized.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
    
    min_values = torch.min(x_enc_reshaped, dim=1)[0]
    max_values = torch.max(x_enc_reshaped, dim=1)[0]
    medians = torch.median(x_enc_reshaped, dim=1).values
    lags = model_baseline.calcute_lags(x_enc_reshaped)
    trends = x_enc_reshaped.diff(dim=1).sum(dim=1)
    
    # 生成原版prompt（只取第一个样本）
    prompt_baseline = (
        f"<|start_prompt|>Dataset description: {model_baseline.description}"
        f"Task description: forecast the next {model_baseline.pred_len} steps given the previous {model_baseline.seq_len} steps information; "
        f"Input statistics: min value {min_values[0].item():.4f}, max value {max_values[0].item():.4f}, "
        f"median value {medians[0].item():.4f}, the trend of input is {'upward' if trends[0] > 0 else 'downward'}, "
        f"top 5 lags are : {lags[0].tolist()}<|<end_prompt>|>"
    )
    
    print(f"原版Prompt (样本0):")
    print(prompt_baseline)
    print(f"Token数估计: {len(prompt_baseline.split())}")
    
    print("\n[3.2] DWT Prompt生成")
    # 使用DWT生成器
    x_sample = x_enc_normalized[0:1, :, :].permute(0, 2, 1)  # (1, N, T)
    dwt_features = model_dwt.dwt_prompt_generator(x_sample)
    
    base_info = {
        'min': min_values[0].item(),
        'max': max_values[0].item(),
        'median': medians[0].item(),
        'lags': lags[0].cpu().numpy(),
        'description': model_dwt.description,
        'seq_len': model_dwt.seq_len,
        'pred_len': model_dwt.pred_len
    }
    
    prompt_dwt = model_dwt.dwt_prompt_generator.build_prompt_text(dwt_features, base_info)
    
    print(f"DWT Prompt (样本0, {model_dwt.prompt_compression}模式):")
    print(prompt_dwt)
    print(f"Token数估计: {len(prompt_dwt.split())}")
    
    print("\n[3.3] 对比分析")
    print(f"原版Token数: {len(prompt_baseline.split())}")
    print(f"DWT Token数: {len(prompt_dwt.split())}")
    print(f"Token增加: +{len(prompt_dwt.split()) - len(prompt_baseline.split())} "
          f"({(len(prompt_dwt.split()) - len(prompt_baseline.split())) / len(prompt_baseline.split()) * 100:.1f}%)")
    
    print("\nDWT新增信息:")
    print(f"  - 频域模式: {dwt_features['freq_pattern']}")
    print(f"  - 趋势细化: {dwt_features['trend_desc']}")
    print(f"  - 信号质量: {dwt_features['signal_quality']} (SNR: {dwt_features['snr_db']:.1f} dB)")
    print(f"  - 预测难度: {dwt_features['difficulty']}")
    
    print("\n✅ Prompt生成对比测试通过!")


def test_batch_compatibility():
    """测试4: 批处理兼容性"""
    print("\n" + "=" * 80)
    print("测试4: 批处理兼容性测试")
    print("=" * 80)
    
    configs_dwt = MockConfigs(use_dwt_prompt=True, prompt_compression='balanced')
    model_dwt = Model(configs_dwt)
    model_dwt.eval()
    
    # 测试不同batch size
    batch_sizes = [1, 2, 4, 8]
    seq_len = 512
    pred_len = 96
    label_len = 48
    enc_in = 7
    
    print(f"\n测试不同Batch Size:")
    for batch_size in batch_sizes:
        x_enc = torch.randn(batch_size, seq_len, enc_in)
        x_mark_enc = torch.zeros(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, label_len + pred_len, enc_in)
        x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4)
        
        try:
            with torch.no_grad():
                output = model_dwt(x_enc, x_mark_enc, x_dec, x_mark_dec)
            assert output.shape == (batch_size, pred_len, enc_in)
            print(f"✅ Batch={batch_size}: 输出形状 {output.shape}")
        except Exception as e:
            print(f"❌ Batch={batch_size}: 失败 - {e}")
            raise
    
    print("\n✅ 批处理兼容性测试通过!")


def test_compression_levels():
    """测试5: 不同压缩级别对比"""
    print("\n" + "=" * 80)
    print("测试5: 压缩级别对比测试")
    print("=" * 80)
    
    batch_size = 2
    seq_len = 512
    pred_len = 96
    label_len = 48
    enc_in = 7
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.zeros(batch_size, seq_len, 4)
    x_dec = torch.randn(batch_size, label_len + pred_len, enc_in)
    x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4)
    
    import time
    
    results = {}
    
    for compression in ['minimal', 'balanced', 'detailed']:
        print(f"\n[5.{['minimal', 'balanced', 'detailed'].index(compression)+1}] 测试{compression}模式")
        
        configs = MockConfigs(use_dwt_prompt=True, prompt_compression=compression)
        model = Model(configs)
        model.eval()
        
        # 预热
        with torch.no_grad():
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 测试性能
        start = time.time()
        with torch.no_grad():
            for _ in range(5):
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elapsed = (time.time() - start) / 5 * 1000  # ms
        
        # 生成一个prompt看token数
        x_enc_norm = model.normalize_layers(x_enc, 'norm')
        x_sample = x_enc_norm[0:1, :, :].permute(0, 2, 1)
        dwt_features = model.dwt_prompt_generator(x_sample)
        
        B, T, N = x_enc_norm.size()
        x_reshaped = x_enc_norm.permute(0, 2, 1).reshape(B * N, T, 1)
        min_val = torch.min(x_reshaped, dim=1)[0][0].item()
        max_val = torch.max(x_reshaped, dim=1)[0][0].item()
        median = torch.median(x_reshaped, dim=1).values[0].item()
        lags = model.calcute_lags(x_reshaped)[0].cpu().numpy()
        
        base_info = {
            'min': min_val, 'max': max_val, 'median': median, 'lags': lags,
            'description': model.description, 'seq_len': model.seq_len, 'pred_len': model.pred_len
        }
        prompt = model.dwt_prompt_generator.build_prompt_text(dwt_features, base_info)
        token_count = len(prompt.split())
        
        results[compression] = {
            'time': elapsed,
            'tokens': token_count,
            'output_shape': output.shape
        }
        
        print(f"  ✅ 输出形状: {output.shape}")
        print(f"  ✅ 推理时间: {elapsed:.2f} ms")
        print(f"  ✅ Token数: {token_count}")
    
    print("\n[5.4] 性能对比总结")
    print(f"{'模式':<12} {'Token数':<12} {'推理时间(ms)':<15}")
    print("-" * 40)
    for mode, result in results.items():
        print(f"{mode:<12} {result['tokens']:<12} {result['time']:<15.2f}")
    
    print("\n✅ 压缩级别对比测试通过!")


def test_gpu_compatibility():
    """测试6: GPU兼容性"""
    print("\n" + "=" * 80)
    print("测试6: GPU兼容性测试")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU测试")
        return
    
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    
    configs_dwt = MockConfigs(use_dwt_prompt=True, prompt_compression='balanced')
    model_dwt = Model(configs_dwt).cuda()
    model_dwt.eval()
    
    batch_size = 2
    seq_len = 512
    pred_len = 96
    label_len = 48
    enc_in = 7
    
    x_enc = torch.randn(batch_size, seq_len, enc_in).cuda()
    x_mark_enc = torch.zeros(batch_size, seq_len, 4).cuda()
    x_dec = torch.randn(batch_size, label_len + pred_len, enc_in).cuda()
    x_mark_dec = torch.zeros(batch_size, label_len + pred_len, 4).cuda()
    
    print(f"\n输入设备: {x_enc.device}")
    
    try:
        with torch.no_grad():
            output = model_dwt(x_enc, x_mark_enc, x_dec, x_mark_dec)
        print(f"✅ GPU Forward通过，输出设备: {output.device}")
        print(f"✅ 输出形状: {output.shape}")
        assert output.device.type == 'cuda'
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n✅ GPU兼容性测试通过!")


if __name__ == "__main__":
    print("=" * 80)
    print("TimeLLM + DWT Prompt 集成测试套件")
    print("=" * 80)
    print("\n本测试验证DWT Prompt生成器与TimeLLM模型的完整集成")
    print("包括：模型初始化、Forward Pass、Prompt生成、批处理、压缩级别、GPU兼容性")
    print("=" * 80)
    
    try:
        test_model_initialization()
        test_forward_pass()
        test_prompt_generation()
        test_batch_compatibility()
        test_compression_levels()
        test_gpu_compatibility()
        
        print("\n" + "=" * 80)
        print("🎉 所有集成测试通过!")
        print("=" * 80)
        print("\n✅ DWT Prompt已成功集成到TimeLLM模型")
        print("✅ 可以开始使用 --use_dwt_prompt True 进行训练")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 集成测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
