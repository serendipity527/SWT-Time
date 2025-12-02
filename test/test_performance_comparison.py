"""
性能对比测试：优化前后的DWTPromptGenerator
"""
import torch
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.DWTPromptGenerator import DWTPromptGenerator as OriginalGenerator
from layers.DWTPromptGenerator_performance_up import DWTPromptGenerator as OptimizedGenerator


def benchmark_generator(generator_class, x_enc, num_runs=100, warmup=10):
    """
    性能基准测试
    
    Args:
        generator_class: 生成器类
        x_enc: 输入数据
        num_runs: 运行次数
        warmup: 预热次数
    
    Returns:
        avg_time: 平均时间(ms)
        std_time: 标准差(ms)
    """
    generator = generator_class(wavelet='db4', level=3, compression_level='balanced')
    generator.eval()
    
    # 确定设备
    device = x_enc.device
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = generator(x_enc)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 基准测试
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = time.perf_counter()
            else:
                start = time.perf_counter()
            
            features = generator(x_enc)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                end = time.perf_counter()
            
            times.append((end - start) * 1000)  # 转换为毫秒
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return avg_time, std_time, features


def test_correctness(x_enc):
    """验证优化前后结果的一致性"""
    print("\n" + "="*60)
    print("正确性验证")
    print("="*60)
    
    original_gen = OriginalGenerator(wavelet='db4', level=3, compression_level='balanced')
    optimized_gen = OptimizedGenerator(wavelet='db4', level=3, compression_level='balanced')
    
    with torch.no_grad():
        original_features = original_gen(x_enc)
        optimized_features = optimized_gen(x_enc)
    
    # 比较关键特征
    keys_to_check = [
        'energy_entropy', 'dominant_band', 'dominant_energy',
        'trend_consistency', 'snr_db', 'difficulty',
        'freq_pattern', 'trend_desc', 'signal_quality'
    ]
    
    all_match = True
    for key in keys_to_check:
        orig_val = original_features.get(key)
        opt_val = optimized_features.get(key)
        
        if isinstance(orig_val, (int, float)):
            # 数值比较（允许小误差）
            if abs(orig_val - opt_val) < 1e-5:
                status = "✓"
            else:
                status = "✗"
                all_match = False
            print(f"{key:25s}: 原始={orig_val:12.6f} | 优化={opt_val:12.6f} [{status}]")
        else:
            # 字符串或其他类型比较
            match = orig_val == opt_val
            status = "✓" if match else "✗"
            if not match:
                all_match = False
            print(f"{key:25s}: 原始={orig_val!r:30s} | 优化={opt_val!r:30s} [{status}]")
    
    if all_match:
        print("\n✓ 所有特征匹配！优化保持了结果一致性")
    else:
        print("\n✗ 部分特征不匹配，请检查优化逻辑")
    
    return all_match


def main():
    print("="*60)
    print("DWTPromptGenerator 性能对比测试")
    print("="*60)
    
    # 测试配置
    batch_sizes = [1, 4, 16]
    seq_lengths = [96, 336, 720]
    n_features = 7
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 为每个配置运行测试
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\n{'='*60}")
            print(f"测试配置: Batch={batch_size}, SeqLen={seq_len}, Features={n_features}")
            print(f"{'='*60}")
            
            # 生成测试数据
            x_enc = torch.randn(batch_size, n_features, seq_len).to(device)
            
            # 测试原始版本
            print("\n[1] 原始版本测试...")
            try:
                orig_time, orig_std, _ = benchmark_generator(
                    OriginalGenerator, x_enc, num_runs=50, warmup=5
                )
                print(f"   平均时间: {orig_time:.3f} ± {orig_std:.3f} ms")
            except Exception as e:
                print(f"   错误: {e}")
                orig_time = float('inf')
                orig_std = 0
            
            # 测试优化版本
            print("\n[2] 优化版本测试...")
            try:
                opt_time, opt_std, _ = benchmark_generator(
                    OptimizedGenerator, x_enc, num_runs=50, warmup=5
                )
                print(f"   平均时间: {opt_time:.3f} ± {opt_std:.3f} ms")
            except Exception as e:
                print(f"   错误: {e}")
                opt_time = float('inf')
                opt_std = 0
            
            # 计算加速比
            if orig_time < float('inf') and opt_time < float('inf'):
                speedup = orig_time / opt_time
                improvement = (orig_time - opt_time) / orig_time * 100
                
                print(f"\n[性能提升]")
                print(f"   加速比: {speedup:.2f}x")
                print(f"   性能提升: {improvement:.1f}%")
                print(f"   时间节省: {orig_time - opt_time:.3f} ms")
                
                results.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'orig_time': orig_time,
                    'opt_time': opt_time,
                    'speedup': speedup,
                    'improvement': improvement
                })
    
    # 打印汇总
    if results:
        print("\n" + "="*60)
        print("性能汇总")
        print("="*60)
        print(f"{'Batch':>6} | {'SeqLen':>6} | {'原始(ms)':>10} | {'优化(ms)':>10} | {'加速比':>8} | {'提升%':>8}")
        print("-"*60)
        for r in results:
            print(f"{r['batch_size']:>6} | {r['seq_len']:>6} | "
                  f"{r['orig_time']:>10.3f} | {r['opt_time']:>10.3f} | "
                  f"{r['speedup']:>8.2f}x | {r['improvement']:>7.1f}%")
        
        # 平均性能提升
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        print("-"*60)
        print(f"{'平均':>6} | {'':>6} | {'':>10} | {'':>10} | "
              f"{avg_speedup:>8.2f}x | {avg_improvement:>7.1f}%")
    
    # 正确性验证
    x_test = torch.randn(2, 7, 96).to(device)
    test_correctness(x_test)
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()
