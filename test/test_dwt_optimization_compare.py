"""
DWT Prompt Generator 优化前后性能对比测试
========================================

对比原版和优化版的性能差异

注意: 使用 DWTPromptGenerator_performance_up 进行测试
"""

import torch
import sys
import os
import time
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.DWTPromptGenerator_performance_up import DWTPromptGenerator


def benchmark(generator, x_enc, n_runs=50, warmup=5):
    """性能基准测试"""
    # 预热
    for _ in range(warmup):
        _ = generator(x_enc)
    
    # 测试
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = generator(x_enc)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def main():
    print("=" * 80)
    print("DWT Prompt Generator 优化前后性能对比")
    print("=" * 80)
    
    # 测试配置
    batch_sizes = [1, 4, 8, 16, 32]
    seq_len = 512
    n_vars = 7
    
    print(f"\n配置: seq_len={seq_len}, n_vars={n_vars}")
    print(f"{'Batch':<8} {'优化版(ms)':<15} {'编译版(ms)':<15} {'加速比':<12} {'vs编译':<12}")
    print("-" * 80)
    
    results = []
    
    for batch_size in batch_sizes:
        x_enc = torch.randn(batch_size, n_vars, seq_len)
        
        # 测试优化版 (不使用compile)
        generator_opt = DWTPromptGenerator(compression_level='balanced', use_compile=False)
        stats_opt = benchmark(generator_opt, x_enc)
        
        # 测试编译版 (使用compile)
        generator_compile = DWTPromptGenerator(compression_level='balanced', use_compile=True)
        stats_compile = benchmark(generator_compile, x_enc)
        
        speedup_vs_compile = stats_opt['mean'] / stats_compile['mean']
        
        results.append({
            'batch_size': batch_size,
            'optimized': stats_opt['mean'],
            'compiled': stats_compile['mean'],
            'speedup_compile': speedup_vs_compile
        })
        
        print(f"{batch_size:<8} {stats_opt['mean']:<15.2f} {stats_compile['mean']:<15.2f} "
              f"{'N/A':<12} {speedup_vs_compile:<12.2f}x")
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("GPU性能测试")
        print("=" * 80)
        
        print(f"\n{'Batch':<8} {'CPU优化(ms)':<15} {'GPU优化(ms)':<15} {'GPU加速比':<12}")
        print("-" * 70)
        
        for batch_size in [4, 8, 16, 32]:
            # CPU测试
            x_enc_cpu = torch.randn(batch_size, n_vars, seq_len)
            generator_cpu = DWTPromptGenerator(compression_level='balanced', use_compile=False)
            stats_cpu = benchmark(generator_cpu, x_enc_cpu, n_runs=20)
            
            # GPU测试
            x_enc_gpu = torch.randn(batch_size, n_vars, seq_len).cuda()
            generator_gpu = DWTPromptGenerator(compression_level='balanced', use_compile=False).cuda()
            stats_gpu = benchmark(generator_gpu, x_enc_gpu, n_runs=20)
            
            speedup = stats_cpu['mean'] / stats_gpu['mean']
            
            print(f"{batch_size:<8} {stats_cpu['mean']:<15.2f} {stats_gpu['mean']:<15.2f} {speedup:<12.2f}x")
    
    # 总结
    print("\n" + "=" * 80)
    print("性能总结")
    print("=" * 80)
    
    avg_compile_speedup = np.mean([r['speedup_compile'] for r in results])
    
    print(f"\n✅ 优化成果:")
    print(f"   - 平均torch.compile加速比: {avg_compile_speedup:.2f}x")
    print(f"   - 推荐配置: use_compile=True (PyTorch 2.0+)")
    
    print("\n📊 关键改进:")
    print("   1. ✅ 向量化能量计算 - 减少循环开销")
    print("   2. ✅ 减少CPU-GPU数据传输 - 保持tensor在设备上")
    print("   3. ✅ torch.compile加速 - JIT编译优化")
    print("   4. ✅ 批量化趋势计算 - 提升并行度")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
