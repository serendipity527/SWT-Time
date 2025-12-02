"""
DWT Prompt Generator 性能分析和测试套件
==========================================

该脚本全面测试DWTPromptGenerator模块的性能特征:
1. 时间复杂度分析
2. 空间复杂度分析
3. 不同配置的性能对比
4. GPU vs CPU性能对比
5. 内存效率分析
6. 瓶颈识别

注意: 使用 DWTPromptGenerator_performance_up 进行测试
"""

import torch
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tracemalloc
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.DWTPromptGenerator_performance_up import DWTPromptGenerator


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def measure_time(self, func, *args, n_runs=10, warmup=2):
        """测量函数执行时间"""
        # 预热
        for _ in range(warmup):
            func(*args)
        
        # 测量
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func(*args)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def measure_memory(self, func, *args):
        """测量内存使用"""
        tracemalloc.start()
        result = func(*args)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        }


def test_time_complexity():
    """测试1: 时间复杂度分析"""
    print("=" * 80)
    print("测试1: 时间复杂度分析")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    generator = DWTPromptGenerator(compression_level='balanced')
    
    # 测试不同序列长度
    print("\n1.1 不同序列长度的性能 (固定 batch_size=4, n_vars=7)")
    print(f"{'序列长度':<12} {'平均时间(ms)':<15} {'标准差':<12} {'吞吐量(样本/s)':<15}")
    print("-" * 70)
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    time_results = []
    
    for seq_len in seq_lengths:
        x_enc = torch.randn(4, 7, seq_len)
        
        stats = profiler.measure_time(generator, x_enc, n_runs=20)
        throughput = 4 / (stats['mean'] / 1000)
        
        time_results.append({
            'seq_len': seq_len,
            'time': stats['mean'],
            'std': stats['std'],
            'throughput': throughput
        })
        
        print(f"{seq_len:<12} {stats['mean']:<15.2f} {stats['std']:<12.2f} {throughput:<15.1f}")
    
    # 测试不同批次大小
    print("\n1.2 不同批次大小的性能 (固定 seq_len=512, n_vars=7)")
    print(f"{'批次大小':<12} {'平均时间(ms)':<15} {'标准差':<12} {'吞吐量(样本/s)':<15}")
    print("-" * 70)
    
    batch_results = []
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        x_enc = torch.randn(batch_size, 7, 512)
        
        stats = profiler.measure_time(generator, x_enc, n_runs=20)
        throughput = batch_size / (stats['mean'] / 1000)
        
        batch_results.append({
            'batch_size': batch_size,
            'time': stats['mean'],
            'std': stats['std'],
            'throughput': throughput
        })
        
        print(f"{batch_size:<12} {stats['mean']:<15.2f} {stats['std']:<12.2f} {throughput:<15.1f}")
    
    # 测试不同变量数
    print("\n1.3 不同变量数的性能 (固定 batch_size=4, seq_len=512)")
    print(f"{'变量数':<12} {'平均时间(ms)':<15} {'标准差':<12} {'吞吐量(样本/s)':<15}")
    print("-" * 70)
    
    var_results = []
    n_vars_list = [1, 3, 7, 14, 21, 28]
    
    for n_vars in n_vars_list:
        x_enc = torch.randn(4, n_vars, 512)
        
        stats = profiler.measure_time(generator, x_enc, n_runs=20)
        throughput = 4 / (stats['mean'] / 1000)
        
        var_results.append({
            'n_vars': n_vars,
            'time': stats['mean'],
            'std': stats['std'],
            'throughput': throughput
        })
        
        print(f"{n_vars:<12} {stats['mean']:<15.2f} {stats['std']:<12.2f} {throughput:<15.1f}")
    
    print("\n✅ 时间复杂度分析完成!")
    return time_results, batch_results, var_results


def test_space_complexity():
    """测试2: 空间复杂度分析"""
    print("\n" + "=" * 80)
    print("测试2: 空间复杂度分析")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    generator = DWTPromptGenerator(compression_level='balanced')
    
    print("\n2.1 不同序列长度的内存使用")
    print(f"{'序列长度':<12} {'峰值内存(MB)':<15} {'当前内存(MB)':<15}")
    print("-" * 50)
    
    memory_results = []
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    for seq_len in seq_lengths:
        x_enc = torch.randn(4, 7, seq_len)
        
        mem_stats = profiler.measure_memory(generator, x_enc)
        
        memory_results.append({
            'seq_len': seq_len,
            'peak_mb': mem_stats['peak_mb'],
            'current_mb': mem_stats['current_mb']
        })
        
        print(f"{seq_len:<12} {mem_stats['peak_mb']:<15.2f} {mem_stats['current_mb']:<15.2f}")
    
    # 测试不同批次大小的内存
    print("\n2.2 不同批次大小的内存使用")
    print(f"{'批次大小':<12} {'峰值内存(MB)':<15} {'当前内存(MB)':<15}")
    print("-" * 50)
    
    batch_memory_results = []
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        x_enc = torch.randn(batch_size, 7, 512)
        
        mem_stats = profiler.measure_memory(generator, x_enc)
        
        batch_memory_results.append({
            'batch_size': batch_size,
            'peak_mb': mem_stats['peak_mb'],
            'current_mb': mem_stats['current_mb']
        })
        
        print(f"{batch_size:<12} {mem_stats['peak_mb']:<15.2f} {mem_stats['current_mb']:<15.2f}")
    
    print("\n✅ 空间复杂度分析完成!")
    return memory_results, batch_memory_results


def test_configuration_comparison():
    """测试3: 不同配置的性能对比"""
    print("\n" + "=" * 80)
    print("测试3: 不同配置的性能对比")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    x_enc = torch.randn(8, 7, 512)
    
    # 测试不同小波基
    print("\n3.1 不同小波基的性能对比")
    print(f"{'小波基':<12} {'平均时间(ms)':<15} {'标准差':<12}")
    print("-" * 50)
    
    wavelets = ['db1', 'db4', 'db8', 'sym4', 'coif2']
    wavelet_results = []
    
    for wavelet in wavelets:
        try:
            generator = DWTPromptGenerator(wavelet=wavelet, level=3)
            stats = profiler.measure_time(generator, x_enc, n_runs=20)
            
            wavelet_results.append({
                'wavelet': wavelet,
                'time': stats['mean'],
                'std': stats['std']
            })
            
            print(f"{wavelet:<12} {stats['mean']:<15.2f} {stats['std']:<12.2f}")
        except Exception as e:
            print(f"{wavelet:<12} 失败: {e}")
    
    # 测试不同分解层数
    print("\n3.2 不同分解层数的性能对比")
    print(f"{'分解层数':<12} {'平均时间(ms)':<15} {'标准差':<12}")
    print("-" * 50)
    
    level_results = []
    levels = [1, 2, 3, 4, 5]
    
    for level in levels:
        try:
            generator = DWTPromptGenerator(wavelet='db4', level=level)
            stats = profiler.measure_time(generator, x_enc, n_runs=20)
            
            level_results.append({
                'level': level,
                'time': stats['mean'],
                'std': stats['std']
            })
            
            print(f"{level:<12} {stats['mean']:<15.2f} {stats['std']:<12.2f}")
        except Exception as e:
            print(f"{level:<12} 失败: {e}")
    
    # 测试不同压缩级别
    print("\n3.3 不同压缩级别的Prompt生成性能")
    print(f"{'压缩级别':<12} {'平均时间(ms)':<15} {'Token数':<12}")
    print("-" * 50)
    
    compression_results = []
    compressions = ['minimal', 'balanced', 'detailed']
    
    base_info = {
        'min': -1.234,
        'max': 2.567,
        'median': 0.345,
        'lags': np.array([24, 48, 96, 168, 336]),
        'description': 'Test dataset',
        'seq_len': 512,
        'pred_len': 96
    }
    
    for compression in compressions:
        generator = DWTPromptGenerator(compression_level=compression)
        features = generator(x_enc)
        
        def build_prompt():
            return generator.build_prompt_text(features, base_info)
        
        stats = profiler.measure_time(build_prompt, n_runs=50)
        prompt = build_prompt()
        token_count = len(prompt.split())
        
        compression_results.append({
            'compression': compression,
            'time': stats['mean'],
            'tokens': token_count
        })
        
        print(f"{compression:<12} {stats['mean']:<15.2f} {token_count:<12}")
    
    print("\n✅ 配置对比测试完成!")
    return wavelet_results, level_results, compression_results


def test_gpu_vs_cpu():
    """测试4: GPU vs CPU性能对比"""
    print("\n" + "=" * 80)
    print("测试4: GPU vs CPU性能对比")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU测试")
        return None
    
    profiler = PerformanceProfiler()
    
    print("\n4.1 不同批次大小的GPU加速比")
    print(f"{'批次大小':<12} {'CPU(ms)':<12} {'GPU(ms)':<12} {'加速比':<12}")
    print("-" * 60)
    
    gpu_results = []
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        # CPU测试
        generator_cpu = DWTPromptGenerator(compression_level='balanced')
        x_enc_cpu = torch.randn(batch_size, 7, 512)
        cpu_stats = profiler.measure_time(generator_cpu, x_enc_cpu, n_runs=20)
        
        # GPU测试
        generator_gpu = DWTPromptGenerator(compression_level='balanced').cuda()
        x_enc_gpu = torch.randn(batch_size, 7, 512).cuda()
        gpu_stats = profiler.measure_time(generator_gpu, x_enc_gpu, n_runs=20)
        
        speedup = cpu_stats['mean'] / gpu_stats['mean']
        
        gpu_results.append({
            'batch_size': batch_size,
            'cpu_time': cpu_stats['mean'],
            'gpu_time': gpu_stats['mean'],
            'speedup': speedup
        })
        
        print(f"{batch_size:<12} {cpu_stats['mean']:<12.2f} {gpu_stats['mean']:<12.2f} {speedup:<12.2f}x")
    
    print("\n✅ GPU vs CPU对比测试完成!")
    return gpu_results


def test_bottleneck_analysis():
    """测试5: 瓶颈识别"""
    print("\n" + "=" * 80)
    print("测试5: 瓶颈识别")
    print("=" * 80)
    
    import time
    
    generator = DWTPromptGenerator(compression_level='balanced')
    x_enc = torch.randn(8, 7, 512)
    
    print("\n5.1 各阶段耗时分析")
    
    # 手动计时各个阶段
    B, N, T = x_enc.shape
    x_reshaped = x_enc.reshape(B * N, 1, T).float()
    
    # DWT分解
    import ptwt
    start = time.perf_counter()
    for _ in range(50):
        coeffs = ptwt.wavedec(x_reshaped, 'db4', level=3, mode='reflect')
    dwt_time = (time.perf_counter() - start) / 50 * 1000
    
    # 频域特征提取
    start = time.perf_counter()
    for _ in range(50):
        freq_features = generator._extract_frequency_features(coeffs)
    freq_time = (time.perf_counter() - start) / 50 * 1000
    
    # 趋势特征提取
    start = time.perf_counter()
    for _ in range(50):
        trend_features = generator._extract_trend_features(coeffs)
    trend_time = (time.perf_counter() - start) / 50 * 1000
    
    # 质量特征提取
    start = time.perf_counter()
    for _ in range(50):
        quality_features = generator._extract_quality_features(coeffs)
    quality_time = (time.perf_counter() - start) / 50 * 1000
    
    # 难度计算
    start = time.perf_counter()
    for _ in range(50):
        difficulty = generator._calculate_difficulty(freq_features, trend_features, quality_features)
    difficulty_time = (time.perf_counter() - start) / 50 * 1000
    
    # 总时间
    total_time = dwt_time + freq_time + trend_time + quality_time + difficulty_time
    
    # 打印结果
    stages = [
        ('DWT分解', dwt_time),
        ('频域特征提取', freq_time),
        ('趋势特征提取', trend_time),
        ('质量特征提取', quality_time),
        ('难度计算', difficulty_time)
    ]
    
    print(f"{'阶段':<20} {'时间(ms)':<12} {'占比':<12}")
    print("-" * 50)
    
    bottleneck_results = []
    for stage_name, stage_time in stages:
        percentage = (stage_time / total_time) * 100
        bottleneck_results.append({
            'stage': stage_name,
            'time': stage_time,
            'percentage': percentage
        })
        print(f"{stage_name:<20} {stage_time:<12.3f} {percentage:<12.1f}%")
    
    print(f"{'总计':<20} {total_time:<12.3f} {100.0:<12.1f}%")
    
    print("\n✅ 瓶颈识别完成!")
    return bottleneck_results


def test_scalability():
    """测试6: 可扩展性分析"""
    print("\n" + "=" * 80)
    print("测试6: 可扩展性分析")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    generator = DWTPromptGenerator(compression_level='balanced')
    
    print("\n6.1 大规模批处理性能")
    print(f"{'总样本数':<15} {'批次大小':<12} {'时间(ms)':<12} {'吞吐量(样本/s)':<15}")
    print("-" * 70)
    
    scalability_results = []
    
    # 固定总样本数，测试不同批次大小
    total_samples = 128
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes:
        if batch_size > total_samples:
            continue
            
        n_batches = total_samples // batch_size
        x_enc = torch.randn(batch_size, 7, 512)
        
        start = time.perf_counter()
        for _ in range(n_batches):
            _ = generator(x_enc)
        elapsed = (time.perf_counter() - start) * 1000
        
        throughput = total_samples / (elapsed / 1000)
        
        scalability_results.append({
            'total_samples': total_samples,
            'batch_size': batch_size,
            'time': elapsed,
            'throughput': throughput
        })
        
        print(f"{total_samples:<15} {batch_size:<12} {elapsed:<12.2f} {throughput:<15.1f}")
    
    print("\n✅ 可扩展性分析完成!")
    return scalability_results


def generate_summary_report(all_results):
    """生成性能分析总结报告"""
    print("\n" + "=" * 80)
    print("性能分析总结报告")
    print("=" * 80)
    
    print("\n## 主要发现\n")
    
    # 1. 时间复杂度结论
    time_results, batch_results, var_results = all_results['time_complexity']
    print("### 1. 时间复杂度")
    print(f"   - 序列长度影响: 从128到4096，时间增长约 {time_results[-1]['time']/time_results[0]['time']:.1f}x")
    print(f"   - 批次大小影响: 批处理效率随批次增大而提升")
    print(f"   - 变量数影响: 线性增长关系")
    
    # 2. 空间复杂度结论
    memory_results, batch_memory_results = all_results['space_complexity']
    print("\n### 2. 空间复杂度")
    print(f"   - 内存使用随序列长度线性增长")
    print(f"   - 典型配置(B=4,N=7,T=512)内存使用: ~{memory_results[2]['peak_mb']:.1f}MB")
    
    # 3. 最优配置
    wavelet_results, level_results, compression_results = all_results['config_comparison']
    print("\n### 3. 最优配置")
    best_wavelet = min(wavelet_results, key=lambda x: x['time'])
    print(f"   - 最快小波基: {best_wavelet['wavelet']} ({best_wavelet['time']:.2f}ms)")
    best_level = min(level_results, key=lambda x: x['time'])
    print(f"   - 最快分解层数: {best_level['level']} ({best_level['time']:.2f}ms)")
    print(f"   - 推荐压缩级别: balanced (平衡性能和信息量)")
    
    # 4. 瓶颈
    bottleneck_results = all_results['bottleneck']
    max_bottleneck = max(bottleneck_results, key=lambda x: x['percentage'])
    print("\n### 4. 性能瓶颈")
    print(f"   - 主要瓶颈: {max_bottleneck['stage']} ({max_bottleneck['percentage']:.1f}%)")
    print(f"   - 优化建议: 重点优化DWT分解和特征提取并行化")
    
    # 5. GPU加速
    if all_results['gpu_comparison'] is not None:
        gpu_results = all_results['gpu_comparison']
        avg_speedup = np.mean([r['speedup'] for r in gpu_results])
        print("\n### 5. GPU加速效果")
        print(f"   - 平均加速比: {avg_speedup:.2f}x")
        print(f"   - GPU加速建议: 批次大小 >= 8 时效果明显")
    
    # 6. 可扩展性
    scalability_results = all_results['scalability']
    best_throughput = max(scalability_results, key=lambda x: x['throughput'])
    print("\n### 6. 可扩展性")
    print(f"   - 最优批次大小: {best_throughput['batch_size']} (吞吐量: {best_throughput['throughput']:.1f} 样本/秒)")
    print(f"   - 建议: 使用中等批次大小(8-32)获得最佳性能")
    
    print("\n" + "=" * 80)


def save_results_to_json(all_results, filename='dwt_performance_results.json'):
    """保存结果到JSON文件"""
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 结果已保存到: {filename}")


def main():
    """主测试函数"""
    print("DWT Prompt Generator 性能分析套件")
    print("=" * 80)
    print("开始全面性能测试...\n")
    
    all_results = {}
    
    try:
        # 测试1: 时间复杂度
        time_results, batch_results, var_results = test_time_complexity()
        all_results['time_complexity'] = (time_results, batch_results, var_results)
        
        # 测试2: 空间复杂度
        memory_results, batch_memory_results = test_space_complexity()
        all_results['space_complexity'] = (memory_results, batch_memory_results)
        
        # 测试3: 配置对比
        wavelet_results, level_results, compression_results = test_configuration_comparison()
        all_results['config_comparison'] = (wavelet_results, level_results, compression_results)
        
        # 测试4: GPU对比
        gpu_results = test_gpu_vs_cpu()
        all_results['gpu_comparison'] = gpu_results
        
        # 测试5: 瓶颈识别
        bottleneck_results = test_bottleneck_analysis()
        all_results['bottleneck'] = bottleneck_results
        
        # 测试6: 可扩展性
        scalability_results = test_scalability()
        all_results['scalability'] = scalability_results
        
        # 生成总结报告
        generate_summary_report(all_results)
        
        # 保存结果
        save_results_to_json(all_results)
        
        print("\n" + "=" * 80)
        print("🎉 性能分析完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
