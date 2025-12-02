#!/usr/bin/env python
"""
训练加速效果对比测试脚本
自动测试不同优化方案的加速效果
"""

import subprocess
import json
import os
import sys
from datetime import timedelta

# 确保在项目根目录执行
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

def format_time(seconds):
    """格式化时间"""
    return str(timedelta(seconds=int(seconds)))

def run_estimate(config_name, extra_args):
    """运行时间预估并返回结果"""
    print("\n" + "="*80)
    print(f"测试配置: {config_name}")
    print("="*80)
    
    # 基础参数
    base_args = [
        "python", "estimate_training_time.py",
        "--model", "TimeLLM",
        "--data", "ETTm1",
        "--root_path", "./dataset",
        "--data_path", "ETT-small/ETTm1.csv",
        "--features", "M",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", "96",
        "--enc_in", "7",
        "--dec_in", "7",
        "--c_out", "7",
        "--d_model", "16",
        "--n_heads", "8",
        "--e_layers", "2",
        "--d_layers", "1",
        "--d_ff", "32",
        "--train_epochs", "10",
        "--itr", "1",
        "--learning_rate", "0.0001",
        "--llm_model", "GPT2",
        "--llm_dim", "768",
        "--llm_layers", "6",
        "--use_swt",
        "--swt_level", "3",
        "--use_dwt_prompt",
        "--dwt_prompt_level", "3",
        "--warmup_batches", "5",
        "--measure_batches", "10"
    ]
    
    cmd = base_args + extra_args
    print(f"额外参数: {' '.join(extra_args)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 读取结果JSON
        with open('training_time_estimate.json', 'r') as f:
            data = json.load(f)
        
        return {
            'success': True,
            'config_name': config_name,
            'avg_batch_time': data['measured_times']['avg_train_batch_time'],
            'epoch_time': data['time_estimates']['total_time_per_epoch_seconds'],
            'total_time': data['time_estimates']['total_training_time_seconds'],
            'batch_size': data['data_info']['batch_size'],
            'use_amp': data['training_config'].get('use_amp', False)
        }
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return {'success': False, 'config_name': config_name}
    except Exception as e:
        print(f"解析结果时出错: {e}")
        return {'success': False, 'config_name': config_name}

def main():
    """主函数"""
    print("="*80)
    print("训练加速效果对比测试")
    print("="*80)
    print("\n将测试以下配置:")
    print("1. 基线配置 (batch_size=32, 无AMP)")
    print("2. 启用AMP (batch_size=32)")
    print("3. 大batch (batch_size=64, 无AMP)")
    print("4. AMP + 大batch (batch_size=64)")
    print("\n每个配置预计耗时: 1-2分钟")
    print("总预计耗时: 4-8分钟")
    
    response = input("\n是否继续? [y/N]: ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 定义测试配置
    configs = [
        ("基线配置", ["--batch_size", "32"]),
        ("启用AMP", ["--batch_size", "32", "--use_amp"]),
        ("大Batch", ["--batch_size", "64"]),
        ("AMP+大Batch", ["--batch_size", "64", "--use_amp"]),
    ]
    
    results = []
    
    # 运行测试
    for config_name, extra_args in configs:
        result = run_estimate(config_name, extra_args)
        results.append(result)
        
        if result['success']:
            print(f"✓ {config_name} 完成")
            print(f"  - 批次时间: {result['avg_batch_time']:.4f}秒")
            print(f"  - Epoch时间: {format_time(result['epoch_time'])}")
        else:
            print(f"✗ {config_name} 失败")
    
    # 输出对比结果
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    # 找出基线配置
    baseline = next((r for r in results if r['success'] and r['config_name'] == "基线配置"), None)
    
    if not baseline:
        print("错误: 基线配置测试失败")
        return
    
    baseline_epoch_time = baseline['epoch_time']
    
    print(f"\n{'配置':<15} {'Batch':<8} {'AMP':<6} {'批次时间(秒)':<15} {'Epoch时间':<12} {'10 Epochs':<12} {'加速比':<8}")
    print("-" * 95)
    
    for result in results:
        if not result['success']:
            continue
        
        config = result['config_name']
        batch_size = result['batch_size']
        use_amp = '是' if result['use_amp'] else '否'
        batch_time = f"{result['avg_batch_time']:.4f}"
        epoch_time = format_time(result['epoch_time'])
        total_time = format_time(result['total_time'])
        speedup = baseline_epoch_time / result['epoch_time']
        speedup_str = f"{speedup:.2f}x"
        
        print(f"{config:<15} {batch_size:<8} {use_amp:<6} {batch_time:<15} {epoch_time:<12} {total_time:<12} {speedup_str:<8}")
    
    # 保存详细结果
    output_file = 'speedup_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': baseline,
            'results': results,
            'summary': {
                'baseline_epoch_time': baseline_epoch_time,
                'baseline_total_time': baseline['total_time']
            }
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 推荐配置
    print("\n" + "="*80)
    print("推荐配置")
    print("="*80)
    
    # 找出最快的配置
    fastest = min((r for r in results if r['success']), key=lambda x: x['epoch_time'])
    
    print(f"\n最快配置: {fastest['config_name']}")
    print(f"  - 批次大小: {fastest['batch_size']}")
    print(f"  - 混合精度: {'启用' if fastest['use_amp'] else '禁用'}")
    print(f"  - Epoch时间: {format_time(fastest['epoch_time'])}")
    print(f"  - 10 Epochs: {format_time(fastest['total_time'])}")
    print(f"  - 加速比: {baseline_epoch_time / fastest['epoch_time']:.2f}x")
    
    # 生成推荐命令
    print("\n推荐的训练命令:")
    print("-" * 80)
    amp_flag = "--use_amp \\" if fastest['use_amp'] else ""
    print(f"""
python run_main.py \\
    --task_name long_term_forecast \\
    --is_training 1 \\
    --model_id optimized \\
    --model_comment speed_optimized \\
    --model TimeLLM \\
    --data ETTm1 \\
    --root_path ./dataset \\
    --data_path ETT-small/ETTm1.csv \\
    --features M \\
    --seq_len 96 \\
    --pred_len 96 \\
    --batch_size {fastest['batch_size']} \\
    {amp_flag}    --llm_model GPT2 \\
    --llm_dim 768 \\
    --llm_layers 6 \\
    --train_epochs 10 \\
    --use_swt \\
    --use_dwt_prompt
""")
    
    print("="*80)

if __name__ == "__main__":
    main()
