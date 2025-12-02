#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练时间预估工具

通过运行少量训练批次来测量实际速度,预估整体训练所需时间。
参数与 run_main.py 保持一致,方便直接对比。

使用方法:
    python estimate_training_time.py \
        --model TimeLLM \
        --data ETTm1 \
        --root_path ./dataset \
        --data_path ETTm1.csv \
        --train_epochs 10 \
        --batch_size 32

作者: 参考 run_main.py 创建
"""

import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
import json
from datetime import datetime, timedelta

# 环境变量配置
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content

# ============================================================================
# 参数解析 (与 run_main.py 保持一致)
# ============================================================================

parser = argparse.ArgumentParser(description='训练时间预估工具 (基于 Time-LLM)')

# 设置随机种子
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ===== 基础配置 =====
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='任务名称: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='训练状态')
parser.add_argument('--model_id', type=str, default='test', help='模型ID')
parser.add_argument('--model_comment', type=str, default='none', help='模型备注')
parser.add_argument('--model', type=str, default='TimeLLM',
                    help='模型名称: [Autoformer, DLinear, TimeLLM]')
parser.add_argument('--seed', type=int, default=2021, help='随机种子')

# ===== 数据加载器配置 =====
parser.add_argument('--data', type=str, default='ETTm1', help='数据集类型')
parser.add_argument('--root_path', type=str, default='./dataset', help='数据文件根路径')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名')
parser.add_argument('--features', type=str, default='M',
                    help='预测任务: [M, S, MS]; M:多变量预测多变量, S:单变量预测单变量, MS:多变量预测单变量')
parser.add_argument('--target', type=str, default='OT', help='S或MS任务中的目标特征')
parser.add_argument('--loader', type=str, default='modal', help='数据集类型')
parser.add_argument('--freq', type=str, default='h',
                    help='时间特征编码频率: [s:秒, t:分钟, h:小时, d:天, b:工作日, w:周, m:月]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存位置')

# ===== 预测任务配置 =====
parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
parser.add_argument('--label_len', type=int, default=48, help='起始token长度')
parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4的子集')

# ===== 模型配置 =====
parser.add_argument('--enc_in', type=int, default=7, help='编码器输入维度')
parser.add_argument('--dec_in', type=int, default=7, help='解码器输入维度')
parser.add_argument('--c_out', type=int, default=7, help='输出维度')
parser.add_argument('--d_model', type=int, default=16, help='模型维度')
parser.add_argument('--n_heads', type=int, default=8, help='注意力头数量')
parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
parser.add_argument('--d_ff', type=int, default=32, help='全连接层维度')
parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
parser.add_argument('--factor', type=int, default=1, help='注意力因子')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
parser.add_argument('--embed', type=str, default='timeF',
                    help='时间特征编码: [timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
parser.add_argument('--output_attention', action='store_true', help='是否在编码器中输出注意力')
parser.add_argument('--patch_len', type=int, default=16, help='patch长度')
parser.add_argument('--stride', type=int, default=8, help='步长')
parser.add_argument('--prompt_domain', type=int, default=0, help='prompt domain')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM模型: [LLAMA, GPT2, BERT]')
parser.add_argument('--llm_dim', type=int, default=4096, 
                    help='LLM模型维度: LLAMA-7b:4096, GPT2-small:768, BERT-base:768')

# ===== 优化配置 =====
parser.add_argument('--num_workers', type=int, default=10, help='数据加载器的worker数量')
parser.add_argument('--itr', type=int, default=1, help='实验迭代次数')
parser.add_argument('--train_epochs', type=int, default=10, help='训练epoch数')
parser.add_argument('--align_epochs', type=int, default=10, help='对齐epoch数')
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--eval_batch_size', type=int, default=8, help='评估批次大小')
parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
parser.add_argument('--des', type=str, default='test', help='实验描述')
parser.add_argument('--loss', type=str, default='MSE', help='损失函数')
parser.add_argument('--lradj', type=str, default='type1', help='学习率调整策略')
parser.add_argument('--pct_start', type=float, default=0.2, help='OneCycleLR的pct_start')
parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练', default=False)
parser.add_argument('--llm_layers', type=int, default=6, help='LLM层数')
parser.add_argument('--percent', type=int, default=100, help='数据使用百分比')

# ===== SWT参数 =====
parser.add_argument('--use_swt', action='store_true', help='使用SWT-Patch Embedding', default=False)
parser.add_argument('--swt_wavelet', type=str, default='db4', 
                    help='小波类型: [db4, db6, sym4, coif2, haar]')
parser.add_argument('--swt_level', type=int, default=3, help='SWT分解层数')
parser.add_argument('--use_all_coeffs', action='store_true', help='使用所有系数', default=False)

# ===== DWT Prompt参数 =====
parser.add_argument('--use_dwt_prompt', action='store_true', help='使用DWT动态prompt', default=False)
parser.add_argument('--dwt_prompt_level', type=int, default=3, help='DWT分解层数')
parser.add_argument('--prompt_compression', type=str, default='balanced',
                    choices=['minimal', 'balanced', 'detailed'], help='prompt压缩级别')

# ===== 时间预估专用参数 =====
parser.add_argument('--warmup_batches', type=int, default=5, 
                    help='预热批次数(不计入时间统计)')
parser.add_argument('--measure_batches', type=int, default=10,
                    help='测量批次数(用于计算平均时间)')
parser.add_argument('--use_deepspeed', action='store_true', 
                    help='使用DeepSpeed进行预估', default=False)

args = parser.parse_args()

# ============================================================================
# 工具函数
# ============================================================================

def format_time(seconds):
    """格式化时间为易读格式"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}分钟 ({seconds:.1f}秒)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.2f}小时 ({minutes:.1f}分钟)"


def get_model_size(model):
    """获取模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_section(title, width=80):
    """打印分节标题"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def measure_training_speed(model, train_loader, model_optim, criterion, 
                          accelerator, args, warmup_batches=5, measure_batches=10):
    """
    测量训练速度
    
    Args:
        warmup_batches: 预热批次数(不计入统计)
        measure_batches: 测量批次数(用于计算平均时间)
    
    Returns:
        avg_batch_time: 平均每批次训练时间(秒)
    """
    model.train()
    batch_times = []
    
    print(f"\n开始测量训练速度...")
    print(f"  - 预热批次: {warmup_batches}")
    print(f"  - 测量批次: {measure_batches}")
    
    total_batches = warmup_batches + measure_batches
    batch_count = 0
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if batch_count >= total_batches:
            break
            
        batch_start = time.time()
        
        # 数据准备
        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)
        
        # 解码器输入
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
        
        # 前向传播
        model_optim.zero_grad()
        
        if args.use_amp:
            with torch.cuda.amp.autocast():
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                loss = criterion(outputs, batch_y)
        else:
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
        
        # 反向传播
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            accelerator.backward(loss)
            model_optim.step()
        
        batch_time = time.time() - batch_start
        
        # 预热阶段后才开始记录
        if batch_count >= warmup_batches:
            batch_times.append(batch_time)
            status = f"  测量批次 {len(batch_times)}/{measure_batches}: {batch_time:.4f}秒"
            print(status)
        else:
            print(f"  预热批次 {batch_count+1}/{warmup_batches}: {batch_time:.4f}秒")
        
        batch_count += 1
    
    if len(batch_times) == 0:
        raise ValueError("没有收集到足够的训练时间数据!")
    
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    
    print(f"\n训练速度测量完成:")
    print(f"  - 平均时间: {avg_batch_time:.4f}秒/批次")
    print(f"  - 标准差: {std_batch_time:.4f}秒")
    print(f"  - 最小值: {min(batch_times):.4f}秒")
    print(f"  - 最大值: {max(batch_times):.4f}秒")
    
    return avg_batch_time


def measure_validation_speed(model, vali_loader, criterion, mae_metric,
                            accelerator, args, measure_batches=10):
    """测量验证速度"""
    model.eval()
    batch_times = []
    
    print(f"\n开始测量验证速度...")
    print(f"  - 测量批次: {measure_batches}")
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            if i >= measure_batches:
                break
            
            batch_start = time.time()
            
            # 数据准备
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            # 解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # 前向传播
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            print(f"  验证批次 {i+1}/{measure_batches}: {batch_time:.4f}秒")
    
    if len(batch_times) == 0:
        raise ValueError("没有收集到足够的验证时间数据!")
    
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)
    
    print(f"\n验证速度测量完成:")
    print(f"  - 平均时间: {avg_batch_time:.4f}秒/批次")
    print(f"  - 标准差: {std_batch_time:.4f}秒")
    
    return avg_batch_time


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print_section("训练时间预估工具", 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化 Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
        print("使用 DeepSpeed 进行预估")
    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        print("使用标准 Accelerator 进行预估")
    
    # 1. 加载数据
    print_section("步骤 1/5: 加载数据", 80)
    print(f"数据集: {args.data}")
    print(f"数据路径: {args.root_path}/{args.data_path}")
    print(f"序列长度: {args.seq_len} -> 预测长度: {args.pred_len}")
    
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    train_batches = len(train_loader)
    vali_batches = len(vali_loader)
    test_batches = len(test_loader)
    
    print(f"\n数据加载完成:")
    print(f"  - 训练批次数: {train_batches}")
    print(f"  - 验证批次数: {vali_batches}")
    print(f"  - 测试批次数: {test_batches}")
    print(f"  - 批次大小: {args.batch_size}")
    
    # 2. 创建模型
    print_section("步骤 2/5: 创建模型", 80)
    print(f"模型: {args.model}")
    
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    total_params, trainable_params = get_model_size(model)
    
    print(f"\n模型信息:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 参数占用: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    if args.use_swt:
        print(f"\n使用 SWT 优化:")
        print(f"  - 小波类型: {args.swt_wavelet}")
        print(f"  - 分解层数: {args.swt_level}")
        print(f"  - 使用所有系数: {args.use_all_coeffs}")
    
    if args.use_dwt_prompt:
        print(f"\n使用 DWT 动态 Prompt:")
        print(f"  - 分解层数: {args.dwt_prompt_level}")
        print(f"  - 压缩级别: {args.prompt_compression}")
    
    # 3. 配置优化器
    print_section("步骤 3/5: 配置优化器", 80)
    
    args.content = load_content(args)
    
    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    
    train_steps = len(train_loader)
    
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate)
    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    print(f"优化器: Adam")
    print(f"学习率: {args.learning_rate}")
    print(f"学习率调整: {args.lradj}")
    print(f"损失函数: {args.loss}")
    print(f"使用AMP: {args.use_amp}")
    
    # 准备模型和数据加载器
    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    # 4. 测量训练和验证速度
    print_section("步骤 4/5: 测量训练速度", 80)
    
    avg_train_batch_time = measure_training_speed(
        model, train_loader, model_optim, criterion, accelerator, args,
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches
    )
    
    avg_vali_batch_time = measure_validation_speed(
        model, vali_loader, criterion, mae_metric, accelerator, args,
        measure_batches=min(args.measure_batches, vali_batches)
    )
    
    # 假设测试速度与验证速度相同
    avg_test_batch_time = avg_vali_batch_time
    
    # 5. 计算时间预估
    print_section("步骤 5/5: 计算时间预估", 80)
    
    # 单个 epoch 的时间
    train_time_per_epoch = avg_train_batch_time * train_batches
    vali_time_per_epoch = avg_vali_batch_time * vali_batches
    test_time_per_epoch = avg_test_batch_time * test_batches
    
    # 每个epoch的总时间(训练+验证)
    # 注意: 测试通常只在训练结束后进行一次
    total_time_per_epoch = train_time_per_epoch + vali_time_per_epoch
    
    # 所有 epochs 的训练时间
    total_training_time = total_time_per_epoch * args.train_epochs
    
    # 加上最后的测试时间
    total_time_with_test = total_training_time + test_time_per_epoch
    
    # 如果有多次迭代
    total_time_all_itr = total_time_with_test * args.itr
    
    # 6. 输出结果
    print_section("时间预估结果", 80)
    
    print("\n【测量结果】")
    print(f"  平均训练时间: {avg_train_batch_time:.4f} 秒/批次")
    print(f"  平均验证时间: {avg_vali_batch_time:.4f} 秒/批次")
    print(f"  平均测试时间: {avg_test_batch_time:.4f} 秒/批次")
    
    print("\n【单个 Epoch 时间预估】")
    print(f"  训练阶段: {format_time(train_time_per_epoch)}")
    print(f"  验证阶段: {format_time(vali_time_per_epoch)}")
    print(f"  测试阶段: {format_time(test_time_per_epoch)}")
    print(f"  总计 (训练+验证): {format_time(total_time_per_epoch)}")
    
    print(f"\n【完整训练时间预估】")
    print(f"  训练 epochs: {args.train_epochs}")
    print(f"  总训练时间: {format_time(total_training_time)}")
    print(f"  包含最终测试: {format_time(total_time_with_test)}")
    
    if args.itr > 1:
        print(f"\n【多次迭代时间预估】")
        print(f"  迭代次数: {args.itr}")
        print(f"  总时间: {format_time(total_time_all_itr)}")
    
    print(f"\n【额外信息】")
    epochs_per_hour = 3600 / total_time_per_epoch if total_time_per_epoch > 0 else 0
    epochs_per_day = 86400 / total_time_per_epoch if total_time_per_epoch > 0 else 0
    print(f"  每小时可完成 epochs: {epochs_per_hour:.2f}")
    print(f"  每天可完成 epochs: {epochs_per_day:.2f}")
    
    # 预计完成时间
    estimated_end_time = datetime.now() + timedelta(seconds=total_time_all_itr)
    print(f"  预计完成时间: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 7. 保存结果到 JSON 文件
    results = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_info": {
            "model_name": args.model,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "data_info": {
            "dataset": args.data,
            "train_batches": train_batches,
            "vali_batches": vali_batches,
            "test_batches": test_batches,
            "batch_size": args.batch_size,
        },
        "training_config": {
            "train_epochs": args.train_epochs,
            "iterations": args.itr,
            "learning_rate": args.learning_rate,
            "use_amp": args.use_amp,
            "use_swt": args.use_swt,
            "use_dwt_prompt": args.use_dwt_prompt,
        },
        "measured_times": {
            "avg_train_batch_time_seconds": float(avg_train_batch_time),
            "avg_vali_batch_time_seconds": float(avg_vali_batch_time),
            "avg_test_batch_time_seconds": float(avg_test_batch_time),
            "warmup_batches": args.warmup_batches,
            "measure_batches": args.measure_batches,
        },
        "time_estimates": {
            "train_time_per_epoch_seconds": float(train_time_per_epoch),
            "vali_time_per_epoch_seconds": float(vali_time_per_epoch),
            "test_time_per_epoch_seconds": float(test_time_per_epoch),
            "total_time_per_epoch_seconds": float(total_time_per_epoch),
            "total_training_time_seconds": float(total_training_time),
            "total_time_with_test_seconds": float(total_time_with_test),
            "total_time_all_itr_seconds": float(total_time_all_itr),
        },
        "additional_info": {
            "epochs_per_hour": float(epochs_per_hour),
            "epochs_per_day": float(epochs_per_day),
            "estimated_end_time": estimated_end_time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    
    output_file = "training_time_estimate.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print_section(f"结果已保存", 80)
    print(f"输出文件: {output_file}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 80)
    print(" 预估完成! 祝训练顺利!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
