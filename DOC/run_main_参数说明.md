# run_main.py 命令参数说明文档

## 概述

`run_main.py` 是 SWT-Time 项目的主要训练脚本，支持多种时间序列预测模型和配置选项。本文档详细介绍了所有可用的命令行参数及其用法。

## 基础配置参数

### 任务配置
```bash
--task_name          # 任务类型，必选
                    # 选项: long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection
                    # 默认值: long_term_forecast

--is_training        # 是否为训练模式，必选
                    # 类型: int (1=训练, 0=测试)
                    # 默认值: 1

--model_id           # 模型标识符，必选
                    # 用于区分不同的实验配置
                    # 示例: test, experiment_001

--model_comment      # 模型注释前缀，必选
                    # 用于保存测试结果时的文件名前缀
                    # 示例: none, baseline, swt_experiment

--model              # 模型名称，必选
                    # 选项: Autoformer, DLinear, TimeLLM
                    # 默认值: Autoformer

--seed               # 随机种子
                    # 类型: int
                    # 默认值: 2021
```

### 数据加载配置
```bash
--data               # 数据集类型，必选
                    # 示例: ETTh1, ETTh2, ETTm1, ETTm2, custom
                    # 默认值: ETTh1

--root_path          # 数据文件根路径
                    # 默认值: ./dataset

--data_path          # 具体数据文件名
                    # 默认值: ETTh1.csv

--features           # 预测任务特征类型
                    # 选项: M(多变量预测多变量), S(单变量预测单变量), MS(多变量预测单变量)
                    # 默认值: M

--target             # S或MS任务中的目标特征
                    # 默认值: OT

--loader             # 数据集加载器类型
                    # 默认值: modal

--freq               # 时间特征编码频率
                    # 选项: s(秒), t(分), h(小时), d(天), b(工作日), w(周), m(月)
                    # 也支持更详细的频率如 15min, 3h
                    # 默认值: h

--checkpoints        # 模型检查点保存位置
                    # 默认值: ./checkpoints/
```

## 预测任务参数

```bash
--seq_len            # 输入序列长度
                    # 类型: int
                    # 默认值: 96

--label_len          # 起始token长度
                    # 类型: int
                    # 默认值: 48

--pred_len           # 预测序列长度
                    # 类型: int
                    # 默认值: 96

--seasonal_patterns  # M4数据集的子集
                    # 默认值: Monthly
```

## 模型定义参数

### 基础模型参数
```bash
--enc_in             # 编码器输入维度
                    # 类型: int
                    # 默认值: 7

--dec_in             # 解码器输入维度
                    # 类型: int
                    # 默认值: 7

--c_out              # 输出维度
                    # 类型: int
                    # 默认值: 7

--d_model            # 模型维度
                    # 类型: int
                    # 默认值: 16

--n_heads            # 注意力头数量
                    # 类型: int
                    # 默认值: 8

--e_layers           # 编码器层数
                    # 类型: int
                    # 默认值: 2

--d_layers           # 解码器层数
                    # 类型: int
                    # 默认值: 1

--d_ff               # 前馈网络维度
                    # 类型: int
                    # 默认值: 32

--moving_avg         # 移动平均窗口大小
                    # 类型: int
                    # 默认值: 25

--factor             # 注意力因子
                    # 类型: int
                    # 默认值: 1

--dropout            # Dropout比例
                    # 类型: float
                    # 默认值: 0.1
```

### 嵌入和激活参数
```bash
--embed              # 时间特征编码方式
                    # 选项: timeF, fixed, learned
                    # 默认值: timeF

--activation         # 激活函数
                    # 选项: gelu, relu
                    # 默认值: gelu

--output_attention   # 是否输出注意力权重
                    # 动作参数，添加后启用
                    # 默认: False
```

### Patch相关参数
```bash
--patch_len          # Patch长度
                    # 类型: int
                    # 默认值: 16

--stride             # Patch步长
                    # 类型: int
                    # 默认值: 8
```

### LLM相关参数
```bash
--prompt_domain      # Prompt域标识
                    # 类型: int
                    # 默认值: 0

--llm_model          # LLM模型类型
                    # 选项: LLAMA, GPT2, BERT
                    # 默认值: GPT2

--llm_dim            # LLM模型维度
                    # 类型: int
                    # 默认值: 768
                    # 说明: LLama7b=4096, GPT2-small=768, BERT-base=768
```

## 优化参数

### 训练配置
```bash
--num_workers        # 数据加载器工作进程数
                    # 类型: int
                    # 默认值: 10

--itr                # 实验重复次数
                    # 类型: int
                    # 默认值: 1

--train_epochs       # 训练轮数
                    # 类型: int
                    # 默认值: 10

--align_epochs       # 对齐轮数
                    # 类型: int
                    # 默认值: 10

--batch_size         # 训练批次大小
                    # 类型: int
                    # 默认值: 32

--eval_batch_size    # 评估批次大小
                    # 类型: int
                    # 默认值: 8

--patience           # 早停耐心值
                    # 类型: int
                    # 默认值: 10

--learning_rate      # 优化器学习率
                    # 类型: float
                    # 默认值: 0.0001

--des                # 实验描述
                    # 默认值: test

--loss               # 损失函数
                    # 选项: MSE, MAE等
                    # 默认值: MSE
```

### 学习率调度
```bash
--lradj              # 学习率调整策略
                    # 选项: type1, COS, TST
                    # 默认值: type1

--pct_start          # OneCycleLR的起始百分比
                    # 类型: float
                    # 默认值: 0.2
```

### 其他优化选项
```bash
--use_amp            # 是否使用自动混合精度训练
                    # 动作参数，添加后启用
                    # 默认: False

--llm_layers         # LLM层数
                    # 类型: int
                    # 默认值: 6

--percent            # 数据使用百分比
                    # 类型: int
                    # 默认值: 100
```

## SWT（平稳小波变换）参数

### SWT核心参数
```bash
--use_swt            # 是否使用SWT-Patch Embedding
                    # 动作参数，默认启用
                    # 默认: True

--swt_wavelet        # 小波基类型
                    # 选项: db4, db6, sym4, coif2, haar
                    # 默认值: db4

--swt_level          # SWT分解层数
                    # 类型: int
                    # 默认值: 3

--use_all_coeffs     # 是否使用所有系数
                    # 动作参数，默认启用
                    # 默认: True
```

## DWT（离散小波变换）动态Prompt参数

### DWT Prompt核心参数
```bash
--use_dwt_prompt     # 是否使用DWT动态提示词
                    # 动作参数，默认禁用
                    # 默认: False

--dwt_prompt_level   # DWT分解层数（用于prompt）
                    # 类型: int
                    # 默认值: 3

--prompt_compression # Prompt压缩级别
                    # 选项: minimal, balanced, detailed
                    # 默认值: balanced
```

## 使用示例

### 基础训练命令
```bash
python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id test \
    --model_comment baseline \
    --model TimeLLM \
    --data ETTh1
```

### 启用SWT和DWT的完整配置
```bash
python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id swt_dwt_experiment \
    --model_comment swt_dwt_test \
    --model TimeLLM \
    --data ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 16 \
    --batch_size 32 \
    --train_epochs 10 \
    --learning_rate 0.0001 \
    --use_swt \
    --swt_wavelet db4 \
    --swt_level 3 \
    --use_all_coeffs \
    --use_dwt_prompt \
    --dwt_prompt_level 3 \
    --prompt_compression balanced
```

### 禁用SWT，启用DWT
```bash
python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id dwt_only \
    --model_comment dwt_test \
    --model TimeLLM \
    --data ETTh1 \
    --use_dwt_prompt \
    --dwt_prompt_level 3 \
    --prompt_compression detailed
```

### 高性能训练配置
```bash
python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id high_performance \
    --model_comment hp_test \
    --model TimeLLM \
    --data ETTh1 \
    --batch_size 64 \
    --train_epochs 20 \
    --learning_rate 0.0005 \
    --use_amp \
    --use_swt \
    --use_dwt_prompt \
    --prompt_compression minimal
```

## 参数组合建议

### 快速验证（小模型）
- `--d_model 16 --batch_size 16 --train_epochs 5`

### 标准训练
- `--d_model 32 --batch_size 32 --train_epochs 10`

### 高精度训练
- `--d_model 64 --batch_size 16 --train_epochs 20 --use_amp`

### 消融实验
- 仅SWT: `--use_swt`（不加DWT参数）
- 仅DWT: `--use_dwt_prompt`（不加SWT参数）
- 两者都启用: `--use_swt --use_dwt_prompt`
- 基线: 不添加SWT和DWT参数

## 注意事项

1. **必选参数**: `--task_name`, `--is_training`, `--model_id`, `--model_comment`, `--model`, `--data` 必须提供
2. **SWT参数**: `--use_swt` 默认启用，如需禁用请不添加此参数
3. **DWT参数**: `--use_dwt_prompt` 默认禁用，需要显式添加
4. **内存优化**: 大批次时建议使用 `--use_amp` 和减小 `--num_workers`
5. **GPU兼容**: 确保CUDA版本与PyTorch版本兼容

## 文件输出

训练完成后，模型检查点和日志将保存在：
- 检查点路径: `./checkpoints/{setting}-{model_comment}/`
- 日志文件: 控制台输出 + 可选的tensorboard日志

---
*更新时间: 2025-12-02*
*版本: v1.0*
