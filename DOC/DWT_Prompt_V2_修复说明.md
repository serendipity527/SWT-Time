# DWT动态提示词 V2 修复说明

## 修复版本概述

`DWTPromptGeneratorV2` 是对原版DWT提示词生成器的重大改进，修复了关键bug并简化了语义表达。

## 修复的主要问题

### 1. ⚠️ **Batch维度处理Bug（严重）**

**原始问题**：
```python
# 错误代码（TimeLLM.py 第266行）
idx = b * N  # 只取第一个变量
base_info = {
    'min': min_values[idx].tolist()[0],  # ❌ 只用第一个变量
    'max': max_values[idx].tolist()[0],  # ❌ 只用第一个变量
    ...
}
```

**问题分析**：
- DWT分析的是 `(1, N, T)` 形状的多变量数据
- 但统计信息只使用了第一个变量（`idx = b*N`）
- 导致提示词无法反映其他N-1个变量的信息
- **多变量预测时信息严重丢失**

**修复方案**：
```python
# 正确代码（V2版本）
start_idx = b * N
end_idx = (b + 1) * N
base_info = {
    'min': min_values[start_idx:end_idx].mean().item(),  # ✅ 聚合所有变量
    'max': max_values[start_idx:end_idx].mean().item(),  # ✅ 聚合所有变量
    'median': medians[start_idx:end_idx].mean().item(),  # ✅ 聚合所有变量
}
```

---

### 2. 🔤 **语义过于抽象（核心问题）**

**原始问题**：
```python
# 原版输出
"Pattern analysis: single-scale dominant (trend), 
trend is consistent upward, 
signal quality is high SNR (clean)"
```

**问题分析**：
- ❌ "single-scale dominant" - LLM无法理解
- ❌ "SNR" - 学术术语，LLM预训练时很少接触
- ❌ 过度工程化的描述

**修复方案**（V2版本）：
```python
# 新版输出
"the trend of input is upward, 
the data is stable and predictable with clear 12-24-step cycles"
```

**改进**：
- ✅ 使用自然语言：upward/downward 替代 consistent upward
- ✅ 简化描述：stable 替代 high SNR (clean)
- ✅ 直观的周期描述：12-24-step cycles 替代抽象的频段分析

---

### 3. 🔄 **Lags信息不一致**

**原始问题**：
- DWT分析小波系数
- 但lags来自FFT自相关（`calcute_lags`）
- **信息来源不一致，导致矛盾**

**修复方案**：
- ✅ 从DWT的能量分布直接推导周期信息
- ✅ 移除FFT的lags参数
- ✅ 保持信息来源一致性

```python
def _extract_periodicity(self, coeffs, seq_len):
    """从DWT能量分布提取周期信息"""
    # 将小波频段映射到实际周期
    # cA3: 长周期 (>16 steps)
    # cD3: 中长周期 (8-16 steps)
    # cD2: 短期 (4-8 steps)
    # cD1: 高频 (2-4 steps)
    ...
```

---

### 4. 📉 **信息损失**

**原始问题**：
虽然提取了丰富的DWT特征，但转换为文本时丢失了LLM真正需要的信息：
- 具体的趋势数值 → 变成了抽象分类
- 简单的变化方向 → 变成了复杂的多尺度描述

**修复方案**：
- ✅ 保留原始统计量（min/max/median）
- ✅ 使用简单的趋势描述（upward/downward）
- ✅ 用DWT增强（稳定性、周期性），而非替换

---

## V2版本的特点

### **核心改进**

1. **自然语言化**
   - 使用LLM熟悉的表达方式
   - 避免学术术语（SNR → stable/volatile）
   - 直观的描述（12-step cycles 而非 cD2 dominant）

2. **信息完整性**
   - 正确聚合多变量信息
   - 从DWT统一提取所有特征
   - 保持信息来源一致

3. **格式兼容性**
   - 保持与原始提示词相似的结构
   - LLM无需重新适应新格式

### **提示词对比**

#### 原始提示词（基线）
```
<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.Task description: forecast the next 96 steps given the previous 96 steps information; Input statistics: min value 0.234, max value 1.876, median value 0.987, the trend of input is upward, top 5 lags are : [24, 48, 72, 96, 12]<|<end_prompt>|>
```

#### DWT V1 (有问题的版本)
```
<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.
Task description: forecast the next 96 steps given the previous 96 steps information; 
Input statistics: range [0.23, 1.88], median 0.99
Pattern analysis: single-scale dominant (trend), trend is consistent upward, signal quality is high SNR (clean)
Dominant periodicities: [24, 48, 72]<|<end_prompt>|>
```

#### DWT V2 (修复版) - Balanced
```
<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.Task description: forecast the next 96 steps given the previous 96 steps information; Input statistics: min value 0.234, max value 1.876, median value 0.987, the trend of input is upward, the data is stable and predictable with clear 12-24-step cycles<|<end_prompt>|>
```

**对比分析**：
- V2保留了基线的结构和min/max/median
- 用自然语言描述趋势和稳定性
- 从DWT提取的周期信息更准确
- 长度适中（约150 tokens）

---

## 使用方法

### 基础使用

```bash
# 使用修复后的DWT Prompt V2
python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id dwt_v2_test \
    --model_comment dwt_v2 \
    --model TimeLLM \
    --data ETTh1 \
    --use_dwt_prompt \
    --prompt_compression balanced
```

### 压缩级别选择

#### Minimal（最简洁，约80 tokens）
```bash
--prompt_compression minimal
```
**示例输出**：
```
<|start_prompt|>The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.
Forecast 96 from 96: range [0.23, 1.88], median 0.99, trend up, stable<|<end_prompt>|>
```

**适用场景**：
- 快速实验
- 资源受限
- 简单任务

#### Balanced（推荐，约150 tokens）
```bash
--prompt_compression balanced
```
**示例输出**：
```
<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.Task description: forecast the next 96 steps given the previous 96 steps information; Input statistics: min value 0.234, max value 1.876, median value 0.987, the trend of input is upward, the data is stable and predictable with clear 12-24-step cycles<|<end_prompt>|>
```

**适用场景**：
- 大多数预测任务
- 平衡信息量和长度
- **默认推荐**

#### Detailed（最详细，约250 tokens）
```bash
--prompt_compression detailed
```
**示例输出**：
```
<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.
Task description: forecast the next 96 steps given the previous 96 steps information;
Input statistics:
  - Value range: min=0.234, max=1.876, median=0.987
  - Trend: moderate upward trend (strength: 0.15)
  - Stability: signal-to-noise ratio is 18.5 dB (high quality signal)
  - Dominant periodicities: 12-24, 24-48 steps; Energy distribution: long-term 35.2%, seasonal 28.1%, short-term 15.3%
  - Short-term volatility accounts for 15.3% of total energy<|<end_prompt>|>
```

**适用场景**：
- 复杂预测任务
- 需要详细模式信息
- 高精度要求

---

## 消融实验设计

### 实验1：验证Bug修复的效果

```bash
# 基线（原始统计）
python run_main.py --model_id baseline --model TimeLLM --data ETTh1

# DWT V1（有bug的版本）
# 需要临时切换回旧版本或者注释掉V2的导入
python run_main.py --model_id dwt_v1 --use_dwt_prompt --prompt_compression balanced

# DWT V2（修复版）
python run_main.py --model_id dwt_v2 --use_dwt_prompt --prompt_compression balanced
```

**预期结果**：
- V2应该显著优于V1（因为修复了batch维度bug）
- V2应该优于或接近基线（因为提供了更丰富的信息）

### 实验2：压缩级别对比

```bash
# Minimal
python run_main.py --model_id dwt_minimal --use_dwt_prompt --prompt_compression minimal

# Balanced
python run_main.py --model_id dwt_balanced --use_dwt_prompt --prompt_compression balanced

# Detailed
python run_main.py --model_id dwt_detailed --use_dwt_prompt --prompt_compression detailed
```

**假设**：
- Balanced应该是最优选择（信息量与长度的平衡）
- Minimal可能信息不足
- Detailed可能导致注意力分散

### 实验3：组合实验

```bash
# 仅SWT Embedding
python run_main.py --model_id swt_only --use_swt

# 仅DWT Prompt V2
python run_main.py --model_id dwt_only --use_dwt_prompt

# 组合（SWT + DWT）
python run_main.py --model_id swt_dwt --use_swt --use_dwt_prompt
```

---

## Debug工具

### 打印提示词内容

在 `TimeLLM.py` 第280行后添加：

```python
# Debug: 打印第一个batch的prompt
if b == 0 and self.training:
    print(f"\n{'='*80}")
    print(f"[DEBUG] Prompt Preview (Batch 0):")
    print(f"{prompt_[:300]}...")
    print(f"[DEBUG] Prompt Length: {len(prompt_)} chars")
    print(f"{'='*80}\n")
```

### 对比Token数量

```python
# 在生成prompt后统计
if b == 0:
    tokens = self.tokenizer(prompt_, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
    print(f"[DEBUG] Token count: {tokens.shape[1]}")
```

### 检查DWT特征值

在 `DWTPromptGeneratorV2.py` 的 `forward` 方法末尾添加：

```python
# Debug输出
if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    print(f"[DWT Features] SNR: {features['stability_info']['snr_db']:.2f} dB")
    print(f"[DWT Features] Trend: {features['trend_info']['direction'].item():.3f}")
    print(f"[DWT Features] Periods: {features['periodicity_info']['periods']}")
```

---

## 预期效果提升

基于修复的问题，预期V2版本应该带来以下提升：

### 1. **性能提升**
- 修复batch维度bug后，多变量信息完整
- 预期MSE/MAE降低 **5-10%**

### 2. **训练稳定性**
- 提示词更自然，LLM理解更好
- 训练loss曲线更平滑

### 3. **泛化能力**
- 周期信息更准确（来自DWT而非FFT）
- 在不同数据集上表现更一致

---

## 注意事项

1. **版本切换**
   - V2是独立的类，不影响旧版本
   - 通过导入路径切换：`from layers.DWTPromptGenerator_v2 import DWTPromptGeneratorV2`

2. **兼容性**
   - V2移除了缓存机制（可后续添加）
   - 接口与V1兼容，可直接替换

3. **性能**
   - V2计算量略低（移除了复杂的语义映射）
   - 内存占用相同

---

## 下一步计划

### 短期优化
- [ ] 添加LRU缓存（提高重复batch的速度）
- [ ] 支持更多小波基（sym, coif等）
- [ ] 自适应压缩级别（根据难度动态调整）

### 长期研究
- [ ] 可学习的提示词生成（用小型MLP替代规则）
- [ ] 多模态提示词（结合时间戳、外部知识）
- [ ] 提示词蒸馏（将DWT知识压缩到更短的token）

---

## 总结

DWTPromptGeneratorV2 通过以下三个核心改进，预期能够显著提升模型性能：

1. ✅ **修复batch维度bug** - 确保多变量信息完整
2. ✅ **简化语义表达** - 使用LLM熟悉的自然语言
3. ✅ **保持信息一致性** - 统一从DWT提取所有特征

**关键优势**：保留原始提示词的优点，用DWT增强而非完全替换。

---

*更新时间: 2025-12-02*  
*版本: V2.0*  
*作者: SWT-Time Team*
