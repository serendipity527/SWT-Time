# DWT Prompt Generator 性能优化建议

## 当前性能评估

### 测试结果
- **平均耗时**: 3-4ms per call (NVIDIA RTX A6000)
- **主要瓶颈**: DWT分解 (62.8%)
- **优化效果**: 已提升8.6%

### 实际影响
```
Batch Size = 32:
  - DWT总耗时: ~112ms (32 × 3.5ms)
  - 占训练时间: ~9%
  - 影响等级: 中等

Batch Size = 8:
  - DWT总耗时: ~28ms (8 × 3.5ms)
  - 占训练时间: ~3%
  - 影响等级: 轻微
```

## 优化决策树

```
是否需要立即优化？
│
├─ 是 → 如果满足以下任一条件:
│   ├─ 使用 batch_size ≥ 32
│   ├─ 需要在线/实时推理
│   ├─ GPU资源紧张 (非A6000级别)
│   └─ 长期训练 (>3天)
│
└─ 否 → 如果满足以下所有条件:
    ├─ 使用 batch_size ≤ 8
    ├─ 仅离线训练/评估
    ├─ GPU资源充足
    └─ 短期实验 (<1天)
```

## 推荐优化方案

### 方案1: LRU缓存 (ROI最高) ⭐⭐⭐⭐⭐

**适用场景**: 训练阶段，数据有重复模式

**实现复杂度**: ⭐⭐ (简单)

**预期收益**:
- 训练阶段: 30-50% 性能提升
- 推理阶段: 5-10% 性能提升

**实现要点**:
```python
from functools import lru_cache
import hashlib

class DWTPromptGenerator(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.cache = {}  # 或使用 lru_cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _compute_hash(self, x):
        # 基于统计特征的快速哈希
        stats = torch.cat([
            x.mean(dim=-1),
            x.std(dim=-1),
            x.min(dim=-1)[0],
            x.max(dim=-1)[0]
        ]).cpu().numpy()
        return hashlib.md5(stats.tobytes()).hexdigest()
    
    def forward(self, x_enc):
        cache_key = self._compute_hash(x_enc)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # 正常计算
        features = self._compute_features(x_enc)
        self.cache[cache_key] = features
        self.cache_misses += 1
        return features
```

### 方案2: 自适应层数 ⭐⭐⭐⭐

**适用场景**: 可以牺牲少量精度换取速度

**预期收益**: 20-40% 性能提升

**实现要点**:
```python
def __init__(self, ...):
    self.adaptive_level = True
    
def _get_adaptive_level(self, seq_len):
    if seq_len < 200:
        return 2
    elif seq_len < 500:
        return 3
    else:
        return 4
```

### 方案3: 批量处理优化 ⭐⭐⭐

**修改点**: TimeLLM.py 的调用方式

**当前问题**:
```python
# 低效: 循环调用B次
for b in range(B):
    dwt_features = self.dwt_prompt_generator(x_enc[b:b+1])  # 逐个处理
```

**优化后**:
```python
# 高效: 一次处理所有batch
dwt_features = self.dwt_prompt_generator(x_enc)  # (B, N, T)
# 返回 List[Dict] 长度为B
```

**预期收益**: 10-20% 性能提升

### 方案4: 混合精度 ⭐⭐

**适用场景**: PyTorch 1.6+，支持AMP

**预期收益**: 10-15% 性能提升

**注意事项**: 需验证精度损失

## 实施优先级

### 阶段1: 快速优化 (1-2小时)
1. ✅ 已完成: 减少GPU-CPU同步 (+8.6%)
2. 🔄 实施: LRU缓存 (预计+30-50%)
3. 🔄 实施: 批量处理优化 (预计+10-20%)

**预期累计提升**: 50-80%

### 阶段2: 深度优化 (可选，1-2天)
4. 自适应层数 (+20-40%)
5. 混合精度 (+10-15%)
6. 自定义CUDA kernel (高级)

**预期累计提升**: 100-150%

## 性能目标

### 当前: 3.5ms (batch=1)
### 目标1 (快速优化): 1.5-2.0ms (2x提升)
### 目标2 (深度优化): <1.0ms (3-4x提升)

## 决策建议

**如果你的项目**:
- ✅ **batch_size ≤ 16**: 当前性能可接受，可以先完成其他工作
- ⚠️ **batch_size = 32**: 建议实施阶段1优化
- 🔴 **batch_size ≥ 64**: 强烈建议立即优化

**如果需要在线推理**: 建议完成阶段1+2全部优化

## 监控指标

在优化过程中持续监控:
```python
# 添加到训练循环
dwt_time = 0
for batch in dataloader:
    start = time.time()
    features = dwt_generator(x_enc)
    dwt_time += time.time() - start

print(f"DWT占比: {dwt_time / total_time * 100:.2f}%")
```

## 总结

**当前状态**: 🟡 可接受但有优化空间
**优化紧迫性**: 🟡 中等 (取决于batch_size)
**最佳方案**: LRU缓存 (收益高，成本低)
**预期改进**: 50-150% 性能提升
