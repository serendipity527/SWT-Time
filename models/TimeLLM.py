from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
from layers.WaveletEmbed import WaveletPatchEmbedding  # 添加小波Patch Embedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FrequencyAnalyzer:
    """频域特征分析器（基于SWT小波系数）
    
    零冗余设计：完全复用WaveletPatchEmbedding的SWT分解结果，无额外计算开销。
    
    实现维度：
        - 维度1：信号质量评估（SNR、可预测性指数）
        - 维度2：频段成分画像（主导成分、平衡度、关联分析）
        - 维度3：频域-时域统一表达（振幅-频率、周期-频段映射、趋势一致性）
    """
    
    @staticmethod
    def analyze(coeffs, x_enc, trends, lags, level=3):
        """分析频域特征（维度1+2+3）
        
        Args:
            coeffs: (B, N, T, num_bands) - SWT小波系数
            x_enc: (B, N, T) - 原始输入序列（用于振幅分析）
            trends: (B*N,) - 趋势特征
            lags: (B*N, top_k) - 周期特征
            level: int - 小波分解层数
        
        Returns:
            dict: 频域特征字典
        """
        B, N, T, num_bands = coeffs.shape
        
        # ========== 维度1：信号质量评估 ==========
        
        # 1.1 计算各频段能量
        band_energies = torch.mean(coeffs ** 2, dim=2)  # (B, N, num_bands)
        total_energy = torch.sum(band_energies, dim=2, keepdim=True) + 1e-9
        energy_ratios = band_energies / total_energy  # (B, N, num_bands)
        
        # 1.2 SNR（信噪比）：低频信号 / 最高频噪声
        signal_power = band_energies[:, :, 0]  # cA{level} - 低频趋势
        noise_power = band_energies[:, :, 1]   # cD{level} - 最高频细节
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-9))
        snr_db = torch.clamp(snr_db, -60, 60)  # 限制范围
        
        # 1.3 可预测性指数：低频占比 × (1 - 归一化熵)
        # 熵计算
        entropy = -torch.sum(
            energy_ratios * torch.log(energy_ratios + 1e-9), 
            dim=2
        )  # (B, N)
        max_entropy = torch.log(torch.tensor(num_bands, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy  # 归一化到[0,1]
        
        low_freq_ratio = energy_ratios[:, :, 0]  # cA占比
        predictability = low_freq_ratio * (1 - normalized_entropy)  # (B, N)
        
        # ========== 维度2：频段成分画像 ==========
        
        # 2.1 主导频段（能量最大的频段）
        dominant_band_idx = torch.argmax(energy_ratios, dim=2)  # (B, N)
        
        # 2.2 成分平衡度：最大能量 / 次大能量
        sorted_energies, _ = torch.sort(energy_ratios, dim=2, descending=True)
        balance_ratio = sorted_energies[:, :, 0] / (sorted_energies[:, :, 1] + 1e-9)
        
        # 2.3 高频占比（细节能量总和）
        high_freq_ratio = torch.sum(energy_ratios[:, :, 1:], dim=2)  # (B, N)
        
        # ========== 维度3：频域-时域统一表达 ==========
        
        # 3.1 振幅-频率关系：(max-min) × 高频占比
        min_vals = torch.min(x_enc, dim=2)[0]  # (B, N)
        max_vals = torch.max(x_enc, dim=2)[0]  # (B, N)
        amplitude_freq_score = (max_vals - min_vals) * high_freq_ratio
        
        # 3.2 趋势-低频一致性
        # cA系数均值的符号 vs trends的符号
        cA_mean = torch.mean(coeffs[:, :, :, 0], dim=2)  # (B, N)
        trends_reshaped = trends.reshape(B, N)  # (B, N)
        trend_consistency = (torch.sign(cA_mean) == torch.sign(trends_reshaped)).float()
        
        # 3.3 周期-频段映射（分析主周期对应的频段）
        # lags[0]是最显著的周期
        primary_lag = lags.reshape(B, N, -1)[:, :, 0]  # (B, N)
        # 根据周期长度推断对应频段
        # 短周期(1-8) -> cD1, 中短(8-24) -> cD2, 中长(24-96) -> cD3, 长(>96) -> cA
        period_band = torch.zeros_like(primary_lag)
        period_band = torch.where(primary_lag <= 8, torch.tensor(3), period_band)  # cD1
        period_band = torch.where((primary_lag > 8) & (primary_lag <= 24), torch.tensor(2), period_band)  # cD2
        period_band = torch.where((primary_lag > 24) & (primary_lag <= 96), torch.tensor(1), period_band)  # cD3
        period_band = torch.where(primary_lag > 96, torch.tensor(0), period_band)  # cA
        
        # 周期与频段的一致性：主导频段是否与周期对应
        period_consistency = (dominant_band_idx == period_band).float()
        
        return {
            # 维度1：信号质量
            'snr_db': snr_db,  # (B, N)
            'predictability': predictability,  # (B, N)
            'entropy': normalized_entropy,  # (B, N)
            
            # 维度2：频段成分
            'energy_ratios': energy_ratios,  # (B, N, num_bands)
            'dominant_band_idx': dominant_band_idx,  # (B, N)
            'balance_ratio': balance_ratio,  # (B, N)
            'high_freq_ratio': high_freq_ratio,  # (B, N)
            'low_freq_ratio': low_freq_ratio,  # (B, N)
            
            # 维度3：频域-时域关联
            'amplitude_freq_score': amplitude_freq_score,  # (B, N)
            'trend_consistency': trend_consistency,  # (B, N)
            'period_consistency': period_consistency,  # (B, N)
            'primary_lag': primary_lag,  # (B, N)
        }
    
    @staticmethod
    def generate_description(features, var_idx=0, level=3):
        """生成自然语言描述（针对单个变量）
        
        Args:
            features: dict - analyze()返回的特征字典
            var_idx: int - 变量索引（在第二维度上）
            level: int - 小波分解层数
        
        Returns:
            str - 自然语言描述
        """
        # 提取当前变量的特征（假设batch_size=1或取第0个batch）
        snr = features['snr_db'][0, var_idx].item()
        pred_score = features['predictability'][0, var_idx].item()
        entropy = features['entropy'][0, var_idx].item()
        
        energy_dist = features['energy_ratios'][0, var_idx].cpu().numpy()
        dominant_idx = features['dominant_band_idx'][0, var_idx].item()
        balance = features['balance_ratio'][0, var_idx].item()
        high_freq = features['high_freq_ratio'][0, var_idx].item()
        low_freq = features['low_freq_ratio'][0, var_idx].item()
        
        amp_freq = features['amplitude_freq_score'][0, var_idx].item()
        trend_cons = features['trend_consistency'][0, var_idx].item()
        period_cons = features['period_consistency'][0, var_idx].item()
        primary_lag = features['primary_lag'][0, var_idx].item()
        
        # 频段名称映射
        band_names = [f"cA{level}"] + [f"cD{level-i}" for i in range(level)]
        
        desc_parts = []
        
        # ========== 维度1：信号质量描述 ==========
        quality_level = "high" if pred_score > 0.5 else "moderate" if pred_score > 0.25 else "low"
        snr_desc = "clean" if snr > 10 else "moderate" if snr > 0 else "noisy"
        
        desc_parts.append(
            f"Signal quality: {snr_desc} (SNR: {snr:.1f} dB), "
            f"{quality_level} predictability (score: {pred_score:.2f})"
        )
        
        # ========== 维度2：频段成分描述 ==========
        # 能量分布
        energy_str = ", ".join([f"{e:.1%}" for e in energy_dist])
        dominant_name = band_names[int(dominant_idx)]
        
        # 主导类型判断
        if low_freq > 0.6:
            composition = f"dominated by low-frequency trend ({dominant_name}: {energy_dist[int(dominant_idx)]:.1%})"
        elif high_freq > 0.6:
            composition = f"dominated by high-frequency details ({energy_dist[1]:.1%} + {energy_dist[2]:.1%} + {energy_dist[3]:.1%})"
        else:
            composition = f"balanced frequency components (peak at {dominant_name}: {energy_dist[int(dominant_idx)]:.1%})"
        
        # 成分平衡
        if balance > 3.0:
            balance_desc = "highly concentrated"
        elif balance > 1.5:
            balance_desc = "moderately focused"
        else:
            balance_desc = "well distributed"
        
        desc_parts.append(
            f"Frequency composition: {composition}, "
            f"energy is {balance_desc} (ratio: {balance:.1f}:1)"
        )
        
        # ========== 维度3：频域-时域关联描述 ==========
        # 趋势一致性
        if trend_cons > 0.5:
            trend_desc = "the trend is confirmed by low-frequency components"
        else:
            trend_desc = "the trend may be disrupted by high-frequency noise"
        
        # 周期-频段映射
        if period_cons > 0.5:
            period_desc = f"the {int(primary_lag)}-step periodicity aligns with {dominant_name} band"
        else:
            period_desc = f"the {int(primary_lag)}-step periodicity shows weak frequency alignment"
        
        # 振幅-频率关系
        if amp_freq > 1.0:
            volatility_desc = "high volatility with frequent rapid changes"
        elif amp_freq > 0.3:
            volatility_desc = "moderate fluctuations"
        else:
            volatility_desc = "stable with smooth variations"
        
        desc_parts.append(
            f"Pattern-frequency alignment: {trend_desc}; {period_desc}; "
            f"{volatility_desc}"
        )
        
        return "; ".join(desc_parts)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class WaveletHead(nn.Module):
    """小波系数预测头 - 对称小波域输出
    
    将LLM隐状态投影到小波系数空间，然后通过ISWT重构回时域信号。
    与WaveletPatchEmbedding形成完整的"小波编码-LLM处理-小波解码"架构。
    
    架构优势：
    1. 对称设计：编码(SWT) ↔ 解码(ISWT)
    2. 频域约束：保证输出符合小波理论和频谱特性
    3. 多尺度预测：LLM分别学习不同频段（趋势、细节）
    4. 可解释性：可以分析各频段的预测质量
    
    Args:
        n_vars: 变量数量
        d_model: LLM隐状态维度
        patch_nums: patch数量
        pred_len: 预测长度
        level: 小波分解层数（需与编码器一致）
        wavelet: 小波基函数（需与编码器一致）
        head_dropout: dropout率
    
    Input:
        x: (B, N, d_model, patch_nums) - LLM处理后的隐状态
    
    Output:
        pred: (B, N, pred_len) - 预测的时域信号
    
    工作流程：
        LLM隐状态 (B, N, d_model, patch_nums)
          ↓ 为每个频段独立投影
        小波系数预测:
          - cA_pred: (B, N, pred_len) 低频趋势
          - cD3_pred: (B, N, pred_len) 高频细节
          - cD2_pred: (B, N, pred_len) 中频细节
          - cD1_pred: (B, N, pred_len) 低频细节
          ↓ Stack到频段维度
        (B, N, pred_len, Level+1)
          ↓ ISWT重构
        (B, N, pred_len)  ← 最终时域预测
    """
    
    def __init__(self, n_vars, d_model, patch_nums, pred_len, 
                 level=3, wavelet='db4', head_dropout=0, use_band_attention=True,
                 return_coeffs=False):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.level = level
        self.num_bands = level + 1  # 频段数量
        self.wavelet = wavelet
        self.d_model = d_model  # 单频段的特征维度
        self.patch_nums = patch_nums
        self.use_band_attention = use_band_attention  # 是否启用频段注意力
        self.return_coeffs = return_coeffs  # 是否返回小波系数（用于损失计算）
        
        # 计算小波滤波器长度（用于边界处理）
        import pywt
        try:
            wavelet_obj = pywt.Wavelet(wavelet)
            self.filter_len = wavelet_obj.dec_len  # 分解滤波器长度
        except Exception as e:
            # 如果pywt不支持该小波，使用默认值
            self.filter_len = 8  # db4的滤波器长度
            print(f"⚠️ 无法获取{wavelet}的滤波器长度，使用默认值{self.filter_len}. 错误: {e}")
        
        # ===== 编码-解码对称设计 =====
        # 为每个频段创建独立的投影层
        # 输入：每个频段独立的 (B, N, d_model, patch_nums)
        # 输出：每个频段独立的 (B, N, pred_len)
        # 这与编码端的频段独立处理完全对称
        self.band_projections = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=-2),  # (B, N, d_model, patch_nums) -> (B, N, d_model*patch_nums)
                nn.Linear(d_model * patch_nums, pred_len),  # -> (B, N, pred_len)
                nn.Dropout(head_dropout)
            )
            for _ in range(self.num_bands)
        ])
        
        # ISWT重构模块
        from layers.WaveletEmbed import ISWTReconstruction
        self.iswt = ISWTReconstruction(wavelet=wavelet, level=level)
        
        # ===== 频段注意力机制（方案2：与编码端对称）=====
        # 可通过use_band_attention参数控制是否启用
        if self.use_band_attention:
            # 为解码端也添加频段重要性建模，与WaveletPatchEmbedding保持一致
            # 初始化策略：与编码端相同的经验权重
            initial_weights = torch.tensor([1.5, 0.3, 1.0, 0.8], dtype=torch.float32)
            self.band_attention_logits = nn.Parameter(
                torch.log(initial_weights + 1e-8)
            )
            
            # 是否打印权重（调试用）
            self.print_band_weights = True
            self._band_weights_printed = False
        else:
            # 不使用频段注意力
            self.band_attention_logits = None
        
        print(f"[WaveletHead] 创建小波输出头：{self.num_bands}个频段投影层 + ISWT重构")
        print(f"  - 小波类型: {wavelet}")
        print(f"  - 分解层数: {level}")
        print(f"  - 滤波器长度: {self.filter_len} (用于边界重构)")
        print("  - 边界优化: ✅ 历史拼接法 (消除边界伪影)")
        print(f"  - 输入维度: {self.num_bands}*{d_model}={self.num_bands*d_model} (频段独立)")
        print("  - 架构设计: ✅ 编码-解码完全对称")
        if self.use_band_attention:
            print("  - 频段注意力: ✅ 可学习的频段重要性权重")
        else:
            print("  - 频段注意力: ⚪ 未启用（所有频段平等对待）")
        print(f"  - 每个频段参数量: {d_model * patch_nums * pred_len:,}")
        print(f"  - 总参数量: {d_model * patch_nums * pred_len * self.num_bands:,}")
    
    def get_band_weights(self) -> torch.Tensor:
        """获取归一化的频段权重（与WaveletPatchEmbedding对称）
        
        通过softmax将可学习的logits转换为权重分布。
        如果未启用频段注意力，返回均匀权重。
        
        Returns:
            weights: (num_bands,) - 归一化的频段权重
                     顺序: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        
        设计理念：
            - 与编码端保持一致，增强重要频段、抑制噪声频段
            - 编码端加权提取特征，解码端加权组合预测
            - 端到端训练，两端权重协同优化
        """
        if self.use_band_attention and self.band_attention_logits is not None:
            # 启用频段注意力：使用可学习的权重
            weights = torch.softmax(self.band_attention_logits, dim=0)
        else:
            # 未启用：返回均匀权重（所有频段平等对待）
            device = next(self.parameters()).device
            weights = torch.ones(self.num_bands, device=device) / self.num_bands
        return weights
    
    def forward(self, x, history_coeffs=None):
        """
        Args:
            x: (B, N, num_bands*d_model, patch_nums) - LLM隐状态（频段独立）
               注意：输入维度是 num_bands*d_model，包含所有频段的独立特征
            history_coeffs: (B, N, T, num_bands) - 历史序列的小波系数（可选）
                           用于边界重构优化，消除ISWT的边界伪影
        
        Returns:
            如果 self.return_coeffs=False（默认）:
                pred: (B, N, pred_len) - 预测的时域信号
            如果 self.return_coeffs=True（用于损失计算）:
                (pred, wavelet_coeffs_weighted): 
                    - pred: (B, N, pred_len) - 预测的时域信号
                    - wavelet_coeffs_weighted: (B, N, pred_len, num_bands) - 加权后的小波系数
        
        编码-解码对称性：
            编码端：4频段独立 → 4*d_model维特征
            解码端：4*d_model维特征 → split回4频段 → 独立预测
            ✅ 信息流畅无损
        
        边界优化原理：
            ISWT是基于卷积滤波器的重构，在t=0处需要访问历史数据。
            通过拼接历史小波系数到预测系数前，为滤波器提供完整的边界上下文。
        """
        B, N, total_d_model, patch_nums = x.shape
        
        # ===== 验证输入维度 =====
        expected_d_model = self.num_bands * self.d_model
        if total_d_model != expected_d_model:
            raise ValueError(
                f"输入维度不匹配！期望: {expected_d_model} "
                f"(num_bands={self.num_bands} × d_model={self.d_model}), "
                f"实际: {total_d_model}"
            )
        
        # ===== Step 1: 分离频段特征（对称解码）=====
        # 将拼接的特征分离回独立频段
        # (B, N, num_bands*d_model, patch_nums) 
        # -> num_bands个(B, N, d_model, patch_nums)
        x_bands = torch.split(x, self.d_model, dim=2)
        # 例如：(B, N, 128, patches) -> 4个(B, N, 32, patches)
        
        # ===== Step 2: 每个频段独立预测小波系数 =====
        wavelet_coeffs = []
        for i, (proj, x_band) in enumerate(zip(self.band_projections, x_bands)):
            # 每个投影层处理对应频段的独立特征
            coeff = proj(x_band)  # (B, N, pred_len)
            wavelet_coeffs.append(coeff)
        
        # ===== Step 3: Stack到频段维度 =====
        # [(B, N, pred_len)] * num_bands -> (B, N, pred_len, num_bands)
        # 顺序: [cA_pred, cD_n_pred, cD_{n-1}_pred, ..., cD_1_pred]
        wavelet_coeffs = torch.stack(wavelet_coeffs, dim=-1)
        
        # ===== Step 3.5: 频段重要性加权（方案2：与编码端对称）=====
        # 在ISWT重构之前，对预测的小波系数进行加权
        # 这与编码端的频段加权形成对称设计
        
        # 获取可学习的频段权重 (num_bands,) = (4,)
        band_weights = self.get_band_weights()  # softmax归一化后的权重
        
        # 可视化频段权重（仅在第一次调用时打印）
        if self.use_band_attention and hasattr(self, 'print_band_weights') and \
           self.print_band_weights and not self._band_weights_printed:
            band_names = [f"cA{self.level}"] + [f"cD{self.level-i+1}" for i in range(1, self.num_bands)]
            print(f"\n[WaveletHead] 频段注意力权重:")
            # 修复：bfloat16不支持numpy，需要先转float32
            for name, weight in zip(band_names, band_weights.detach().float().cpu().numpy()):
                print(f"  - {name}: {weight:.4f}")
            self._band_weights_printed = True
        
        # 应用频段权重到小波系数
        # wavelet_coeffs: (B, N, pred_len, num_bands)
        # band_weights: (num_bands,)
        # 广播到 (1, 1, 1, num_bands) 以进行逐频段缩放
        band_weights_expanded = band_weights.view(1, 1, 1, self.num_bands)
        wavelet_coeffs_weighted = wavelet_coeffs * band_weights_expanded
        # (B, N, pred_len, num_bands)
        # 现在重要频段（如cA3）的系数被放大，噪声频段（如cD3）被抑制
        
        # 对称性确认：
        # 编码端输出：(B*N, patches, 4*d_model)  - 4个频段独立
        # 解码端处理：split成4个频段 → 独立预测 → stack  ✅ 完全对称
        
        # ===== Step 4: 边界优化 - 拼接历史小波系数 =====
        if history_coeffs is not None:
            # 取历史序列的最后filter_len个点作为边界上下文
            # 这些点提供了ISWT滤波器在边界处所需的历史信息
            history_len = history_coeffs.shape[2]
            if history_len >= self.filter_len:
                history_suffix = history_coeffs[:, :, -self.filter_len:, :]  
                # (B, N, filter_len, num_bands)
            else:
                # 如果历史长度不足，用全部历史数据
                history_suffix = history_coeffs
                print(f"⚠️ 历史长度{history_len}小于滤波器长度{self.filter_len}，使用全部历史数据")
            
            # 拼接：[历史尾部 | 加权后的预测系数]
            # 注意：这里使用加权后的系数，保证频段重要性在整个流程中一致
            coeffs_with_context = torch.cat([history_suffix, wavelet_coeffs_weighted], dim=2)
            # (B, N, filter_len + pred_len, num_bands)
            
            # ISWT重构（包含边界上下文）
            pred_with_context = self.iswt(coeffs_with_context)
            # (B, N, filter_len + pred_len)
            
            # 裁剪掉历史部分，只保留预测部分
            pred = pred_with_context[:, :, history_suffix.shape[2]:]
            # (B, N, pred_len)
        else:
            # 降级方案：如果没有历史数据，直接重构（保持向后兼容）
            # 注意：这种情况下边界可能存在伪影
            # 使用加权后的系数
            pred = self.iswt(wavelet_coeffs_weighted)
            # 只在第一次遇到时警告，避免日志刷屏
            if not hasattr(self, '_warned_no_history'):
                print("⚠️ [WaveletHead] 未提供历史小波系数，边界可能有伪影")
                print("   建议：在调用时传递history_coeffs参数以获得最佳性能")
                self._warned_no_history = True
        
        # 根据配置返回
        if self.return_coeffs:
            # 返回时域预测 + 小波系数（用于损失计算）
            return pred, wavelet_coeffs_weighted
        else:
            # 仅返回时域预测（默认行为）
            return pred


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)
        
        # ===== 频域Prompt开关（默认关闭）=====
        self.use_freq_prompt = getattr(configs, 'use_freq_prompt', False)
        if self.use_freq_prompt:
            print("[TimeLLM] ✅ 频域Prompt增强已启用（维度1+2+3）")
        else:
            print("[TimeLLM] ⚪ 频域Prompt增强未启用")

        # 使用WaveletPatchEmbedding替代原始PatchEmbedding
        # 可以通过configs.use_wavelet控制是否启用（默认启用）
        use_wavelet = getattr(configs, 'use_wavelet', True)
        
        if use_wavelet:
            # 小波Patch Embedding：先SWT分解，再Patching
            swt_level = getattr(configs, 'swt_level', 3)
            num_bands = swt_level + 1  # 频段数量
            
            # 是否启用频段注意力机制（默认启用）
            use_band_attention = getattr(configs, 'use_band_attention', True)
            
            self.patch_embedding = WaveletPatchEmbedding(
                d_model=configs.d_model,
                patch_len=self.patch_len,
                stride=self.stride,
                wavelet=getattr(configs, 'wavelet', 'db4'),  # 默认db4小波
                level=swt_level,      # 默认3层分解
                dropout=configs.dropout,
                use_band_attention=use_band_attention  # 频段注意力开关
            )
            
            # ===== 关键修复：计算实际的embedding维度 =====
            # WaveletPatchEmbedding输出: num_bands * d_model (例如：4×32=128)
            # 而不是原来的 d_model (32)
            self.patch_embedding_dim = num_bands * configs.d_model
            
            print(f"[TimeLLM] 使用 WaveletPatchEmbedding (小波={getattr(configs, 'wavelet', 'db4')}, 层数={swt_level})")
            print(f"  - Embedding输出维度: {self.patch_embedding_dim} ({num_bands}频段 × {configs.d_model}维)")
        else:
            # 原始Patch Embedding
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)
            self.patch_embedding_dim = configs.d_model  # 原始维度
            print("[TimeLLM] 使用 原始 PatchEmbedding")

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # ===== 关键修复：使用正确的embedding维度 =====
        # ReprogrammingLayer需要匹配patch_embedding的实际输出维度
        self.reprogramming_layer = ReprogrammingLayer(
            self.patch_embedding_dim,  # 使用实际维度（可能是128而不是32）
            configs.n_heads, 
            d_keys=None,  # 让它自动计算 d_model // n_heads
            d_llm=self.d_llm
        )
        print(f"[TimeLLM] ReprogrammingLayer 输入维度: {self.patch_embedding_dim}")

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 可通过配置选择输出头类型
            use_wavelet_head = getattr(configs, 'use_wavelet_head', False)
            
            # ===== 关键修复：添加 Output Adapter =====
            # 解决信息瓶颈：使用线性层从d_llm投影到目标维度，而不是直接切片
            # 这允许模型从冻结的LLM中提取有效信息，并将其组织成所需的格式
            
            if use_wavelet_head:
                # 小波系数输出头：对称小波域架构
                # LLM隐状态 -> 小波系数预测 -> ISWT重构 -> 时域预测
                
                # 计算目标维度：所有频段的总维度
                swt_level = getattr(configs, 'swt_level', 3)
                num_bands = swt_level + 1
                self.dec_out_dim = num_bands * configs.d_model
                
                # 是否启用频段注意力机制（默认启用，与编码端保持一致）
                use_band_attention = getattr(configs, 'use_band_attention', True)
                
                # 是否返回小波系数（用于混合损失函数）
                return_coeffs = getattr(configs, 'use_wavelet_loss', False)
                
                self.output_projection = WaveletHead(
                    n_vars=configs.enc_in,
                    d_model=configs.d_model,  # 使用单频段维度
                    patch_nums=self.patch_nums,
                    pred_len=self.pred_len,
                    level=swt_level,
                    wavelet=getattr(configs, 'wavelet', 'db4'),
                    head_dropout=configs.dropout,
                    use_band_attention=use_band_attention,  # 频段注意力开关
                    return_coeffs=return_coeffs  # 是否返回小波系数
                )
                print("[TimeLLM] 使用 WaveletHead 输出层")
                print(f"  - 架构: LLM隐状态 → 小波系数({num_bands}频段) → ISWT重构 → 时域预测")
                print(f"  - Output Adapter目标维度: {self.dec_out_dim}")
            else:
                # 原始线性输出头：直接时域映射
                # LLM隐状态 -> 线性层 -> 时域预测
                self.dec_out_dim = self.d_ff  # 保持与原始逻辑一致，投影到d_ff
                
                self.output_projection = FlattenHead(
                    configs.enc_in, self.head_nf, self.pred_len,
                    head_dropout=configs.dropout
                )
                print("[TimeLLM] 使用 FlattenHead 输出层（直接时域映射）")
                print(f"  - Output Adapter目标维度: {self.dec_out_dim}")
            
            # 创建输出适配器：d_llm -> dec_out_dim
            self.output_adapter = nn.Linear(self.d_llm, self.dec_out_dim)
            # 初始化适配器权重，使其更易于训练
            nn.init.xavier_normal_(self.output_adapter.weight)
            
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 检查是否返回了小波系数
            if isinstance(output, tuple):
                dec_out, pred_coeffs = output
                return dec_out[:, -self.pred_len:, :], pred_coeffs
            else:
                dec_out = output
                return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        
        # ===== 频域特征分析（如果启用）=====
        freq_features = None
        if self.use_freq_prompt and hasattr(self.patch_embedding, 'swt'):
            # 复用patch_embedding的SWT模块提取小波系数
            with torch.no_grad():
                # 转换数据形状: (B*N, T, 1) -> (B, N, T)
                x_for_swt = x_enc.reshape(B, N, T)
                # 调用SWT分解
                coeffs = self.patch_embedding.swt(x_for_swt.to(torch.bfloat16))
                # coeffs: (B, N, T, num_bands)
                
                # 提取频域特征
                swt_level = getattr(self.patch_embedding, 'level', 3)
                freq_features = FrequencyAnalyzer.analyze(
                    coeffs=coeffs,
                    x_enc=x_for_swt,
                    trends=trends,
                    lags=lags,
                    level=swt_level
                )

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            # ===== 生成频域描述（如果启用）=====
            freq_desc = ""
            if freq_features is not None:
                # 当前样本的变量索引
                var_idx = b % N
                # 生成自然语言描述
                swt_level = getattr(self.patch_embedding, 'level', 3)
                freq_analysis = FrequencyAnalyzer.generate_description(
                    freq_features, var_idx=var_idx, level=swt_level
                )
                freq_desc = f"Frequency analysis: {freq_analysis}; "
            
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}; "
                f"{freq_desc}"
                f"<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()  # (B, N, T)
        
        # ===== 边界优化：提取历史小波系数 =====
        # 如果输出层是WaveletHead，提前对历史序列进行SWT分解
        # 这些系数将用于ISWT重构时的边界处理，消除边界伪影
        history_coeffs = None
        if isinstance(self.output_projection, WaveletHead):
            # 复用patch_embedding中的SWT模块提取历史序列的小波系数
            # 注意：这里直接调用swt模块，不经过patching过程
            if hasattr(self.patch_embedding, 'swt'):
                history_coeffs = self.patch_embedding.swt(x_enc.to(torch.bfloat16))
                # history_coeffs: (B, N, T, num_bands)
                # 这些系数包含了历史序列的多尺度频域信息
            else:
                print("⚠️ patch_embedding没有swt属性，无法提取历史小波系数")
        # =========================================
        
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        
        # ===== 关键修复：使用Output Adapter =====
        # 原始代码：dec_out = dec_out[:, :, :self.d_ff] (丢弃了绝大部分信息)
        # 新代码：投影到目标维度
        dec_out = self.output_adapter(dec_out)
        # =====================================

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # ===== 修改：传递历史系数到WaveletHead + 支持返回小波系数 =====
        pred_coeffs = None  # 用于存储预测的小波系数
        
        if isinstance(self.output_projection, WaveletHead):
            # WaveletHead使用历史系数进行边界优化
            output = self.output_projection(
                dec_out[:, :, :, -self.patch_nums:], 
                history_coeffs=history_coeffs
            )
            # 检查是否返回了小波系数
            if isinstance(output, tuple):
                dec_out, pred_coeffs = output  # (B, N, pred_len), (B, N, pred_len, num_bands)
            else:
                dec_out = output  # (B, N, pred_len)
        else:
            # FlattenHead或其他输出层，保持原有调用方式
            dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        # ============================================
        
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        # 如果有小波系数，一起返回
        if pred_coeffs is not None:
            return dec_out, pred_coeffs
        else:
            return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
