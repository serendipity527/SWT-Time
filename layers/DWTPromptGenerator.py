"""
DWT-based Dynamic Prompt Generator (方案B优化版)
使用离散小波变换(DWT)提取多尺度特征用于动态prompt生成
"""

import torch
import torch.nn as nn
import numpy as np
import ptwt


class DWTPromptGenerator(nn.Module):
    """
    基于DWT的动态Prompt生成器 (方案B)
    
    核心功能:
    1. 频域特征: 能量分布 + 熵值 → 序列类型分类
    2. 趋势特征: 多尺度趋势一致性 → 趋势描述
    3. 信号质量: SNR计算 → 稳定性评估
    4. 预测难度: 综合评分 → 难度指示
    """
    
    def __init__(self, 
                 wavelet='db4', 
                 level=3,
                 compression_level='balanced',
                 use_adaptive_thresholds=False):
        """
        Args:
            wavelet: 小波基，默认'db4'
            level: DWT分解层数，默认3
            compression_level: prompt压缩级别 ('minimal'|'balanced'|'detailed')
            use_adaptive_thresholds: 是否使用自适应阈值（暂未实现，预留接口）
            
        Raises:
            ValueError: 当参数值无效时
            TypeError: 当参数类型不正确时
        """
        super(DWTPromptGenerator, self).__init__()
        
        # 验证wavelet参数
        if not isinstance(wavelet, str) or not wavelet.strip():
            raise ValueError("wavelet must be a non-empty string")
        
        # 验证level参数
        if not isinstance(level, int) or level < 1 or level > 10:
            raise ValueError(f"level must be an integer between 1 and 10, got {level}")
        
        # 验证compression_level参数
        valid_compressions = ['minimal', 'balanced', 'detailed']
        if compression_level not in valid_compressions:
            raise ValueError(f"compression_level must be one of {valid_compressions}, got '{compression_level}'")
        
        # 验证use_adaptive_thresholds参数
        if not isinstance(use_adaptive_thresholds, bool):
            raise TypeError(f"use_adaptive_thresholds must be a boolean, got {type(use_adaptive_thresholds)}")
        
        self.wavelet = wavelet
        self.level = level
        self.compression = compression_level
        self.use_adaptive = use_adaptive_thresholds
        
        # 固定阈值配置（基础版本）
        self.thresholds = {
            # 频域特征阈值
            'energy_entropy_low': 0.5,      # 单一尺度主导
            'energy_entropy_high': 1.0,     # 多尺度复杂
            'dominant_energy_high': 0.7,    # 明显主导频段
            
            # 趋势一致性阈值
            'trend_consistency_high': 0.75, # 大部分一致
            
            # 信噪比阈值 (dB)
            'snr_high': 20,                 # 高信噪比
            'snr_mid': 10,                  # 中等信噪比
            
            # 预测难度阈值
            'difficulty_low': 15,           # 低难度
            'difficulty_high': 30           # 高难度
        }
        
        # 频段名称映射（预定义模板，但会根据level动态生成）
        self.band_name_templates = {
            'basic': ['trend', 'seasonal', 'fluctuation', 'noise'],
            'descriptive': ['long-term', 'seasonal', 'medium-term', 'short-term'],
            'technical': ['low-freq', 'mid-low-freq', 'mid-high-freq', 'high-freq']
        }
    
    def _get_band_names(self, level, style='basic'):
        """
        根据分解层数动态生成频段名称
        
        Args:
            level: DWT分解层数
            style: 命名风格 ('basic'|'descriptive'|'technical')
            
        Returns:
            list: 频段名称列表，长度为level+1
        """
        template = self.band_name_templates.get(style, self.band_name_templates['basic'])
        num_bands = level + 1  # cA + level个cD
        
        if num_bands <= len(template):
            # 模板足够，直接取前num_bands个
            return template[:num_bands]
        else:
            # 需要扩展模板
            result = template.copy()
            for i in range(len(template), num_bands):
                if style == 'basic':
                    result.append(f"band{i}")
                elif style == 'descriptive':
                    result.append(f"scale{i}")
                else:  # technical
                    result.append(f"freq{i}")
            return result
    
    def forward(self, x_enc):
        """
        提取DWT特征并生成prompt描述
        
        Args:
            x_enc: (B, N, T) 输入序列
            
        Returns:
            features: dict, 包含所有特征的字典
            
        Raises:
            ValueError: 当输入维度不正确或包含无效数值时
            RuntimeError: 当序列长度不足以进行DWT分解时
        """
        # 输入验证
        if not torch.is_tensor(x_enc):
            raise TypeError(f"Expected torch.Tensor, got {type(x_enc)}")
        
        if x_enc.dim() != 3:
            raise ValueError(f"Expected 3D tensor (B, N, T), got {x_enc.dim()}D tensor with shape {x_enc.shape}")
        
        B, N, T = x_enc.shape
        
        # 检查序列长度是否足够进行DWT分解
        min_length = 2 ** self.level  # 最小长度要求
        if T < min_length:
            raise RuntimeError(f"Sequence length {T} is too short for DWT level {self.level}. "
                              f"Minimum required length: {min_length}")
        
        # 检查数值有效性
        if torch.isnan(x_enc).any():
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(x_enc).any():
            raise ValueError("Input contains infinite values")
        
        # 检查是否为空张量
        if x_enc.numel() == 0:
            raise ValueError("Input tensor is empty")
        
        # 获取输入设备信息，确保后续计算的设备一致性
        device = x_enc.device
        dtype = x_enc.dtype
        
        # 重塑为 (B*N, 1, T) 进行批量DWT
        try:
            x_reshaped = x_enc.reshape(B * N, 1, T).float()
            # 确保在相同设备上
            x_reshaped = x_reshaped.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to reshape input tensor: {e}")
        
        # DWT分解
        try:
            coeffs = ptwt.wavedec(x_reshaped, self.wavelet, level=self.level, mode='reflect')
            # coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1]
            # 对于level=3: [cA3, cD3, cD2, cD1]
        except Exception as e:
            raise RuntimeError(f"DWT decomposition failed with wavelet '{self.wavelet}' and level {self.level}: {e}")
        
        # 验证DWT结果
        if not coeffs or len(coeffs) != (self.level + 1):
            raise RuntimeError(f"DWT returned unexpected number of coefficients: expected {self.level + 1}, got {len(coeffs)}")
        
        # 检查系数有效性
        for i, coeff in enumerate(coeffs):
            if torch.isnan(coeff).any() or torch.isinf(coeff).any():
                raise ValueError(f"DWT coefficient {i} contains invalid values (NaN or Inf)")
            if coeff.numel() == 0:
                raise ValueError(f"DWT coefficient {i} is empty")
        
        # 提取四大特征组
        freq_features = self._extract_frequency_features(coeffs)
        trend_features = self._extract_trend_features(coeffs)
        quality_features = self._extract_quality_features(coeffs)
        difficulty = self._calculate_difficulty(freq_features, trend_features, quality_features)
        
        # 合并所有特征
        features = {
            **freq_features,
            **trend_features,
            **quality_features,
            'difficulty': difficulty
        }
        
        return features
    
    def _extract_frequency_features(self, coeffs):
        """
        特征组1: 频域特征
        
        提取:
        - 能量分布占比
        - 信息熵
        - 主导频段
        
        Raises:
            ValueError: 当系数无效或计算出现数值问题时
        """
        try:
            # 计算各频段能量
            energies = []
            for i, c in enumerate(coeffs):
                energy = torch.sum(c ** 2, dim=-1)
                if torch.isnan(energy).any() or torch.isinf(energy).any():
                    raise ValueError(f"Invalid energy computed for coefficient {i}")
                energies.append(energy)
            
            total_energy = sum(energies)
            
            # 检查总能量有效性
            if torch.isnan(total_energy).any() or torch.isinf(total_energy).any():
                raise ValueError("Total energy contains invalid values")
            
            if (total_energy == 0).any():
                # 处理零能量情况：使用均匀分布
                energy_ratio = torch.ones(len(coeffs), energies[0].shape[0]) / len(coeffs)
                energy_ratio = energy_ratio.to(energies[0].device)
            else:
                total_energy = torch.clamp(total_energy, min=1e-10)  # 更强的数值保护
                # 能量占比
                energy_ratio = torch.stack([e / total_energy for e in energies], dim=0)
            
            # (level+1, B*N)
            
            # 计算信息熵
            energy_ratio_safe = torch.clamp(energy_ratio, min=1e-10)  # 避免log(0)
            energy_entropy = -torch.sum(energy_ratio_safe * torch.log(energy_ratio_safe), dim=0)
            
            # 安全地计算均值
            if energy_entropy.numel() == 0:
                energy_entropy_mean = 0.0
            else:
                energy_entropy_mean = torch.clamp(energy_entropy, min=0).mean().item()
        
        except Exception as e:
            raise ValueError(f"Error in frequency feature extraction: {e}")
        
        # 主导频段
        try:
            dominant_idx = torch.argmax(energy_ratio, dim=0)  # (B*N,)
            # 统计众数作为整体主导频段（使用bincount更可靠）
            if dominant_idx.numel() > 0:
                # 确保索引在有效范围内
                valid_indices = torch.clamp(dominant_idx.flatten(), 0, len(coeffs) - 1)
                counts = torch.bincount(valid_indices, minlength=len(coeffs))
                if counts.sum() > 0:
                    dominant_band_idx = torch.argmax(counts).item()
                else:
                    dominant_band_idx = 0
            else:
                dominant_band_idx = 0
            
            # 安全获取主导能量
            dominant_band_idx = min(dominant_band_idx, len(energy_ratio) - 1)
            dominant_energy = energy_ratio[dominant_band_idx].mean().item()
            
        except Exception as e:
            # 回退方案：使用第一个频段作为主导
            dominant_band_idx = 0
            dominant_energy = energy_ratio[0].mean().item() if energy_ratio.numel() > 0 else 0.0
        
        # 语义映射
        freq_desc = self._map_frequency_pattern(
            energy_entropy_mean, 
            dominant_band_idx, 
            dominant_energy
        )
        
        # 统一返回值类型为numpy数组（保持CPU上的浮点数）
        return {
            'energy_ratio': energy_ratio.mean(dim=1).detach().cpu().numpy().astype(np.float32),  # (level+1,)
            'energy_entropy': float(energy_entropy_mean),
            'dominant_band': int(dominant_band_idx),
            'dominant_energy': float(dominant_energy),
            'freq_pattern': freq_desc  # 字符串保持不变
        }
    
    def _extract_trend_features(self, coeffs):
        """
        特征组2: 趋势特征
        
        提取:
        - 各频段趋势方向
        - 趋势一致性分数
        - 趋势语义描述
        
        Raises:
            ValueError: 当系数无效或趋势计算出现问题时
        """
        try:
            # 计算各频段的趋势（一阶差分平均变化率）
            trends = []
            for i, c in enumerate(coeffs):
                # c: (B*N, 1, length)
                if c.shape[-1] < 2:
                    raise ValueError(f"Coefficient {i} too short for trend calculation (length: {c.shape[-1]})")
                
                diff = c[..., 1:] - c[..., :-1]
                
                # 检查差分结果的有效性
                if torch.isnan(diff).any() or torch.isinf(diff).any():
                    raise ValueError(f"Invalid trend computed for coefficient {i}")
                
                if diff.numel() == 0:
                    trend_mean = 0.0
                else:
                    trend_mean = diff.mean().item()  # 使用平均变化率，确保跨尺度可比较
                
                trends.append(trend_mean)
        
        except Exception as e:
            raise ValueError(f"Error in trend feature extraction: {e}")
        
        # 趋势方向标签
        trends_sign = [1 if t > 0 else (-1 if t < 0 else 0) for t in trends]
        
        # 计算一致性分数
        if trends_sign[0] != 0:
            consistency_score = sum([1 for s in trends_sign if s == trends_sign[0]]) / len(trends_sign)
        else:
            consistency_score = 0.5  # 主趋势为0，认为中等一致
        
        # 语义映射
        trend_desc = self._map_trend_pattern(trends_sign, consistency_score)
        
        # 统一返回值类型
        return {
            'trends': np.array(trends, dtype=np.float32),
            'trends_sign': np.array(trends_sign, dtype=np.int32),
            'trend_consistency': float(consistency_score),
            'trend_desc': trend_desc  # 字符串保持不变
        }
    
    def _extract_quality_features(self, coeffs):
        """
        特征组3: 信号质量
        
        提取:
        - 4个频段标准差 [std_cA, std_cD3, std_cD2, std_cD1]
        - 波动性层级分析
        - 质量语义描述
        
        Raises:
            ValueError: 当系数无效或质量计算出现问题时
        """
        try:
            # 计算各频段标准差 (按架构设计)
            stds = []
            for i, c in enumerate(coeffs):
                # c: (B*N, 1, length) -> 计算所有批次和变量的平均标准差
                if c.numel() == 0:
                    raise ValueError(f"Coefficient {i} is empty")
                
                std_value = torch.std(c).item()
                
                # 检查标准差有效性
                if np.isnan(std_value) or np.isinf(std_value):
                    raise ValueError(f"Invalid standard deviation for coefficient {i}: {std_value}")
                
                # 确保非负
                std_value = max(0.0, std_value)
                stds.append(std_value)
            
            if len(stds) == 0:
                raise ValueError("No valid standard deviations computed")
            
            # 传统SNR计算 (cA作为信号, cD1作为噪声)
            signal_std = stds[0]  # cA
            noise_std = stds[-1]  # cD1
            
            # 安全的SNR计算
            if signal_std == 0 and noise_std == 0:
                snr_db = 0.0  # 都为0，认为SNR为0
            elif noise_std == 0:
                snr_db = 60.0  # 无噪声，设为高SNR
            else:
                noise_std = max(noise_std, 1e-10)  # 确保分母不为0
                try:
                    snr_ratio = (signal_std ** 2) / (noise_std ** 2)
                    snr_ratio = max(snr_ratio, 1e-10)  # 避免log(0)
                    snr_db = 10 * np.log10(snr_ratio)
                    
                    # 限制SNR范围，避免极端值
                    snr_db = np.clip(snr_db, -60, 60)
                except:
                    snr_db = 0.0  # 计算失败时的默认值
        
        except Exception as e:
            raise ValueError(f"Error in quality feature extraction: {e}")
        
        # 波动性层级分析
        low_freq_std = stds[0]  # cA
        high_freq_std = stds[-1]  # cD1
        volatility_ratio = high_freq_std / (low_freq_std + 1e-10)
        
        # 频段稳定性分析
        stability_desc = self._analyze_volatility_pattern(stds)
        
        # 传统质量描述
        quality_desc = self._map_quality_pattern(snr_db)
        
        # 统一返回值类型
        return {
            'frequency_stds': np.array(stds, dtype=np.float32),  # 4个频段标准差
            'signal_std': float(signal_std),
            'noise_std': float(noise_std),
            'snr_db': float(snr_db),
            'volatility_ratio': float(volatility_ratio),
            'stability_desc': stability_desc,  # 字符串保持不变
            'signal_quality': quality_desc  # 字符串保持不变
        }
    
    def _calculate_difficulty(self, freq_feat, trend_feat, quality_feat):
        """
        特征组4: 预测难度
        
        综合考虑:
        - 趋势一致性 (低一致性 → 高难度) [0-40分]
        - 能量熵 (高熵 → 高难度) [0-35分] 
        - SNR (低SNR → 高难度) [0-25分]
        总分范围: [0-100分]
        
        Returns:
            difficulty: 'low' | 'moderate' | 'high'
        """
        try:
            # 1. 趋势不一致性贡献 [0-40分]
            trend_inconsistency = 1 - trend_feat['trend_consistency']
            trend_score = np.clip(trend_inconsistency * 40, 0, 40)
            
            # 2. 能量熵贡献 [0-35分] - 归一化熵值
            # 理论最大熵约为 ln(level+1)，这里用level=4的情况作为参考
            max_entropy = np.log(self.level + 1)
            normalized_entropy = np.clip(freq_feat['energy_entropy'] / max_entropy, 0, 1)
            entropy_score = normalized_entropy * 35
            
            # 3. SNR贡献 [0-25分] - 使用sigmoid映射
            snr_db = quality_feat['snr_db']
            # SNR在[-60, 60]范围内，使用sigmoid函数映射到[0,1]
            # 当SNR=10时，difficulty约为0.5；SNR越低，difficulty越高
            snr_normalized = 1 / (1 + np.exp((snr_db - 10) / 5))  # sigmoid
            snr_score = snr_normalized * 25
            
            # 总分计算
            difficulty_score = trend_score + entropy_score + snr_score
            
            # 确保范围在[0, 100]内
            difficulty_score = np.clip(difficulty_score, 0, 100)
            
        except Exception as e:
            # 如果计算失败，返回中等难度
            difficulty_score = 50.0
        
        # 三级分类
        if difficulty_score < self.thresholds['difficulty_low']:
            difficulty = 'low'
        elif difficulty_score < self.thresholds['difficulty_high']:
            difficulty = 'moderate'
        else:
            difficulty = 'high'
        
        return difficulty
    
    def _map_frequency_pattern(self, entropy, dominant_idx, dominant_energy):
        """将频域特征映射为语义描述"""
        
        # 基于熵值判断复杂度
        if entropy < self.thresholds['energy_entropy_low']:
            base_desc = "single-scale dominant"
        elif entropy < self.thresholds['energy_entropy_high']:
            base_desc = "dual-scale pattern"
        else:
            base_desc = "multi-scale complex"
        
        # 如果有明显主导频段，补充说明
        if dominant_energy > self.thresholds['dominant_energy_high']:
            band_map = self._get_band_names(self.level)
            if dominant_idx < len(band_map):
                base_desc += f" ({band_map[dominant_idx]})"
        
        return base_desc
    
    def _map_trend_pattern(self, trends_sign, consistency):
        """将趋势特征映射为语义描述"""
        
        if consistency == 1.0:
            # 完全一致
            direction = 'upward' if trends_sign[0] > 0 else 'downward'
            return f"consistent {direction}"
        
        elif consistency >= self.thresholds['trend_consistency_high']:
            # 大部分一致
            direction = 'upward' if trends_sign[0] > 0 else 'downward'
            # 找出不一致的频段
            inconsistent_idx = [i for i, s in enumerate(trends_sign) if s != trends_sign[0]]
            if inconsistent_idx:
                band_names = self._get_band_names(self.level, style='descriptive')
                if inconsistent_idx[0] < len(band_names):
                    return f"mostly {direction}, except {band_names[inconsistent_idx[0]]}"
                else:
                    return f"mostly {direction}, except band{inconsistent_idx[0]}"
            else:
                return f"consistent {direction}"
        
        else:
            # 多尺度混合
            up_count = sum([1 for s in trends_sign if s > 0])
            total = len(trends_sign)
            return f"mixed ({up_count}/{total} scales upward)"
    
    def _analyze_volatility_pattern(self, stds):
        """
        分析4个频段的波动性模式
        
        Args:
            stds: [std_cA, std_cD3, std_cD2, std_cD1] 4个频段的标准差
            
        Returns:
            stability_desc: 稳定性语义描述
        """
        if len(stds) != 4:
            return "insufficient frequency bands"
        
        # 低频(cA) vs 高频(cD1) 的对比
        low_freq_std = stds[0]
        high_freq_std = stds[-1]
        
        # 高频vs低频比率
        ratio = high_freq_std / (low_freq_std + 1e-10)
        
        if ratio > 5:
            base_desc = "stable trend with high short-term volatility"
        elif ratio > 2:
            base_desc = "moderate volatility across scales"
        elif ratio > 0.5:
            base_desc = "balanced multi-scale variations"
        else:
            base_desc = "trend-dominated with low noise"
        
        # 检查中间频段的分布
        mid_stds = stds[1:3]  # cD3, cD2
        mid_avg = sum(mid_stds) / len(mid_stds)
        
        # 如果中频特别突出，补充说明
        if mid_avg > max(low_freq_std, high_freq_std):
            base_desc += " (strong mid-frequency components)"
        
        return base_desc

    def _map_quality_pattern(self, snr_db):
        """将信号质量映射为语义描述"""
        
        if snr_db > self.thresholds['snr_high']:
            return "high SNR (clean)"
        elif snr_db > self.thresholds['snr_mid']:
            return "moderate SNR"
        else:
            return "low SNR (noisy)"
    
    def build_prompt_text(self, features, base_info):
        """
        构建最终的prompt文本
        
        Args:
            features: 提取的DWT特征字典
            base_info: 基础信息 {'min', 'max', 'median', 'lags', 'description', 'seq_len', 'pred_len'}
            
        Returns:
            prompt_text: str
            
        Raises:
            ValueError: 当输入参数无效时
            TypeError: 当参数类型不正确时
        """
        # 验证features参数
        if not isinstance(features, dict):
            raise TypeError(f"features must be a dictionary, got {type(features)}")
        
        required_features = ['freq_pattern', 'trend_desc', 'signal_quality', 'difficulty']
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # 验证base_info参数
        if not isinstance(base_info, dict):
            raise TypeError(f"base_info must be a dictionary, got {type(base_info)}")
        
        required_info = ['min', 'max', 'median', 'lags', 'description', 'seq_len', 'pred_len']
        missing_info = [f for f in required_info if f not in base_info]
        if missing_info:
            raise ValueError(f"Missing required base_info fields: {missing_info}")
        
        # 验证数值有效性
        for key in ['min', 'max', 'median']:
            value = base_info[key]
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                raise ValueError(f"Invalid {key} value: {value}")
        
        for key in ['seq_len', 'pred_len']:
            value = base_info[key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid {key} value: {value} (must be positive integer)")
        
        if not isinstance(base_info['lags'], (list, tuple)) or len(base_info['lags']) == 0:
            raise ValueError("lags must be a non-empty list or tuple")
        
        if not isinstance(base_info['description'], str) or not base_info['description'].strip():
            raise ValueError("description must be a non-empty string")
        
        try:
            if self.compression == 'minimal':
                return self._build_minimal_prompt(features, base_info)
            elif self.compression == 'balanced':
                return self._build_balanced_prompt(features, base_info)
            else:  # detailed
                return self._build_detailed_prompt(features, base_info)
        except Exception as e:
            raise RuntimeError(f"Failed to build prompt text: {e}")
    
    def _build_balanced_prompt(self, features, base_info):
        """构建平衡版本的prompt (推荐)"""
        
        # 提取信息
        min_val = base_info['min']
        max_val = base_info['max']
        median = base_info['median']
        lags = base_info['lags'][:3]  # 只取前3个周期
        
        # 构建prompt
        prompt = (
            f"<|start_prompt|>Dataset description: {base_info['description']}\n"
            f"Task description: forecast the next {base_info['pred_len']} steps "
            f"given the previous {base_info['seq_len']} steps information; \n"
            f"Input statistics: range [{min_val:.2f}, {max_val:.2f}], median {median:.2f}\n"
            f"Pattern analysis: {features['freq_pattern']}, "
            f"trend is {features['trend_desc']}, "
            f"signal quality is {features['signal_quality']}"
        )
        
        # 只在非moderate难度时添加难度说明
        if features['difficulty'] != 'moderate':
            prompt += f", forecast difficulty: {features['difficulty']}"
        
        prompt += f"\nDominant periodicities: {[int(l) for l in lags]}<|<end_prompt>|>"
        
        return prompt
    
    def _build_minimal_prompt(self, features, base_info):
        """构建最精简版本的prompt"""
        
        min_val = base_info['min']
        max_val = base_info['max']
        median = base_info['median']
        lags = base_info['lags'][:2]  # 只取前2个
        
        # 极简格式
        prompt = (
            f"<|start_prompt|>{base_info['description']}\n"
            f"Forecast {base_info['pred_len']} from {base_info['seq_len']}: "
            f"[{min_val:.1f},{max_val:.1f}]@{median:.1f}\n"
            f"{features['freq_pattern']}, {features['trend_desc']}, {features['signal_quality']}"
        )
        
        if features['difficulty'] == 'high':
            prompt += ", high difficulty"
        
        prompt += f"\nCycles: {[int(l) for l in lags]}<|<end_prompt>|>"
        
        return prompt
    
    def _build_detailed_prompt(self, features, base_info):
        """构建详细版本的prompt"""
        
        min_val = base_info['min']
        max_val = base_info['max']
        median = base_info['median']
        lags = base_info['lags'][:5]  # 保留前5个
        
        # 详细格式
        prompt = (
            f"<|start_prompt|>Dataset description: {base_info['description']}\n"
            f"Task description: forecast the next {base_info['pred_len']} steps "
            f"given the previous {base_info['seq_len']} steps information; \n"
            f"Input statistics:\n"
            f"  - Value range: [{min_val:.2f}, {max_val:.2f}]\n"
            f"  - Median: {median:.2f}\n"
            f"Multi-scale wavelet analysis:\n"
            f"  - Frequency pattern: {features['freq_pattern']}\n"
            f"  - Trend behavior: {features['trend_desc']}\n"
            f"  - Signal quality: {features['signal_quality']} (SNR: {features['snr_db']:.1f} dB)\n"
            f"  - Energy entropy: {features['energy_entropy']:.2f}\n"
            f"  - Prediction difficulty: {features['difficulty']}\n"
            f"Temporal patterns: top periodicities are {[int(l) for l in lags]}<|<end_prompt>|>"
        )
        
        return prompt
