# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    保持dtype一致性，避免BFloat16转换为Float32
    """
    result = a / b
    # 使用与输入相同dtype的零张量，而不是Python float
    zero = t.zeros_like(result)
    result = t.where(t.isnan(result), zero, result)
    result = t.where(t.isinf(result), zero, result)
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        # 使用张量常量避免dtype提升
        factor = t.tensor(200.0, dtype=forecast.dtype, device=forecast.device)
        return factor * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


# ==================== 频域损失函数 ====================


class WaveletCoeffLoss(nn.Module):
    """小波系数多尺度损失
    
    直接监督小波系数的预测，在频域空间计算损失。
    通过频段权重平衡不同频率成分的重要性。
    
    Args:
        level: SWT分解层数（默认3）
        band_weights: 频段权重列表 [cA_n, cD_n, cD_{n-1}, ..., cD_1]
                     默认: [2.0, 0.3, 1.0, 0.8] - 低频趋势最重要
        loss_type: 损失类型，'mse' 或 'mae'
    
    Input:
        pred_coeffs: (B, N, T, num_bands) - 预测的小波系数
        true_coeffs: (B, N, T, num_bands) - 真实的小波系数
    
    Output:
        loss: 标量损失值
    """
    def __init__(self, level=3, band_weights=None, loss_type='mse'):
        super(WaveletCoeffLoss, self).__init__()
        self.level = level
        self.num_bands = level + 1
        self.loss_type = loss_type
        
        # 默认频段权重：低频趋势 > 中频模式 > 低频细节 > 高频噪声
        if band_weights is None:
            # [cA3=2.0, cD3=0.3, cD2=1.0, cD1=0.8]
            band_weights = [2.0, 0.3, 1.0, 0.8]
        
        # 不固定dtype，让权重在forward时自动匹配输入张量的dtype
        self.band_weights = band_weights
        
        print(f"[WaveletCoeffLoss] 初始化完成")
        print(f"  - 分解层数: {level}, 频段数: {self.num_bands}")
        print(f"  - 频段权重: {band_weights}")
        print(f"  - 损失类型: {loss_type.upper()}")
    
    def forward(self, pred_coeffs, true_coeffs):
        """
        计算加权多频段损失
        
        Args:
            pred_coeffs: (B, N, T, num_bands) - 预测的小波系数
            true_coeffs: (B, N, T, num_bands) - 真实的小波系数
        
        Returns:
            loss: 加权平均损失
        """
        # 输入验证
        assert pred_coeffs.shape == true_coeffs.shape, \
            f"形状不匹配: pred {pred_coeffs.shape} vs true {true_coeffs.shape}"
        assert pred_coeffs.shape[-1] == self.num_bands, \
            f"频段数不匹配: 期望{self.num_bands}, 实际{pred_coeffs.shape[-1]}"
        
        # 获取输入的dtype和device
        device = pred_coeffs.device
        dtype = pred_coeffs.dtype
        
        # 将权重转换为与输入相同的dtype和device
        band_weights_tensor = t.tensor(self.band_weights, dtype=dtype, device=device)
        
        # 初始化损失（与输入同dtype）
        total_loss = t.tensor(0.0, dtype=dtype, device=device)
        
        # 逐频段计算损失并加权
        for i in range(self.num_bands):
            pred_band = pred_coeffs[..., i]
            true_band = true_coeffs[..., i]
            
            if self.loss_type == 'mse':
                band_loss = nn.functional.mse_loss(pred_band, true_band)
            elif self.loss_type == 'mae':
                band_loss = nn.functional.l1_loss(pred_band, true_band)
            else:
                raise ValueError(f"不支持的损失类型: {self.loss_type}")
            
            total_loss = total_loss + band_weights_tensor[i] * band_loss
        
        # 归一化（除以权重和）
        total_loss = total_loss / band_weights_tensor.sum()
        
        return total_loss


class SpectralEnergyLoss(nn.Module):
    """频谱能量损失
    
    通过FFT计算频谱，对比预测和真实信号在不同频段的能量分布。
    适用于周期性时间序列预测，保证频域结构相似性。
    
    Args:
        freq_bands: 频段范围列表 [(low1, high1), (low2, high2), ...]
                   归一化频率范围 [0, 0.5]，例如：
                   [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
                   分别对应低频、中频、高频
        loss_type: 'l1' 或 'l2'
    
    Input:
        pred: (B, T, N) - 时域预测信号
        target: (B, T, N) - 时域真实信号
    
    Output:
        loss: 频段能量差异
    """
    def __init__(self, freq_bands=None, loss_type='l1'):
        super(SpectralEnergyLoss, self).__init__()
        
        if freq_bands is None:
            # 默认三频段：低频[0-10%]、中频[10-30%]、高频[30-50%]
            freq_bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)]
        
        self.freq_bands = freq_bands
        self.loss_type = loss_type
        
        print(f"[SpectralEnergyLoss] 初始化完成")
        print(f"  - 频段划分: {freq_bands}")
        print(f"  - 损失类型: {loss_type.upper()}")
    
    def forward(self, pred, target):
        """
        计算频谱能量损失
        
        Args:
            pred: (B, T, N) - 预测信号
            target: (B, T, N) - 真实信号
        
        Returns:
            loss: 频段能量相对误差
        """
        # 获取输入的dtype和device
        device = pred.device
        dtype = pred.dtype
        
        # FFT变换到频域 (B, T, N) -> (B, T//2+1, N)
        # 注意：FFT输出是复数，需要保持dtype一致性
        pred_fft = t.fft.rfft(pred, dim=1)
        target_fft = t.fft.rfft(target, dim=1)
        
        # 计算功率谱密度 (PSD)
        # abs()可能返回float32，需要显式转换为输入的dtype
        pred_power = (t.abs(pred_fft) ** 2).to(dtype)
        target_power = (t.abs(target_fft) ** 2).to(dtype)
        
        # 初始化损失（与输入同dtype）
        total_loss = t.tensor(0.0, dtype=dtype, device=device)
        freq_len = pred_power.shape[1]
        
        # 逐频段计算能量误差
        for low, high in self.freq_bands:
            # 频率索引范围
            low_idx = int(low * freq_len)
            high_idx = int(high * freq_len)
            
            # 确保索引有效
            if low_idx >= high_idx or high_idx > freq_len:
                continue
            
            # 该频段的总能量
            pred_energy = pred_power[:, low_idx:high_idx, :].sum(dim=1)  # (B, N)
            target_energy = target_power[:, low_idx:high_idx, :].sum(dim=1)  # (B, N)
            
            # 相对误差（避免除零）
            if self.loss_type == 'l1':
                band_loss = nn.functional.l1_loss(pred_energy, target_energy)
            else:  # l2
                band_loss = nn.functional.mse_loss(pred_energy, target_energy)
            
            # 归一化（相对于目标能量）
            # 使用与输入同dtype的epsilon避免dtype提升
            epsilon = t.tensor(1e-6, dtype=dtype, device=device)
            band_loss = band_loss / (target_energy.mean() + epsilon)
            total_loss = total_loss + band_loss
        
        # 平均 - 使用张量除法避免dtype提升
        num_bands = t.tensor(len(self.freq_bands), dtype=dtype, device=device)
        total_loss = total_loss / num_bands
        
        return total_loss


class HybridLoss(nn.Module):
    """混合损失函数 - 时域 + 频域多尺度监督
    
    综合三种损失：
    1. 时域MSE：点对点预测精度
    2. 小波系数损失：频域多尺度监督（可选）
    3. 频谱能量损失：全局频域结构约束
    
    Args:
        use_wavelet_loss: 是否启用小波系数损失（需要模型返回小波系数）
        use_spectral_loss: 是否启用频谱能量损失
        alpha: 时域MSE权重（默认1.0）
        beta: 小波系数权重（默认0.5）
        gamma: 频谱能量权重（默认0.3）
        wavelet_level: 小波分解层数
        wavelet_band_weights: 小波频段权重
        spectral_freq_bands: 频谱频段划分
    
    Usage:
        # 完全启用
        criterion = HybridLoss(
            use_wavelet_loss=True,
            use_spectral_loss=True,
            alpha=1.0, beta=0.5, gamma=0.3
        )
        
        # 仅时域+频谱（无需修改模型）
        criterion = HybridLoss(
            use_wavelet_loss=False,
            use_spectral_loss=True,
            alpha=1.0, gamma=0.3
        )
    """
    def __init__(self,
                 use_wavelet_loss=True,
                 use_spectral_loss=True,
                 alpha=1.0,
                 beta=0.5,
                 gamma=0.3,
                 wavelet_level=3,
                 wavelet_band_weights=None,
                 spectral_freq_bands=None):
        super(HybridLoss, self).__init__()
        
        self.use_wavelet_loss = use_wavelet_loss
        self.use_spectral_loss = use_spectral_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 时域损失（必选）
        self.time_loss = nn.MSELoss()
        
        # 小波系数损失（可选）
        if self.use_wavelet_loss:
            self.wavelet_loss = WaveletCoeffLoss(
                level=wavelet_level,
                band_weights=wavelet_band_weights,
                loss_type='mse'
            )
        else:
            self.wavelet_loss = None
        
        # 频谱能量损失（可选）
        if self.use_spectral_loss:
            self.spectral_loss = SpectralEnergyLoss(
                freq_bands=spectral_freq_bands,
                loss_type='l1'
            )
        else:
            self.spectral_loss = None
        
        print(f"\n[HybridLoss] 混合损失函数初始化完成")
        print(f"  - 时域MSE: ✅ (权重={alpha})")
        print(f"  - 小波系数损失: {'✅' if use_wavelet_loss else '❌'} " + 
              (f"(权重={beta})" if use_wavelet_loss else ""))
        print(f"  - 频谱能量损失: {'✅' if use_spectral_loss else '❌'} " +
              (f"(权重={gamma})" if use_spectral_loss else ""))
        print(f"  总权重: α={alpha}, β={beta}, γ={gamma}\n")
    
    def forward(self, pred_time, target_time, pred_coeffs=None, target_coeffs=None):
        """
        计算混合损失
        
        Args:
            pred_time: (B, T, N) - 时域预测
            target_time: (B, T, N) - 时域真值
            pred_coeffs: (B, N, T, num_bands) - 小波系数预测（可选）
            target_coeffs: (B, N, T, num_bands) - 小波系数真值（可选）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典（用于日志）
        """
        loss_dict = {}
        
        # 1. 时域MSE损失（主要）
        loss_time = self.time_loss(pred_time, target_time)
        loss_dict['time'] = loss_time.item()
        
        # 获取输入的dtype和device以保持一致性
        device = pred_time.device
        dtype = pred_time.dtype
        
        # 2. 小波系数损失（可选）
        if self.use_wavelet_loss and pred_coeffs is not None and target_coeffs is not None:
            loss_wavelet = self.wavelet_loss(pred_coeffs, target_coeffs)
            loss_dict['wavelet'] = loss_wavelet.item()
        else:
            # 返回与输入同dtype的零张量，而不是Python float
            loss_wavelet = t.tensor(0.0, dtype=dtype, device=device)
            loss_dict['wavelet'] = 0.0
        
        # 3. 频谱能量损失（可选）
        if self.use_spectral_loss:
            loss_spectral = self.spectral_loss(pred_time, target_time)
            loss_dict['spectral'] = loss_spectral.item()
        else:
            # 返回与输入同dtype的零张量，而不是Python float
            loss_spectral = t.tensor(0.0, dtype=dtype, device=device)
            loss_dict['spectral'] = 0.0
        
        # 加权求和
        # 关键修复：将Python float权重转换为与loss相同的dtype，避免dtype提升
        # BFloat16 * Python_float -> Float32 (错误❌)
        # BFloat16 * BFloat16_tensor -> BFloat16 (正确✅)
        alpha_tensor = t.tensor(self.alpha, dtype=dtype, device=device)
        beta_tensor = t.tensor(self.beta, dtype=dtype, device=device)
        gamma_tensor = t.tensor(self.gamma, dtype=dtype, device=device)
        
        total_loss = (
            alpha_tensor * loss_time +
            beta_tensor * loss_wavelet +
            gamma_tensor * loss_spectral
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
