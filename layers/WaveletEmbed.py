import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class SWTDecomposition(nn.Module):
    """
    平稳小波变换(Stationary Wavelet Transform)分解层
    在 Patching 之前对时间序列进行多尺度分解
    """
    def __init__(self, wavelet='db4', level=3, mode='symmetric'):
        """
        Args:
            wavelet: 小波基，默认 'db4'
            level: 分解层数，默认 3
            mode: 边界处理模式，默认 'symmetric'
        """
        super(SWTDecomposition, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        
        # 验证小波基是否可用
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' is not available. Choose from {pywt.wavelist()}")
    
    def forward(self, x):
        """
        前向传播进行 SWT 分解
        
        Args:
            x: 输入张量 (B, N, T) - batch_size, n_vars, seq_len
            
        Returns:
            coeffs_tensor: 重构后的小波系数张量 (B, N, T, level+1)
                          包含 [cA3, cD3, cD2, cD1] (对于3层分解)
        """
        B, N, T = x.shape
        device = x.device
        
        # 存储所有变量的小波系数
        all_coeffs = []
        
        # 对每个变量进行 SWT 分解
        for b in range(B):
            batch_coeffs = []
            for n in range(N):
                # 转换到 CPU numpy 进行 SWT
                # 先转换为 float32，因为 numpy 不支持 bfloat16
                signal = x[b, n, :].float().cpu().detach().numpy()
                
                # 执行 SWT 分解
                # coeffs 格式: [(cA_n, cD_n), (cA_n-1, cD_n-1), ..., (cA_1, cD_1)]
                # 对于 level=3: [(cA3, cD3), (cA2, cD2), (cA1, cD1)]
                coeffs = pywt.swt(signal, wavelet=self.wavelet, level=self.level, 
                                 norm=True)
                
                # 提取系数: cA_n, cD_n, cD_n-1, ..., cD_1
                # 确保每个系数都是一维数组且长度为 T
                var_coeffs = []
                
                # 提取最后一层的近似系数 (低频)
                cA = np.asarray(coeffs[0][0]).flatten()  # cA3 (最粗尺度的近似系数)
                var_coeffs.append(cA)
                
                # 提取所有层的细节系数 (高频)，从粗到细
                for i in range(self.level):
                    cD = np.asarray(coeffs[i][1]).flatten()  # cD3, cD2, cD1
                    var_coeffs.append(cD)
                
                # 堆叠成 (T, level+1)
                # 转置使得形状为 (T, level+1) 而不是 (level+1, T)
                var_coeffs_array = np.array(var_coeffs)  # (level+1, T)
                var_coeffs_array = var_coeffs_array.T  # (T, level+1)
                
                batch_coeffs.append(var_coeffs_array)
            
            # 堆叠成 (N, T, level+1)
            batch_coeffs = np.stack(batch_coeffs, axis=0)
            all_coeffs.append(batch_coeffs)
        
        # 堆叠成 (B, N, T, level+1)
        coeffs_tensor = torch.from_numpy(np.stack(all_coeffs, axis=0)).float().to(device)
        
        return coeffs_tensor
    
    def reconstruct(self, coeffs_tensor):
        """
        从小波系数重构信号
        
        Args:
            coeffs_tensor: (B, N, T, level+1) 小波系数
            
        Returns:
            reconstructed: (B, N, T) 重构后的信号
        """
        B, N, T, _ = coeffs_tensor.shape
        device = coeffs_tensor.device
        
        reconstructed_signals = []
        
        for b in range(B):
            batch_signals = []
            for n in range(N):
                # 获取该变量的所有系数
                # 先转换为 float32，因为 numpy 不支持 bfloat16
                var_coeffs = coeffs_tensor[b, n, :, :].float().cpu().detach().numpy()
                
                # 重组成 pywt.iswt 格式
                # var_coeffs: (T, level+1) -> [cA3, cD3, cD2, cD1]
                # pywt.iswt 需要: [(cA3, cD3), (cA2, cD2), (cA1, cD1)]
                coeffs = []
                cA = var_coeffs[:, 0]  # cA3 (最粗尺度的近似系数)
                
                # 对于 level=3: cD3, cD2, cD1
                for i in range(self.level):
                    cD = var_coeffs[:, i + 1]  # cD3, cD2, cD1
                    coeffs.append((cA, cD))
                    # 对于后续层，不再需要近似系数（iswt会忽略除第一个外的cA）
                    cA = None
                
                # 执行逆 SWT
                signal = pywt.iswt(coeffs, wavelet=self.wavelet, norm=True)
                
                # 由于 SWT 可能会改变长度，截断或填充到原始长度
                if len(signal) > T:
                    signal = signal[:T]
                elif len(signal) < T:
                    signal = np.pad(signal, (0, T - len(signal)), mode='edge')
                
                batch_signals.append(signal)
            
            batch_signals = np.stack(batch_signals, axis=0)
            reconstructed_signals.append(batch_signals)
        
        reconstructed = torch.from_numpy(np.stack(reconstructed_signals, axis=0)).float().to(device)
        
        return reconstructed


class WaveletPatchEmbedding(nn.Module):
    """
    结合 SWT 和 Patch Embedding 的嵌入层
    工作流程：原始序列 -> SWT分解 -> Patch切分 -> Token Embedding
    """
    def __init__(self, d_model, patch_len, stride, dropout, 
                 wavelet='db4', swt_level=3, use_all_coeffs=True):
        """
        Args:
            d_model: 嵌入维度
            patch_len: patch 长度
            stride: 滑动步长
            dropout: dropout 比率
            wavelet: 小波基，默认 'db4'
            swt_level: SWT 分解层数，默认 3
            use_all_coeffs: 是否使用所有小波系数（True）还是仅用近似系数（False）
        """
        super(WaveletPatchEmbedding, self).__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.use_all_coeffs = use_all_coeffs
        self.swt_level = swt_level
        
        # SWT 分解层
        self.swt = SWTDecomposition(wavelet=wavelet, level=swt_level)
        
        # Padding 层
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        
        # 根据是否使用所有系数来确定输入通道数
        if use_all_coeffs:
            # 使用所有系数: cA + cD1 + cD2 + cD3 = level + 1
            input_channels = patch_len * (swt_level + 1)
        else:
            # 仅使用近似系数 cA
            input_channels = patch_len
        
        # Token Embedding: 将 patch 映射到 d_model 维度
        self.value_embedding = nn.Conv1d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )
        
        # 初始化权重
        nn.init.kaiming_normal_(self.value_embedding.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, N, T) - batch_size, n_vars, seq_len
            
        Returns:
            embedded: (B*N, num_patches, d_model)
            n_vars: 变量数量
        """
        # 如果输入是 bfloat16，转换为 float32 进行处理
        # 注意：输出始终为 float32，因为后续层期望 float32 输入
        if x.dtype == torch.bfloat16:
            x = x.float()
        
        B, N, T = x.shape
        n_vars = N
        
        # Step 1: SWT 分解
        # coeffs: (B, N, T, level+1)
        coeffs = self.swt(x)
        
        # Step 2: 处理小波系数，保持时间局部性
        if self.use_all_coeffs:
            # 使用所有系数: (B, N, T, level+1)
            # 我们需要保持每个时间点的多尺度特征在一起
            # 先对时间维度进行 padding
            # coeffs: (B, N, T, level+1) -> permute -> (B, N, level+1, T)
            coeffs_permuted = coeffs.permute(0, 1, 3, 2)  # (B, N, 4, T)
            
            # Padding: 在时间维度（最后一维）使用 replicate 模式
            # (B, N, 4, T) -> (B, N, 4, T+stride)
            # 手动实现replicate padding以避免PyTorch版本兼容问题
            if self.stride > 0:
                # 取最后一个时间步并重复stride次
                last_values = coeffs_permuted[..., -1:].repeat(1, 1, 1, self.stride)
                coeffs_padded = torch.cat([coeffs_permuted, last_values], dim=-1)
            else:
                coeffs_padded = coeffs_permuted
            
            # Unfold: 在时间维度上切分 patch
            # (B, N, 4, T+stride) -> unfold on dim=-1
            # 结果: (B, N, 4, num_patches, patch_len)
            x_patch = coeffs_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            
            # 重排维度: (B, N, 4, num_patches, patch_len) -> (B, N, num_patches, patch_len, 4)
            x_patch = x_patch.permute(0, 1, 3, 4, 2)
            
            # Flatten 最后两维: (B, N, num_patches, patch_len*4)
            num_patches = x_patch.shape[2]
            x_patch = x_patch.reshape(B, N, num_patches, self.patch_len * (self.swt_level + 1))
            
        else:
            # 仅使用近似系数 cA (第一个系数)
            x_swt = coeffs[:, :, :, 0]  # (B, N, T)
            
            # Padding
            x_swt = self.padding_patch_layer(x_swt)
            
            # Unfold 进行 patching
            x_patch = x_swt.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # x_patch: (B, N, num_patches, patch_len)
        
        # Step 3: 重塑为 (B*N, num_patches, patch_len*(...))
        x_patch = x_patch.reshape(B * N, x_patch.shape[2], x_patch.shape[3])
        
        # Step 4: Token Embedding
        # 转置为 (B*N, patch_len*(...), num_patches)
        x_patch = x_patch.transpose(1, 2)
        
        # 确保数据类型为 float32（Conv1d 需要）
        if x_patch.dtype != torch.float32:
            x_patch = x_patch.float()
        
        # 通过卷积进行嵌入: (B*N, d_model, num_patches)
        x_embed = self.value_embedding(x_patch)
        
        # 转置回: (B*N, num_patches, d_model)
        x_embed = x_embed.transpose(1, 2)
        
        # Step 5: Dropout
        x_embed = self.dropout(x_embed)
        
        # 注意：不转换回 bfloat16，因为后续层（ReprogrammingLayer）期望 float32
        # TimeLLM 的设计是：输入转为 bfloat16 只是为了内存效率，但 embedding 输出应为 float32
        
        return x_embed, n_vars


class ReplicationPad1d(nn.Module):
    """
    复制填充层 - 在最后一维进行 replicate padding
    支持任意维度的输入，只要最后一维是时间维度
    """
    def __init__(self, padding):
        """
        Args:
            padding: (left, right) 元组，表示在最后一维左右两侧的填充量
        """
        super(ReplicationPad1d, self).__init__()
        self.padding = padding
    
    def forward(self, input):
        """
        Args:
            input: (..., T) 任意维度，最后一维是时间
        Returns:
            output: (..., T + padding[0] + padding[1])
        """
        left_pad, right_pad = self.padding
        
        # 如果不需要填充，直接返回
        if left_pad == 0 and right_pad == 0:
            return input
        
        # 手动实现 replicate padding 以支持所有维度
        parts = [input]
        
        # 左侧填充：复制第一个元素
        if left_pad > 0:
            # 获取第一个时间步: (..., 1)
            first_elem = input[..., :1]
            # 重复 left_pad 次
            left_padding = first_elem.repeat(*([1] * (input.ndim - 1) + [left_pad]))
            parts.insert(0, left_padding)
        
        # 右侧填充：复制最后一个元素
        if right_pad > 0:
            # 获取最后一个时间步: (..., 1)
            last_elem = input[..., -1:]
            # 重复 right_pad 次
            right_padding = last_elem.repeat(*([1] * (input.ndim - 1) + [right_pad]))
            parts.append(right_padding)
        
        # 在最后一维拼接
        return torch.cat(parts, dim=-1)
