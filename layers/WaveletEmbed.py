import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import ptwt
    from ptwt.stationary_transform import swt as ptwt_swt, iswt as ptwt_iswt
    import pywt  # 用于验证小波名称
    PTWT_AVAILABLE = True
except ImportError:
    PTWT_AVAILABLE = False
    import pywt
    print("Warning: ptwt not available, falling back to slow CPU implementation. Install with: pip install ptwt")


class SWTDecomposition(nn.Module):
    """
    平稳小波变换(Stationary Wavelet Transform)分解层
    在 Patching 之前对时间序列进行多尺度分解
    优化版本：使用 ptwt 进行 GPU 加速的批处理小波变换
    """
    def __init__(self, wavelet='db4', level=3, mode='symmetric'):
        """
        Args:
            wavelet: 小波基，默认 'db4'
            level: 分解层数，默认 3
            mode: 边界处理模式，默认 'symmetric' (ptwt使用'reflect')
        """
        super(SWTDecomposition, self).__init__()
        self.wavelet = wavelet
        self.level = level
        # ptwt 使用 'reflect', 'zero', 'constant' 等模式
        # 将 symmetric 映射到 reflect
        self.mode = 'reflect' if mode == 'symmetric' else mode
        self.use_gpu = PTWT_AVAILABLE
        
        # 验证小波名称有效性（GPU 和 CPU 模式都检查）
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet '{wavelet}' is not available. Choose from {pywt.wavelist()}")
    
    def forward(self, x):
        """
        前向传播进行小波分解 (GPU 加速批处理版本)
        
        Args:
            x: 输入张量 (B, N, T) - batch_size, n_vars, seq_len
            
        Returns:
            coeffs_tensor: 小波系数张量 (B, N, T, level+1)
                          包含 [cA_level, cD_level, ..., cD_1]
        """
        if self.use_gpu:
            return self._forward_gpu(x)
        else:
            return self._forward_cpu(x)
    
    def _forward_gpu(self, x):
        """
        GPU 加速版本：使用 ptwt 的 SWT 进行批处理小波变换
        """
        B, N, T = x.shape
        device = x.device
        dtype = x.dtype
        
        # 确保是 float32 (ptwt 要求)
        if dtype == torch.bfloat16 or dtype == torch.float16:
            x = x.float()
        
        # 重塑为 (B*N, 1, T) 以便批处理
        x_reshaped = x.reshape(B * N, 1, T)
        
        # 使用 ptwt.swt 进行批量平稳小波分解
        # swt 返回 [cA_n, cD_n, cD_n-1, ..., cD_1]，每个系数长度与原始信号相同
        coeffs_list = ptwt_swt(x_reshaped, self.wavelet, level=self.level)
        
        # coeffs_list 中每个元素形状都是 (B*N, 1, T)
        # 堆叠并重塑: (level+1, B*N, 1, T) -> (B*N, level+1, T)
        coeffs_stacked = torch.stack([c.squeeze(1) for c in coeffs_list], dim=1)
        
        # 重塑: (B*N, level+1, T) -> (B, N, level+1, T) -> (B, N, T, level+1)
        coeffs_tensor = coeffs_stacked.reshape(B, N, self.level + 1, T).permute(0, 1, 3, 2)
        
        return coeffs_tensor
    
    def _forward_cpu(self, x):
        """
        CPU 回退版本：使用原始 pywt 实现（保持向后兼容）
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
                signal = x[b, n, :].float().cpu().detach().numpy()
                
                # 执行 SWT 分解
                coeffs = pywt.swt(signal, wavelet=self.wavelet, level=self.level, 
                                 norm=True)
                
                # 提取系数
                var_coeffs = []
                
                # 提取最后一层的近似系数
                cA = np.asarray(coeffs[0][0]).flatten()
                var_coeffs.append(cA)
                
                # 提取所有层的细节系数
                for i in range(self.level):
                    cD = np.asarray(coeffs[i][1]).flatten()
                    var_coeffs.append(cD)
                
                # 堆叠成 (level+1, T) -> (T, level+1)
                var_coeffs_array = np.array(var_coeffs).T
                batch_coeffs.append(var_coeffs_array)
            
            # 堆叠成 (N, T, level+1)
            batch_coeffs = np.stack(batch_coeffs, axis=0)
            all_coeffs.append(batch_coeffs)
        
        # 堆叠成 (B, N, T, level+1)
        coeffs_tensor = torch.from_numpy(np.stack(all_coeffs, axis=0)).float().to(device)
        
        return coeffs_tensor
    
    def reconstruct(self, coeffs_tensor):
        """
        从小波系数重构信号 (GPU 加速批处理版本)
        
        Args:
            coeffs_tensor: (B, N, T, level+1) 小波系数
            
        Returns:
            reconstructed: (B, N, T) 重构后的信号
        """
        if self.use_gpu:
            return self._reconstruct_gpu(coeffs_tensor)
        else:
            return self._reconstruct_cpu(coeffs_tensor)
    
    def _reconstruct_gpu(self, coeffs_tensor):
        """
        GPU 加速重构：使用 ptwt.iswt
        """
        B, N, T, num_coeffs = coeffs_tensor.shape
        device = coeffs_tensor.device
        
        # 重塑为 (B*N, T, level+1) -> (B*N, level+1, T)
        coeffs = coeffs_tensor.reshape(B * N, T, num_coeffs).permute(0, 2, 1)
        
        # 将系数格式化为 ptwt.iswt 需要的列表格式
        # coeffs: (B*N, level+1, T) -> list of (B*N, 1, T)
        coeffs_list = [coeffs[:, i:i+1, :] for i in range(num_coeffs)]
        
        # 使用 ptwt.iswt 重构
        reconstructed = ptwt_iswt(coeffs_list, self.wavelet)
        
        # 重塑回 (B, N, T)
        reconstructed = reconstructed.squeeze(1).reshape(B, N, T)
        
        return reconstructed
    
    def _reconstruct_cpu(self, coeffs_tensor):
        """
        CPU 回退重构
        """
        B, N, T, _ = coeffs_tensor.shape
        device = coeffs_tensor.device
        
        reconstructed_signals = []
        
        for b in range(B):
            batch_signals = []
            for n in range(N):
                var_coeffs = coeffs_tensor[b, n, :, :].float().cpu().detach().numpy()
                
                # 重组成 pywt.iswt 格式
                coeffs = []
                cA = var_coeffs[:, 0]
                
                for i in range(self.level):
                    cD = var_coeffs[:, i + 1]
                    coeffs.append((cA, cD))
                    cA = None
                
                # 执行逆 SWT
                signal = pywt.iswt(coeffs, wavelet=self.wavelet, norm=True)
                
                # 截断或填充
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
