"""
WaveletEmbed 模块的全面测试套件
使用 pytest 框架进行结构化测试，支持参数化、性能测试、GPU测试等
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pywt
import time
from typing import Tuple, List
from layers.WaveletEmbed import (
    SWTDecomposition, 
    WaveletPatchEmbedding, 
    ReplicationPad1d
)


# ============================================================================
# 测试配置和工具函数
# ============================================================================

@pytest.fixture
def device():
    """返回可用的设备（GPU优先）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def seed():
    """设置随机种子以确保测试可重复"""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


def get_wavelet_list():
    """获取常用的小波基列表"""
    return ['db4', 'db6', 'sym4', 'coif2', 'haar']


def measure_time(func):
    """装饰器：测量函数执行时间"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper


# ============================================================================
# SWTDecomposition 测试类
# ============================================================================

class TestSWTDecomposition:
    """SWT 分解层的全面测试"""
    
    @pytest.mark.parametrize("batch_size,n_vars,seq_len", [
        (1, 1, 64),
        (4, 3, 128),
        (8, 5, 256),
        (16, 10, 512),
    ])
    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_output_shape(self, batch_size, n_vars, seq_len, level, device):
        """测试不同参数下的输出形状"""
        swt = SWTDecomposition(wavelet='db4', level=level).to(device)
        x = torch.randn(batch_size, n_vars, seq_len).to(device)
        
        coeffs = swt(x)
        expected_shape = (batch_size, n_vars, seq_len, level + 1)
        
        assert coeffs.shape == expected_shape, \
            f"期望 {expected_shape}, 得到 {coeffs.shape}"
        assert not torch.isnan(coeffs).any(), "输出包含 NaN"
        assert not torch.isinf(coeffs).any(), "输出包含 Inf"
    
    @pytest.mark.parametrize("wavelet", get_wavelet_list())
    def test_different_wavelets(self, wavelet, device):
        """测试不同的小波基"""
        B, N, T = 4, 3, 256
        swt = SWTDecomposition(wavelet=wavelet, level=2).to(device)
        
        x = torch.randn(B, N, T).to(device)
        coeffs = swt(x)
        
        assert coeffs.shape == (B, N, T, 3)
        assert not torch.isnan(coeffs).any()
    
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    def test_perfect_reconstruction(self, seq_len, device):
        """测试完美重构特性 - 不同序列长度"""
        B, N = 2, 2
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, seq_len).to(device)
        coeffs = swt(x)
        x_reconstructed = swt.reconstruct(coeffs)
        
        mse = torch.mean((x - x_reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(x - x_reconstructed)).item()
        
        assert mse < 1e-6, f"重构MSE过大: {mse}"
        assert max_error < 1e-5, f"最大误差过大: {max_error}"
    
    def test_reconstruction_consistency(self, device):
        """测试重构的一致性：多次重构应该得到相同结果"""
        B, N, T = 2, 2, 256
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, T).to(device)
        coeffs = swt(x)
        
        # 多次重构
        rec1 = swt.reconstruct(coeffs)
        rec2 = swt.reconstruct(coeffs)
        rec3 = swt.reconstruct(coeffs)
        
        assert torch.allclose(rec1, rec2, atol=1e-8)
        assert torch.allclose(rec2, rec3, atol=1e-8)
    
    def test_coefficient_structure(self, device):
        """测试系数结构：验证近似系数和细节系数的顺序"""
        B, N, T = 1, 1, 256
        level = 3
        swt = SWTDecomposition(wavelet='db4', level=level).to(device)
        
        # 创建纯低频信号
        t = torch.linspace(0, 4*np.pi, T).to(device)
        x_low = torch.sin(t).view(B, N, T)
        
        # 创建纯高频信号
        x_high = torch.sin(20 * t).view(B, N, T)
        
        coeffs_low = swt(x_low)
        coeffs_high = swt(x_high)
        
        # 低频信号应该在 cA (第0个系数) 有更多能量
        cA_low = coeffs_low[:, :, :, 0].abs().sum()
        cD_low = coeffs_low[:, :, :, 1:].abs().sum()
        
        # 高频信号应该在 cD 有更多能量
        cA_high = coeffs_high[:, :, :, 0].abs().sum()
        cD_high = coeffs_high[:, :, :, 1:].abs().sum()
        
        assert cA_low > cD_low * 0.5, "低频信号应该在cA有更多能量"
        assert cD_high > cA_high * 0.5, "高频信号应该在cD有更多能量"
    
    def test_batch_independence(self, device):
        """测试批次独立性：不同批次样本互不影响"""
        N, T = 3, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        # 创建不同的样本
        x1 = torch.randn(1, N, T).to(device)
        x2 = torch.randn(1, N, T).to(device)
        x_batch = torch.cat([x1, x2], dim=0)
        
        # 分别处理
        coeffs1 = swt(x1)
        coeffs2 = swt(x2)
        
        # 批次处理
        coeffs_batch = swt(x_batch)
        
        # 验证批次处理的每个样本与单独处理的结果一致
        assert torch.allclose(coeffs_batch[0], coeffs1[0], atol=1e-7)
        assert torch.allclose(coeffs_batch[1], coeffs2[0], atol=1e-7)
    
    def test_zero_input(self, device):
        """测试零输入"""
        B, N, T = 2, 2, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x_zero = torch.zeros(B, N, T).to(device)
        coeffs = swt(x_zero)
        x_rec = swt.reconstruct(coeffs)
        
        assert coeffs.abs().max() < 1e-10, "零输入应产生零系数"
        assert x_rec.abs().max() < 1e-10, "零系数应重构为零"
    
    def test_constant_input(self, device):
        """测试常数输入"""
        B, N, T = 2, 2, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        # 常数信号
        x_const = torch.ones(B, N, T).to(device) * 5.0
        coeffs = swt(x_const)
        x_rec = swt.reconstruct(coeffs)
        
        # 重构应该恢复原始常数
        mse = torch.mean((x_const - x_rec) ** 2).item()
        assert mse < 1e-6, f"常数重构误差: {mse}"
    
    def test_invalid_wavelet(self):
        """测试无效的小波基"""
        with pytest.raises(ValueError, match="Wavelet .* is not available"):
            swt = SWTDecomposition(wavelet='invalid_wavelet_name', level=3)
    
    def test_energy_conservation(self, device):
        """测试能量守恒：Parseval定理"""
        B, N, T = 2, 2, 256
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, T).to(device)
        coeffs = swt(x)
        
        # 原始信号能量
        energy_original = torch.sum(x ** 2).item()
        
        # 小波系数能量（需要考虑归一化）
        energy_coeffs = torch.sum(coeffs ** 2).item()
        
        # 由于使用了norm=True，能量应该基本守恒
        ratio = energy_coeffs / energy_original
        assert 0.8 < ratio < 1.2, f"能量比率异常: {ratio}"


# ============================================================================
# WaveletPatchEmbedding 测试类
# ============================================================================

class TestWaveletPatchEmbedding:
    """WaveletPatchEmbedding 层的全面测试"""
    
    @pytest.mark.parametrize("use_all_coeffs", [True, False])
    @pytest.mark.parametrize("batch_size,n_vars,seq_len", [
        (4, 3, 128),
        (8, 5, 256),
        (16, 7, 512),
    ])
    def test_output_shape(self, use_all_coeffs, batch_size, n_vars, seq_len, device):
        """测试不同配置下的输出形状"""
        d_model = 128
        patch_len = 16
        stride = 8
        
        model = WaveletPatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=0.0,
            use_all_coeffs=use_all_coeffs
        ).to(device)
        
        x = torch.randn(batch_size, n_vars, seq_len).to(device)
        output, n_vars_out = model(x)
        
        num_patches = (seq_len + stride - patch_len) // stride + 1
        expected_shape = (batch_size * n_vars, num_patches, d_model)
        
        assert output.shape == expected_shape
        assert n_vars_out == n_vars
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("patch_len,stride", [
        (8, 4),
        (16, 8),
        (16, 16),  # 无重叠
        (32, 16),
        (32, 8),   # 大重叠
    ])
    def test_different_patch_configs(self, patch_len, stride, device):
        """测试不同的patch配置"""
        B, N, T = 4, 3, 512
        d_model = 256
        
        model = WaveletPatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        output, _ = model(x)
        
        num_patches = (T + stride - patch_len) // stride + 1
        expected_shape = (B * N, num_patches, d_model)
        
        assert output.shape == expected_shape
    
    @pytest.mark.parametrize("d_model", [64, 128, 256, 512, 768, 1024])
    def test_different_d_models(self, d_model, device):
        """测试不同的嵌入维度"""
        B, N, T = 4, 3, 256
        
        model = WaveletPatchEmbedding(
            d_model=d_model,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        output, _ = model(x)
        
        assert output.shape[-1] == d_model
    
    @pytest.mark.parametrize("swt_level", [1, 2, 3, 4, 5])
    def test_different_swt_levels(self, swt_level, device):
        """测试不同的SWT分解层数"""
        B, N, T = 4, 3, 512
        
        model = WaveletPatchEmbedding(
            d_model=256,
            patch_len=16,
            stride=8,
            dropout=0.0,
            swt_level=swt_level,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        output, _ = model(x)
        
        # 应该能正常工作
        assert not torch.isnan(output).any()
    
    def test_dropout_training_vs_eval(self, device):
        """测试训练模式和评估模式下的dropout行为"""
        B, N, T = 4, 3, 128
        
        model = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.5,  # 高dropout率
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        
        # 训练模式
        model.train()
        output1, _ = model(x)
        output2, _ = model(x)
        diff_train = torch.mean(torch.abs(output1 - output2)).item()
        
        # 评估模式
        model.eval()
        with torch.no_grad():
            output3, _ = model(x)
            output4, _ = model(x)
        diff_eval = torch.mean(torch.abs(output3 - output4)).item()
        
        assert diff_train > 0.01, "训练模式下dropout应该有随机性"
        assert diff_eval < 1e-6, "评估模式下输出应该确定"
    
    def test_gradient_flow(self, device):
        """测试梯度反向传播"""
        B, N, T = 4, 3, 128
        
        model = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        output, _ = model(x)
        
        loss = output.sum()
        loss.backward()
        
        # 检查所有可学习参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含NaN"
    
    def test_embedding_statistics(self, device):
        """测试嵌入输出的统计特性"""
        B, N, T = 16, 5, 256
        
        model = WaveletPatchEmbedding(
            d_model=256,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        # 标准化输入
        x = torch.randn(B, N, T).to(device)
        x = (x - x.mean()) / x.std()
        
        with torch.no_grad():
            output, _ = model(x)
        
        mean = output.mean().item()
        std = output.std().item()
        
        # 检查统计特性在合理范围
        assert abs(mean) < 1.0, f"输出均值过大: {mean}"
        assert 0.05 < std < 20.0, f"输出标准差异常: {std}"
    
    def test_all_coeffs_vs_approx_only(self, device, seed):
        """测试全系数模式和近似系数模式的差异"""
        B, N, T = 4, 3, 256
        
        # 使用相同的输入
        x = torch.randn(B, N, T).to(device)
        
        model_all = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        model_approx = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=False
        ).to(device)
        
        with torch.no_grad():
            output_all, n_vars_all = model_all(x)
            output_approx, n_vars_approx = model_approx(x)
        
        # 输出形状应该相同
        assert output_all.shape == output_approx.shape
        assert n_vars_all == n_vars_approx
    
    def test_single_variable(self, device):
        """测试单变量输入"""
        B, N, T = 8, 1, 256  # N=1
        
        model = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        output, n_vars = model(x)
        
        assert n_vars == 1
        assert output.shape[0] == B * N
    
    def test_large_batch(self, device):
        """测试大批次处理"""
        if not torch.cuda.is_available():
            pytest.skip("需要GPU才能高效测试大批次")
        
        B, N, T = 128, 10, 512
        
        model = WaveletPatchEmbedding(
            d_model=256,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=False  # 使用轻量配置
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        
        with torch.no_grad():
            output, n_vars = model(x)
        
        assert output.shape[0] == B * N
        assert not torch.isnan(output).any()


# ============================================================================
# ReplicationPad1d 测试类
# ============================================================================

class TestReplicationPad1d:
    """ReplicationPad1d 层的全面测试"""
    
    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
    def test_different_dimensions(self, ndim, device):
        """测试不同维度的输入"""
        pad = ReplicationPad1d((2, 3))
        
        # 创建不同维度的张量，最后一维长度为10
        shape = [2] * (ndim - 1) + [10]
        x = torch.randn(*shape).to(device)
        
        output = pad(x)
        expected_shape = shape[:-1] + [15]  # 10 + 2 + 3
        
        assert list(output.shape) == expected_shape
    
    @pytest.mark.parametrize("left,right", [
        (0, 0),    # 无填充
        (0, 5),    # 只填充右侧
        (5, 0),    # 只填充左侧
        (3, 3),    # 两侧相等
        (1, 10),   # 不对称填充
        (10, 1),
    ])
    def test_different_paddings(self, left, right, device):
        """测试不同的填充配置"""
        pad = ReplicationPad1d((left, right))
        x = torch.arange(10).float().to(device)
        
        output = pad(x)
        expected_len = 10 + left + right
        
        assert len(output) == expected_len
        
        # 检查填充值
        if left > 0:
            assert all(output[:left] == x[0])
        if right > 0:
            assert all(output[-right:] == x[-1])
    
    def test_padding_values_correctness(self, device):
        """测试填充值的正确性"""
        pad = ReplicationPad1d((3, 3))
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
        
        output = pad(x)
        expected = torch.tensor([1., 1., 1., 1., 2., 3., 4., 5., 5., 5., 5.]).to(device)
        
        assert torch.allclose(output, expected)
    
    def test_2d_padding(self, device):
        """测试2D张量的填充"""
        pad = ReplicationPad1d((0, 5))
        x = torch.arange(20).reshape(4, 5).float().to(device)
        
        output = pad(x)
        
        assert output.shape == (4, 10)
        
        # 检查每一行的填充
        for i in range(4):
            assert all(output[i, -5:] == x[i, -1])
    
    def test_zero_padding(self, device):
        """测试零填充（无填充）"""
        pad = ReplicationPad1d((0, 0))
        x = torch.randn(3, 10).to(device)
        
        output = pad(x)
        
        assert torch.allclose(output, x)


# ============================================================================
# 性能和压力测试
# ============================================================================

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要GPU")
    def test_gpu_speedup(self):
        """测试GPU加速效果"""
        B, N, T = 32, 7, 512
        
        model_cpu = WaveletPatchEmbedding(
            d_model=256,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=False
        )
        
        model_gpu = WaveletPatchEmbedding(
            d_model=256,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=False
        ).cuda()
        
        x_cpu = torch.randn(B, N, T)
        x_gpu = x_cpu.cuda()
        
        # CPU时间
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_cpu(x_cpu)
        cpu_time = time.time() - start
        
        # GPU时间
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\nCPU时间: {cpu_time:.4f}s")
        print(f"GPU时间: {gpu_time:.4f}s")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
    
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024, 2048])
    def test_scaling_with_sequence_length(self, seq_len, device):
        """测试随序列长度的扩展性"""
        B, N = 8, 5
        
        model = WaveletPatchEmbedding(
            d_model=256,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, seq_len).to(device)
        
        start = time.time()
        with torch.no_grad():
            output, _ = model(x)
        elapsed = time.time() - start
        
        print(f"\nT={seq_len}: {elapsed:.4f}s")
        
        assert not torch.isnan(output).any()
    
    def test_memory_efficiency(self, device):
        """测试内存效率"""
        if not torch.cuda.is_available():
            pytest.skip("需要GPU测试内存")
        
        B, N, T = 64, 10, 1024
        
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        
        model = WaveletPatchEmbedding(
            d_model=512,
            patch_len=32,
            stride=16,
            dropout=0.0,
            use_all_coeffs=False
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        
        with torch.no_grad():
            output, _ = model(x)
        
        mem_after = torch.cuda.memory_allocated()
        mem_used = (mem_after - mem_before) / 1024 / 1024  # MB
        
        print(f"\n内存使用: {mem_used:.2f} MB")


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试：测试组件之间的配合"""
    
    def test_swt_to_embedding_pipeline(self, device):
        """测试SWT到Embedding的完整流程"""
        B, N, T = 4, 3, 256
        
        # 单独使用SWT
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        # 使用集成的embedding
        model = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.0,
            wavelet='db4',
            swt_level=3,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        
        # 测试流程
        coeffs = swt(x)
        output, n_vars = model(x)
        
        assert coeffs.shape == (B, N, T, 4)
        assert output.shape[0] == B * N
    
    def test_with_downstream_transformer(self, device):
        """测试与下游Transformer的集成"""
        B, N, T = 8, 5, 256
        
        embedding = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.1,
            use_all_coeffs=True
        ).to(device)
        
        # 简单的Transformer层
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        
        # 前向传播
        embedded, n_vars = embedding(x)
        transformer_out = transformer_layer(embedded)
        
        assert transformer_out.shape == embedded.shape
        assert not torch.isnan(transformer_out).any()
    
    def test_end_to_end_gradient(self, device):
        """测试端到端的梯度流"""
        B, N, T = 4, 3, 128
        
        # 构建简单的端到端模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = WaveletPatchEmbedding(
                    d_model=64,
                    patch_len=16,
                    stride=8,
                    dropout=0.0,
                    use_all_coeffs=True
                )
                self.fc = nn.Linear(64, 1)
            
            def forward(self, x):
                embedded, n_vars = self.embedding(x)
                # 池化
                pooled = embedded.mean(dim=1)  # (B*N, d_model)
                output = self.fc(pooled)  # (B*N, 1)
                return output
        
        model = SimpleModel().to(device)
        x = torch.randn(B, N, T).to(device)
        target = torch.randn(B * N, 1).to(device)
        
        # 前向传播
        pred = model(x)
        loss = nn.MSELoss()(pred, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 没有梯度"


# ============================================================================
# 运行配置
# ============================================================================

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([
        __file__,
        "-v",  # 详细输出
        "-s",  # 显示print输出
        "--tb=short",  # 简短的traceback
        "--color=yes",  # 彩色输出
    ])
