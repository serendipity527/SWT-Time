"""
WaveletEmbed 模块的全面测试套件
覆盖 SWTDecomposition, WaveletPatchEmbedding, ReplicationPad1d 的所有功能

运行方式:
    pytest test_wavelet_embed.py -v
    pytest test_wavelet_embed.py -v -k "TestSWT"  # 只运行 SWT 测试
    pytest test_wavelet_embed.py -v --tb=short    # 简短错误信息
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, List

from layers.WaveletEmbed import (
    SWTDecomposition,
    WaveletPatchEmbedding,
    ReplicationPad1d,
    PTWT_AVAILABLE
)


# ============================================================================
# Fixtures 和工具函数
# ============================================================================

@pytest.fixture
def device():
    """返回可用的设备（GPU优先）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def seed():
    """设置随机种子"""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


def get_common_wavelets():
    """常用小波基列表"""
    return ['db4', 'db6', 'sym4', 'coif2', 'haar']


# ============================================================================
# SWTDecomposition 测试
# ============================================================================

class TestSWTDecomposition:
    """SWT 分解层的全面测试"""

    # ---------- 基本功能测试 ----------
    
    @pytest.mark.parametrize("batch_size,n_vars,seq_len", [
        (1, 1, 64),
        (4, 3, 128),
        (8, 7, 256),
        (16, 10, 512),
    ])
    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_output_shape(self, batch_size, n_vars, seq_len, level, device):
        """测试不同参数下的输出形状"""
        swt = SWTDecomposition(wavelet='db4', level=level).to(device)
        x = torch.randn(batch_size, n_vars, seq_len).to(device)
        
        coeffs = swt(x)
        expected_shape = (batch_size, n_vars, seq_len, level + 1)
        
        assert coeffs.shape == expected_shape, f"期望 {expected_shape}, 得到 {coeffs.shape}"
        assert not torch.isnan(coeffs).any(), "输出包含 NaN"
        assert not torch.isinf(coeffs).any(), "输出包含 Inf"

    @pytest.mark.parametrize("wavelet", get_common_wavelets())
    def test_different_wavelets(self, wavelet, device):
        """测试不同的小波基"""
        B, N, T = 4, 3, 256
        swt = SWTDecomposition(wavelet=wavelet, level=2).to(device)
        
        x = torch.randn(B, N, T).to(device)
        coeffs = swt(x)
        
        assert coeffs.shape == (B, N, T, 3)
        assert not torch.isnan(coeffs).any()

    def test_invalid_wavelet(self):
        """测试无效的小波基"""
        with pytest.raises(ValueError, match="Wavelet .* is not available"):
            SWTDecomposition(wavelet='invalid_wavelet_name', level=3)

    # ---------- 重构测试 ----------

    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_perfect_reconstruction_gpu(self, seq_len, device):
        """测试 GPU 模式下的完美重构"""
        if not PTWT_AVAILABLE:
            pytest.skip("需要 ptwt 库")
        
        B, N = 2, 3
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, seq_len).to(device)
        coeffs = swt(x)
        x_reconstructed = swt.reconstruct(coeffs)
        
        mse = torch.mean((x - x_reconstructed) ** 2).item()
        max_error = torch.max(torch.abs(x - x_reconstructed)).item()
        
        assert mse < 1e-10, f"GPU 重构 MSE 过大: {mse}"
        assert max_error < 1e-5, f"GPU 重构最大误差过大: {max_error}"

    def test_reconstruction_consistency(self, device):
        """测试重构的一致性：多次重构应该得到相同结果"""
        B, N, T = 2, 2, 256
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, T).to(device)
        coeffs = swt(x)
        
        rec1 = swt.reconstruct(coeffs)
        rec2 = swt.reconstruct(coeffs)
        rec3 = swt.reconstruct(coeffs)
        
        assert torch.allclose(rec1, rec2, atol=1e-10)
        assert torch.allclose(rec2, rec3, atol=1e-10)

    # ---------- GPU/CPU 对比测试 ----------

    @pytest.mark.skipif(not PTWT_AVAILABLE, reason="需要 ptwt")
    def test_gpu_cpu_output_similarity(self, device):
        """测试 GPU 和 CPU 输出的相似性"""
        B, N, T = 4, 3, 128
        
        swt_gpu = SWTDecomposition(wavelet='db4', level=3).to(device)
        swt_cpu = SWTDecomposition(wavelet='db4', level=3)
        swt_cpu.use_gpu = False  # 强制 CPU
        
        x = torch.randn(B, N, T)
        
        coeffs_gpu = swt_gpu(x.to(device)).cpu()
        coeffs_cpu = swt_cpu(x)
        
        # GPU 使用 norm=False，CPU 使用 norm=True，所以会有差异
        # 但趋势应该一致
        correlation = torch.corrcoef(torch.stack([
            coeffs_gpu.flatten(), 
            coeffs_cpu.flatten()
        ]))[0, 1]
        
        assert correlation > 0.9, f"GPU/CPU 相关性过低: {correlation}"

    # ---------- 特殊输入测试 ----------

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
        
        x_const = torch.ones(B, N, T).to(device) * 5.0
        coeffs = swt(x_const)
        x_rec = swt.reconstruct(coeffs)
        
        mse = torch.mean((x_const - x_rec) ** 2).item()
        assert mse < 1e-10, f"常数重构误差: {mse}"

    def test_large_values(self, device):
        """测试大数值输入"""
        B, N, T = 2, 2, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, T).to(device) * 1e6
        coeffs = swt(x)
        x_rec = swt.reconstruct(coeffs)
        
        relative_error = torch.mean(torch.abs(x - x_rec) / (torch.abs(x) + 1e-8)).item()
        assert relative_error < 1e-5, f"大数值相对误差: {relative_error}"

    def test_small_values(self, device):
        """测试小数值输入"""
        B, N, T = 2, 2, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, T).to(device) * 1e-6
        coeffs = swt(x)
        x_rec = swt.reconstruct(coeffs)
        
        mse = torch.mean((x - x_rec) ** 2).item()
        assert mse < 1e-20, f"小数值 MSE: {mse}"

    # ---------- 批次独立性测试 ----------

    def test_batch_independence(self, device):
        """测试批次独立性：不同批次样本互不影响"""
        N, T = 3, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x1 = torch.randn(1, N, T).to(device)
        x2 = torch.randn(1, N, T).to(device)
        x_batch = torch.cat([x1, x2], dim=0)
        
        coeffs1 = swt(x1)
        coeffs2 = swt(x2)
        coeffs_batch = swt(x_batch)
        
        assert torch.allclose(coeffs_batch[0], coeffs1[0], atol=1e-7)
        assert torch.allclose(coeffs_batch[1], coeffs2[0], atol=1e-7)

    # ---------- 数据类型测试 ----------

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_different_dtypes(self, dtype, device):
        """测试不同数据类型"""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 需要 GPU")
        
        B, N, T = 2, 2, 128
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        x = torch.randn(B, N, T, dtype=dtype).to(device)
        coeffs = swt(x)
        
        assert not torch.isnan(coeffs).any()
        assert coeffs.dtype == torch.float32  # ptwt 要求 float32

    # ---------- 频率特性测试 ----------

    def test_frequency_separation(self, device):
        """测试频率分离特性"""
        B, N, T = 1, 1, 512
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
        
        # 低频信号
        t = torch.linspace(0, 4*np.pi, T).to(device)
        x_low = torch.sin(t).view(B, N, T)
        
        # 高频信号
        x_high = torch.sin(50 * t).view(B, N, T)
        
        coeffs_low = swt(x_low)
        coeffs_high = swt(x_high)
        
        # 低频信号应在近似系数(cA)有更多能量
        cA_energy_low = coeffs_low[:, :, :, 0].pow(2).sum()
        cD_energy_low = coeffs_low[:, :, :, 1:].pow(2).sum()
        
        # 高频信号应在细节系数(cD)有更多能量
        cA_energy_high = coeffs_high[:, :, :, 0].pow(2).sum()
        cD_energy_high = coeffs_high[:, :, :, 1:].pow(2).sum()
        
        assert cA_energy_low > cD_energy_low, "低频信号应在 cA 有更多能量"
        assert cD_energy_high > cA_energy_high, "高频信号应在 cD 有更多能量"


# ============================================================================
# WaveletPatchEmbedding 测试
# ============================================================================

class TestWaveletPatchEmbedding:
    """WaveletPatchEmbedding 层的全面测试"""

    # ---------- 基本功能测试 ----------

    @pytest.mark.parametrize("use_all_coeffs", [True, False])
    @pytest.mark.parametrize("batch_size,n_vars,seq_len", [
        (4, 3, 128),
        (8, 7, 256),
        (16, 10, 512),
    ])
    def test_output_shape(self, use_all_coeffs, batch_size, n_vars, seq_len, device):
        """测试输出形状"""
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
        
        assert output.shape == expected_shape, f"期望 {expected_shape}, 得到 {output.shape}"
        assert n_vars_out == n_vars
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("patch_len,stride", [
        (8, 4),
        (16, 8),
        (16, 16),   # 无重叠
        (32, 16),
        (32, 8),    # 大重叠
    ])
    def test_different_patch_configs(self, patch_len, stride, device):
        """测试不同的 patch 配置"""
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

    @pytest.mark.parametrize("d_model", [64, 128, 256, 512, 768])
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

    @pytest.mark.parametrize("swt_level", [1, 2, 3, 4])
    def test_different_swt_levels(self, swt_level, device):
        """测试不同的 SWT 分解层数"""
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
        
        assert not torch.isnan(output).any()

    # ---------- Dropout 测试 ----------

    def test_dropout_training_vs_eval(self, device):
        """测试训练模式和评估模式下的 dropout 行为"""
        B, N, T = 4, 3, 128
        
        model = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.5,
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
        
        assert diff_train > 0.01, "训练模式下 dropout 应有随机性"
        assert diff_eval < 1e-6, "评估模式下输出应确定"

    # ---------- 梯度测试 ----------

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
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含 NaN"

    # ---------- 模式对比测试 ----------

    def test_all_coeffs_vs_approx_only(self, device, seed):
        """测试全系数模式和近似系数模式的差异"""
        B, N, T = 4, 3, 256
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
        
        # 输出形状应相同
        assert output_all.shape == output_approx.shape
        assert n_vars_all == n_vars_approx

    # ---------- 特殊输入测试 ----------

    def test_single_variable(self, device):
        """测试单变量输入"""
        B, N, T = 8, 1, 256
        
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

    def test_single_batch(self, device):
        """测试单批次输入"""
        B, N, T = 1, 5, 256
        
        model = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.0,
            use_all_coeffs=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        output, n_vars = model(x)
        
        assert output.shape[0] == B * N


# ============================================================================
# ReplicationPad1d 测试
# ============================================================================

class TestReplicationPad1d:
    """ReplicationPad1d 层的全面测试"""

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
    def test_different_dimensions(self, ndim, device):
        """测试不同维度的输入"""
        pad = ReplicationPad1d((2, 3))
        
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
        """测试 2D 张量的填充"""
        pad = ReplicationPad1d((0, 5))
        x = torch.arange(20).reshape(4, 5).float().to(device)
        
        output = pad(x)
        
        assert output.shape == (4, 10)
        
        for i in range(4):
            assert all(output[i, -5:] == x[i, -1])

    def test_zero_padding(self, device):
        """测试零填充（无填充）"""
        pad = ReplicationPad1d((0, 0))
        x = torch.randn(3, 10).to(device)
        
        output = pad(x)
        
        assert torch.allclose(output, x)

    def test_gradient_flow(self, device):
        """测试梯度流"""
        pad = ReplicationPad1d((2, 3))
        x = torch.randn(3, 10, device=device, requires_grad=True)
        
        output = pad(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试：测试组件之间的配合"""

    def test_swt_to_embedding_pipeline(self, device):
        """测试 SWT 到 Embedding 的完整流程"""
        B, N, T = 4, 3, 256
        
        swt = SWTDecomposition(wavelet='db4', level=3).to(device)
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
        
        coeffs = swt(x)
        output, n_vars = model(x)
        
        assert coeffs.shape == (B, N, T, 4)
        assert output.shape[0] == B * N

    def test_with_transformer(self, device):
        """测试与 Transformer 的集成"""
        B, N, T = 8, 5, 256
        
        embedding = WaveletPatchEmbedding(
            d_model=128,
            patch_len=16,
            stride=8,
            dropout=0.1,
            use_all_coeffs=True
        ).to(device)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        x = torch.randn(B, N, T).to(device)
        
        embedded, n_vars = embedding(x)
        transformer_out = transformer_layer(embedded)
        
        assert transformer_out.shape == embedded.shape
        assert not torch.isnan(transformer_out).any()

    def test_end_to_end_gradient(self, device):
        """测试端到端的梯度流"""
        B, N, T = 4, 3, 128
        
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
                pooled = embedded.mean(dim=1)
                return self.fc(pooled)
        
        model = SimpleModel().to(device)
        x = torch.randn(B, N, T).to(device)
        target = torch.randn(B * N, 1).to(device)
        
        pred = model(x)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} 没有梯度"


# ============================================================================
# 性能测试
# ============================================================================

class TestPerformance:
    """性能测试"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
    def test_gpu_vs_cpu_speed(self):
        """测试 GPU 与 CPU 的速度对比"""
        B, N, T = 32, 7, 512
        
        swt_cpu = SWTDecomposition(wavelet='db4', level=3)
        swt_cpu.use_gpu = False
        
        swt_gpu = SWTDecomposition(wavelet='db4', level=3).cuda()
        
        x_cpu = torch.randn(B, N, T)
        x_gpu = x_cpu.cuda()
        
        # 预热
        _ = swt_cpu(x_cpu)
        torch.cuda.synchronize()
        _ = swt_gpu(x_gpu)
        torch.cuda.synchronize()
        
        # CPU 计时
        start = time.time()
        for _ in range(5):
            _ = swt_cpu(x_cpu)
        cpu_time = time.time() - start
        
        # GPU 计时
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            _ = swt_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"\nCPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, 加速比: {speedup:.1f}x")
        
        assert speedup > 1.0, "GPU 应该比 CPU 快"

    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
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


# ============================================================================
# 运行配置
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--color=yes",
    ])
