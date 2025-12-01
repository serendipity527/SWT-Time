"""
WaveletEmbed 模块性能测试
测试 SWT 分解、重构和 Patch Embedding 的时间开销
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import time
from layers.WaveletEmbed import SWTDecomposition, WaveletPatchEmbedding


def warmup_gpu():
    """GPU 预热"""
    if torch.cuda.is_available():
        x = torch.randn(100, 100, device='cuda')
        for _ in range(10):
            _ = x @ x
        torch.cuda.synchronize()


def test_swt_decomposition_time():
    """测试 SWT 分解时间"""
    print("\n" + "=" * 60)
    print("SWT 分解性能测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    configs = [
        # (batch, n_vars, seq_len, level)
        (32, 7, 512, 3),
        (32, 7, 1024, 3),
        (64, 7, 512, 3),
        (32, 21, 512, 3),
        (32, 7, 512, 5),
    ]
    
    warmup_gpu()
    
    for batch, n_vars, seq_len, level in configs:
        swt = SWTDecomposition(wavelet='db4', level=level).to(device)
        x = torch.randn(batch, n_vars, seq_len, device=device)
        
        # 预热
        with torch.no_grad():
            _ = swt(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        n_runs = 50
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                coeffs = swt(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000
        
        print(f"B={batch:2d}, N={n_vars:2d}, T={seq_len:4d}, L={level} | "
              f"分解: {elapsed:.3f} ms | 输出: {tuple(coeffs.shape)}")


def test_swt_reconstruction_time():
    """测试 SWT 重构时间"""
    print("\n" + "=" * 60)
    print("SWT 重构性能测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        (32, 7, 512, 3),
        (32, 7, 1024, 3),
        (64, 7, 512, 3),
    ]
    
    warmup_gpu()
    
    for batch, n_vars, seq_len, level in configs:
        swt = SWTDecomposition(wavelet='db4', level=level).to(device)
        x = torch.randn(batch, n_vars, seq_len, device=device)
        
        with torch.no_grad():
            coeffs = swt(x)
        
        # 预热
        with torch.no_grad():
            _ = swt.reconstruct(coeffs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                recon = swt.reconstruct(coeffs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000
        
        print(f"B={batch:2d}, N={n_vars:2d}, T={seq_len:4d}, L={level} | "
              f"重构: {elapsed:.3f} ms")


def test_wavelet_patch_embedding_time():
    """测试 WaveletPatchEmbedding 时间"""
    print("\n" + "=" * 60)
    print("WaveletPatchEmbedding 性能测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    configs = [
        # (batch, n_vars, seq_len, patch_len, stride, d_model, use_all_coeffs)
        (32, 7, 512, 16, 8, 768, True),
        (32, 7, 512, 16, 8, 768, False),
        (32, 7, 1024, 16, 8, 768, True),
        (64, 7, 512, 16, 8, 768, True),
        (32, 21, 512, 16, 8, 768, True),
    ]
    
    warmup_gpu()
    
    for batch, n_vars, seq_len, patch_len, stride, d_model, use_all in configs:
        embed = WaveletPatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            dropout=0.1,
            wavelet='db4',
            swt_level=3,
            use_all_coeffs=use_all
        ).to(device)
        embed.eval()
        
        x = torch.randn(batch, n_vars, seq_len, device=device)
        
        # 预热
        with torch.no_grad():
            _ = embed(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                out, _ = embed(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs * 1000
        
        mode = "全系数" if use_all else "仅近似"
        print(f"B={batch:2d}, N={n_vars:2d}, T={seq_len:4d}, P={patch_len}, S={stride} | "
              f"{mode} | {elapsed:.3f} ms | 输出: {tuple(out.shape)}")


def test_end_to_end_time():
    """端到端性能测试（模拟实际使用场景）"""
    print("\n" + "=" * 60)
    print("端到端性能测试（分解 + Embedding）")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 典型 TimeLLM 配置
    batch, n_vars, seq_len = 32, 7, 512
    patch_len, stride, d_model = 16, 8, 768
    
    embed = WaveletPatchEmbedding(
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        dropout=0.0,
        wavelet='db4',
        swt_level=3,
        use_all_coeffs=True
    ).to(device)
    embed.eval()
    
    x = torch.randn(batch, n_vars, seq_len, device=device)
    
    warmup_gpu()
    
    # 预热
    with torch.no_grad():
        _ = embed(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 多次运行取平均
    n_runs = 100
    times = []
    
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            out, _ = embed(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"配置: B={batch}, N={n_vars}, T={seq_len}, P={patch_len}, S={stride}, D={d_model}")
    print(f"运行次数: {n_runs}")
    print(f"平均时间: {avg_time:.3f} ms")
    print(f"最小时间: {min_time:.3f} ms")
    print(f"最大时间: {max_time:.3f} ms")
    print(f"吞吐量:   {batch * 1000 / avg_time:.1f} samples/s")


if __name__ == "__main__":
    print("WaveletEmbed 性能测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    test_swt_decomposition_time()
    test_swt_reconstruction_time()
    test_wavelet_patch_embedding_time()
    test_end_to_end_time()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
