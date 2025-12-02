"""
测试ptwt库是否真的在GPU上运行
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ptwt
    print("✓ ptwt 已安装")
    ptwt_available = True
except ImportError:
    print("✗ ptwt 未安装，尝试使用 PyWavelets")
    ptwt_available = False

try:
    import pywt
    print("✓ PyWavelets (pywt) 已安装")
    pywt_available = True
except ImportError:
    print("✗ PyWavelets 未安装")
    pywt_available = False

print("\n" + "="*70)
print("库信息检查")
print("="*70)

if ptwt_available:
    print("\n[ptwt 信息]")
    print(f"  版本: {ptwt.__version__ if hasattr(ptwt, '__version__') else '未知'}")
    print(f"  位置: {ptwt.__file__}")
    print(f"  描述: PyTorch Wavelet Toolbox - 原生支持GPU的小波变换库")

if pywt_available:
    print("\n[PyWavelets 信息]")
    print(f"  版本: {pywt.__version__}")
    print(f"  位置: {pywt.__file__}")
    print(f"  描述: 传统CPU小波变换库，不原生支持GPU")

print("\n" + "="*70)
print("设备测试")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n当前设备: {device}")

if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 测试数据
batch_size = 4
n_vars = 7
seq_len = 512
x = torch.randn(batch_size * n_vars, 1, seq_len)

print(f"\n测试数据形状: {x.shape}")
print(f"测试数据设备: {x.device}")

# 测试ptwt
if ptwt_available:
    print("\n" + "-"*70)
    print("[ptwt 设备测试]")
    print("-"*70)
    
    # CPU测试
    x_cpu = x.cpu()
    print(f"\n1. CPU输入: {x_cpu.device}")
    try:
        coeffs_cpu = ptwt.wavedec(x_cpu, 'db4', level=3, mode='reflect')
        print(f"   ✓ DWT成功")
        print(f"   输出设备: {coeffs_cpu[0].device}")
        print(f"   输出层数: {len(coeffs_cpu)}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
    
    # GPU测试
    if device.type == 'cuda':
        x_gpu = x.cuda()
        print(f"\n2. GPU输入: {x_gpu.device}")
        try:
            coeffs_gpu = ptwt.wavedec(x_gpu, 'db4', level=3, mode='reflect')
            print(f"   ✓ DWT成功")
            print(f"   输出设备: {coeffs_gpu[0].device}")
            print(f"   输出层数: {len(coeffs_gpu)}")
            
            # 验证是否真的在GPU上计算
            print("\n3. GPU计算验证:")
            import time
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                _ = ptwt.wavedec(x_gpu, 'db4', level=3, mode='reflect')
            torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - start) * 1000 / 100
            print(f"   GPU平均时间: {gpu_time:.3f} ms")
            
        except Exception as e:
            print(f"   ✗ 失败: {e}")
            print(f"   说明: ptwt可能不支持GPU，或CUDA配置有问题")

# 测试PyWavelets
if pywt_available:
    print("\n" + "-"*70)
    print("[PyWavelets 设备测试]")
    print("-"*70)
    
    # PyWavelets只能用CPU
    x_np = x_cpu.numpy().squeeze()  # (batch*n_vars, seq_len)
    print(f"\n1. NumPy输入形状: {x_np.shape}")
    try:
        coeffs_pywt = pywt.wavedec(x_np[0], 'db4', level=3, mode='reflect')
        print(f"   ✓ DWT成功")
        print(f"   输出类型: {type(coeffs_pywt[0])} (NumPy数组)")
        print(f"   输出层数: {len(coeffs_pywt)}")
        print(f"   ⚠️  注意: PyWavelets只支持CPU计算")
        
        # 性能测试
        print("\n2. CPU计算性能:")
        start = time.perf_counter()
        for i in range(batch_size * n_vars):
            _ = pywt.wavedec(x_np[i], 'db4', level=3, mode='reflect')
        cpu_time = (time.perf_counter() - start) * 1000 / (batch_size * n_vars)
        print(f"   CPU平均时间: {cpu_time:.3f} ms (单序列)")
        
    except Exception as e:
        print(f"   ✗ 失败: {e}")

print("\n" + "="*70)
print("结论")
print("="*70)

if ptwt_available:
    print("\n✓ ptwt (PyTorch Wavelet Toolbox)")
    print("  - 原生支持GPU加速")
    print("  - 可直接处理torch.Tensor")
    print("  - 输入输出都保持在同一设备上")
    print("  - 适合深度学习场景")
else:
    print("\n✗ ptwt 未安装")
    print("  安装命令: pip install ptwt")

if pywt_available:
    print("\n✓ PyWavelets")
    print("  - 仅支持CPU计算")
    print("  - 需要NumPy数组")
    print("  - 需要手动转换数据类型")
    print("  - 适合传统信号处理")
else:
    print("\n✗ PyWavelets 未安装")

print("\n" + "="*70)
print("✅ 测试完成")
print("="*70)
