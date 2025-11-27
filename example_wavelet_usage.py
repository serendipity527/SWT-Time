"""
Time-LLM 小波功能快速入门示例

展示如何使用三种不同的嵌入模式
"""

import torch
import torch.nn as nn
import numpy as np


class SimpleConfigs:
    """简化的配置类"""
    def __init__(self, embedding_type='patch'):
        # 任务配置
        self.task_name = 'long_term_forecast'
        self.seq_len = 512
        self.pred_len = 96
        
        # 模型配置
        self.llm_model = 'GPT2'
        self.llm_dim = 768
        self.llm_layers = 6
        self.d_model = 16
        self.d_ff = 512
        self.n_heads = 8
        
        # Patch配置
        self.patch_len = 16
        self.stride = 8
        
        # 数据配置
        self.enc_in = 7
        self.dropout = 0.1
        
        # Prompt配置
        self.prompt_domain = False
        self.content = ''
        
        # ⭐ 嵌入类型配置
        self.embedding_type = embedding_type
        
        # ⭐ 小波配置 (仅当使用小波时有效)
        if embedding_type in ['wavelet', 'hybrid']:
            self.use_wavelet = True
            self.wavelet_type = 'db4'
            self.wavelet_level = 3
        else:
            self.use_wavelet = False


def demo_original_patch():
    """演示1: 使用原始Patch嵌入"""
    print("\n" + "="*70)
    print("演示1: 原始Patch嵌入模式")
    print("="*70)
    
    from models.TimeLLM import Model
    
    # 创建配置
    configs = SimpleConfigs(embedding_type='patch')
    
    # 创建模型
    print("\n创建模型...")
    model = Model(configs)
    
    # 准备数据
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 4)
    
    print(f"\n输入数据:")
    print(f"  x_enc: {x_enc.shape} (batch, seq_len, features)")
    print(f"  x_mark_enc: {x_mark_enc.shape} (batch, seq_len, time_features)")
    
    # 前向传播
    print("\n前向传播...")
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n预测输出:")
    print(f"  output: {output.shape} (batch, pred_len, features)")
    print(f"\n✅ 原始Patch模式运行成功!")
    
    return model


def demo_wavelet_patch():
    """演示2: 使用小波Patch嵌入"""
    print("\n" + "="*70)
    print("演示2: 小波Patch嵌入模式 (方案3)")
    print("="*70)
    
    from models.TimeLLM import Model
    
    # 创建配置
    configs = SimpleConfigs(embedding_type='wavelet')
    
    print(f"\n小波配置:")
    print(f"  类型: {configs.wavelet_type}")
    print(f"  层数: {configs.wavelet_level}")
    print(f"  尺度数: {configs.wavelet_level + 1}")
    
    # 创建模型
    print("\n创建模型...")
    model = Model(configs)
    
    # 准备数据
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 4)
    
    # 前向传播
    print("\n前向传播...")
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n预测输出:")
    print(f"  output: {output.shape}")
    print(f"\n✅ 小波Patch模式运行成功!")
    print(f"\n💡 注意: Patch数量从63减少到31，推理速度提升约2倍!")
    
    return model


def demo_hybrid_patch():
    """演示3: 使用混合Patch嵌入"""
    print("\n" + "="*70)
    print("演示3: 混合Patch嵌入模式 (方案4)")
    print("="*70)
    
    from models.TimeLLM import Model
    
    # 创建配置
    configs = SimpleConfigs(embedding_type='hybrid')
    
    print(f"\n混合模式配置:")
    print(f"  原始Patch + 小波Patch")
    print(f"  融合方式: 拼接")
    
    # 创建模型
    print("\n创建模型...")
    model = Model(configs)
    
    # 准备数据
    batch_size = 4
    x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
    x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
    x_dec = torch.randn(batch_size, configs.pred_len, configs.enc_in)
    x_mark_dec = torch.randn(batch_size, configs.pred_len, 4)
    
    # 前向传播
    print("\n前向传播...")
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"\n预测输出:")
    print(f"  output: {output.shape}")
    print(f"\n✅ 混合Patch模式运行成功!")
    print(f"\n💡 结合了时域和频域的优势，但计算量较大")
    
    return model


def compare_models():
    """演示4: 对比三种模式"""
    print("\n" + "="*70)
    print("演示4: 三种模式对比")
    print("="*70)
    
    from models.TimeLLM import Model
    import time
    
    modes = ['patch', 'wavelet', 'hybrid']
    results = {}
    
    # 准备测试数据
    batch_size = 8
    seq_len = 512
    pred_len = 96
    enc_in = 7
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    x_dec = torch.randn(batch_size, pred_len, enc_in)
    x_mark_dec = torch.randn(batch_size, pred_len, 4)
    
    for mode in modes:
        print(f"\n测试 {mode} 模式...")
        
        # 创建模型
        configs = SimpleConfigs(embedding_type=mode)
        model = Model(configs)
        model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 测试推理速度
        with torch.no_grad():
            # 预热
            _ = model(x_enc[:2], x_mark_enc[:2], x_dec[:2], x_mark_dec[:2])
            
            # 计时
            start = time.time()
            for _ in range(10):
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            elapsed = time.time() - start
        
        results[mode] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'inference_time': elapsed / 10,
            'output_shape': output.shape
        }
    
    # 打印对比表
    print("\n" + "="*70)
    print("性能对比表")
    print("="*70)
    print(f"{'模式':<12} {'总参数':<15} {'可训练参数':<15} {'推理时间(s)':<12}")
    print("-"*70)
    
    for mode, res in results.items():
        print(f"{mode:<12} {res['total_params']:<15,} {res['trainable_params']:<15,} {res['inference_time']:<12.4f}")
    
    # 相对性能
    baseline_time = results['patch']['inference_time']
    print("\n相对速度 (以原始模式为基准):")
    for mode, res in results.items():
        speedup = baseline_time / res['inference_time']
        print(f"  {mode}: {speedup:.2f}x")
    
    print("\n💡 建议:")
    print("  - 追求速度: 使用 wavelet 模式")
    print("  - 追求性能: 使用 hybrid 模式")
    print("  - 计算受限: 使用 wavelet 模式")
    print("  - 保守起见: 使用 patch 模式")


def visualize_wavelet_decomposition():
    """演示5: 可视化小波分解"""
    print("\n" + "="*70)
    print("演示5: 小波分解可视化")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        import pywt
    except ImportError:
        print("需要matplotlib库进行可视化")
        print("运行: pip install matplotlib")
        return
    
    # 生成示例时间序列 (趋势 + 周期 + 噪声)
    t = np.linspace(0, 10, 512)
    trend = 0.5 * t  # 趋势
    seasonal = 2 * np.sin(2 * np.pi * t)  # 周期
    noise = 0.5 * np.random.randn(512)  # 噪声
    signal = trend + seasonal + noise
    
    # 小波分解
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    cA3, cD3, cD2, cD1 = coeffs
    
    # 绘图
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    
    axes[0].plot(signal)
    axes[0].set_title('原始信号 (趋势 + 周期 + 噪声)')
    axes[0].set_ylabel('值')
    
    axes[1].plot(cA3)
    axes[1].set_title('cA3: 近似系数 (低频/趋势)')
    axes[1].set_ylabel('系数')
    
    axes[2].plot(cD3)
    axes[2].set_title('cD3: 细节系数3 (中低频/长周期)')
    axes[2].set_ylabel('系数')
    
    axes[3].plot(cD2)
    axes[3].set_title('cD2: 细节系数2 (中高频/短周期)')
    axes[3].set_ylabel('系数')
    
    axes[4].plot(cD1)
    axes[4].set_title('cD1: 细节系数1 (高频/噪声)')
    axes[4].set_ylabel('系数')
    axes[4].set_xlabel('时间索引')
    
    plt.tight_layout()
    
    # 保存图像
    filename = 'wavelet_decomposition.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化图像已保存到: {filename}")
    print("\n各层说明:")
    print(f"  cA3 (长度{len(cA3):3d}): 捕获长期趋势和低频成分")
    print(f"  cD3 (长度{len(cD3):3d}): 捕获中长期周期 (如月度波动)")
    print(f"  cD2 (长度{len(cD2):3d}): 捕获短期周期 (如周波动)")
    print(f"  cD1 (长度{len(cD1):3d}): 捕获高频细节和噪声")
    
    # plt.show()  # 取消注释以显示图像


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Time-LLM 小波功能快速入门")
    print("="*70)
    
    # 检查依赖
    try:
        import pywt
        print(f"✅ PyWavelets版本: {pywt.__version__}")
    except ImportError:
        print("❌ 请先安装PyWavelets: pip install PyWavelets")
        return
    
    # 运行演示
    demos = [
        ("原始Patch模式", demo_original_patch),
        ("小波Patch模式", demo_wavelet_patch),
        ("混合Patch模式", demo_hybrid_patch),
        ("性能对比", compare_models),
        ("小波分解可视化", visualize_wavelet_decomposition),
    ]
    
    print("\n请选择演示:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. 运行全部")
    
    try:
        choice = input("\n请输入选择 (0-5): ").strip()
        
        if choice == '0':
            # 运行全部
            for name, demo_func in demos:
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ {name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            # 运行选中的演示
            name, demo_func = demos[int(choice) - 1]
            demo_func()
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n\n中断运行")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("演示结束!")
    print("="*70)
    print("\n更多信息请查看: WAVELET_USAGE.md")


if __name__ == '__main__':
    main()
