"""
快速验证 WaveletEmbed 集成
5分钟验证脚本
"""
import torch
from types import SimpleNamespace

print("🔍 快速验证 WaveletEmbed 集成到 TimeLLM...")
print("=" * 60)

# 步骤1: 导入
print("\n步骤 1/4: 导入模块...")
try:
    from models.TimeLLM import Model
    from layers.WaveletEmbed import WaveletPatchEmbedding
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 步骤2: 创建配置
print("\n步骤 2/4: 创建配置...")
configs = SimpleNamespace()
configs.task_name = 'long_term_forecast'
configs.seq_len = 96
configs.pred_len = 96
configs.enc_in = 7
configs.d_model = 16
configs.d_ff = 32
configs.n_heads = 4
configs.llm_layers = 2
configs.llm_dim = 768
configs.dropout = 0.1
configs.patch_len = 16
configs.stride = 8
configs.llm_model = 'GPT2'
configs.prompt_domain = False

# SWT 配置
configs.use_swt = True
configs.swt_wavelet = 'db4'
configs.swt_level = 3
configs.use_all_coeffs = True

print(f"✓ 配置创建成功 (use_swt={configs.use_swt})")

# 步骤3: 创建模型
print("\n步骤 3/4: 创建模型...")
try:
    model = Model(configs)
    
    # 验证是否使用了 WaveletPatchEmbedding
    if isinstance(model.patch_embedding, WaveletPatchEmbedding):
        print("✓ 模型成功使用 WaveletPatchEmbedding")
        print(f"  - 小波基: {model.patch_embedding.swt.wavelet}")
        print(f"  - 分解层数: {model.patch_embedding.swt.level}")
    else:
        print("✗ 模型未使用 WaveletPatchEmbedding")
        exit(1)
except Exception as e:
    print(f"✗ 创建模型失败: {e}")
    exit(1)

# 步骤4: 前向传播测试
print("\n步骤 4/4: 测试前向传播...")
try:
    model.eval()
    
    # 创建测试输入
    B, T, N = 2, configs.seq_len, configs.enc_in
    x_enc = torch.randn(B, T, N)
    x_mark_enc = torch.randn(B, T, 4)
    x_dec = torch.randn(B, configs.pred_len, N)
    x_mark_dec = torch.randn(B, configs.pred_len, 4)
    
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    expected_shape = (B, configs.pred_len, N)
    
    if output.shape == expected_shape:
        print(f"✓ 前向传播成功")
        print(f"  输入: {x_enc.shape}")
        print(f"  输出: {output.shape}")
        print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    else:
        print(f"✗ 输出形状不匹配: 期望{expected_shape}, 得到{output.shape}")
        exit(1)
        
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 成功
print("\n" + "=" * 60)
print("🎉 验证成功！WaveletEmbed 已正确集成到 TimeLLM！")
print("=" * 60)
print("\n下一步:")
print("  1. 运行完整测试: python test_timellm_integration.py")
print("  2. 在实际数据上训练模型")
print("  3. 对比 use_swt=True 和 use_swt=False 的性能")
