"""
验证 WaveletEmbed 集成到 TimeLLM 的测试脚本
"""
import torch
import torch.nn as nn
import sys
from types import SimpleNamespace


def create_test_config(use_swt=True):
    """创建测试配置"""
    configs = SimpleNamespace()
    
    # 基础配置
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
    configs.content = 'Test dataset'
    
    # SWT 配置
    configs.use_swt = use_swt
    configs.swt_wavelet = 'db4'
    configs.swt_level = 3
    configs.use_all_coeffs = True
    
    return configs


def test_import():
    """测试1: 导入模块"""
    print("=" * 70)
    print("测试 1: 导入模块")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        from layers.WaveletEmbed import WaveletPatchEmbedding
        print("✓ 成功导入 TimeLLM.Model")
        print("✓ 成功导入 WaveletPatchEmbedding")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation_with_swt():
    """测试2: 创建使用 SWT 的模型"""
    print("\n" + "=" * 70)
    print("测试 2: 创建使用 SWT 的模型")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        
        configs = create_test_config(use_swt=True)
        print(f"配置: use_swt={configs.use_swt}, wavelet={configs.swt_wavelet}, level={configs.swt_level}")
        
        model = Model(configs)
        
        # 检查是否使用了 WaveletPatchEmbedding
        from layers.WaveletEmbed import WaveletPatchEmbedding
        is_wavelet = isinstance(model.patch_embedding, WaveletPatchEmbedding)
        
        if is_wavelet:
            print("✓ 模型成功使用 WaveletPatchEmbedding")
            print(f"  - 小波基: {model.patch_embedding.swt.wavelet}")
            print(f"  - 分解层数: {model.patch_embedding.swt.level}")
            print(f"  - 使用全系数: {model.patch_embedding.use_all_coeffs}")
            return True
        else:
            print("✗ 模型未使用 WaveletPatchEmbedding")
            return False
            
    except Exception as e:
        print(f"✗ 创建模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation_without_swt():
    """测试3: 创建不使用 SWT 的模型"""
    print("\n" + "=" * 70)
    print("测试 3: 创建不使用 SWT 的模型")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        from layers.Embed import PatchEmbedding
        
        configs = create_test_config(use_swt=False)
        print(f"配置: use_swt={configs.use_swt}")
        
        model = Model(configs)
        
        # 检查是否使用了原始 PatchEmbedding
        is_original = isinstance(model.patch_embedding, PatchEmbedding)
        
        if is_original:
            print("✓ 模型成功使用原始 PatchEmbedding")
            return True
        else:
            print("✗ 模型未使用原始 PatchEmbedding")
            return False
            
    except Exception as e:
        print(f"✗ 创建模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_with_swt():
    """测试4: 使用 SWT 的前向传播"""
    print("\n" + "=" * 70)
    print("测试 4: 使用 SWT 的前向传播")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        
        configs = create_test_config(use_swt=True)
        model = Model(configs)
        model.eval()
        
        # 创建测试输入
        B, T, N = 2, configs.seq_len, configs.enc_in
        x_enc = torch.randn(B, T, N)
        x_mark_enc = torch.randn(B, T, 4)  # 时间特征
        x_dec = torch.randn(B, configs.pred_len, N)
        x_mark_dec = torch.randn(B, configs.pred_len, 4)
        
        print(f"输入形状: x_enc={x_enc.shape}, x_dec={x_dec.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (B, configs.pred_len, N)
        print(f"输出形状: {output.shape}")
        print(f"期望形状: {expected_shape}")
        
        if output.shape == expected_shape:
            print("✓ 前向传播成功，输出形状正确")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"  输出均值: {output.mean():.4f}")
            print(f"  输出标准差: {output.std():.4f}")
            return True
        else:
            print(f"✗ 输出形状不匹配: 期望{expected_shape}, 得到{output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_without_swt():
    """测试5: 不使用 SWT 的前向传播"""
    print("\n" + "=" * 70)
    print("测试 5: 不使用 SWT 的前向传播")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        
        configs = create_test_config(use_swt=False)
        model = Model(configs)
        model.eval()
        
        # 创建测试输入
        B, T, N = 2, configs.seq_len, configs.enc_in
        x_enc = torch.randn(B, T, N)
        x_mark_enc = torch.randn(B, T, 4)
        x_dec = torch.randn(B, configs.pred_len, N)
        x_mark_dec = torch.randn(B, configs.pred_len, 4)
        
        print(f"输入形状: x_enc={x_enc.shape}, x_dec={x_dec.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (B, configs.pred_len, N)
        print(f"输出形状: {output.shape}")
        print(f"期望形状: {expected_shape}")
        
        if output.shape == expected_shape:
            print("✓ 前向传播成功，输出形状正确")
            print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"  输出均值: {output.mean():.4f}")
            print(f"  输出标准差: {output.std():.4f}")
            return True
        else:
            print(f"✗ 输出形状不匹配: 期望{expected_shape}, 得到{output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_wavelets():
    """测试6: 不同小波基"""
    print("\n" + "=" * 70)
    print("测试 6: 不同小波基")
    print("=" * 70)
    
    wavelets = ['db4', 'db6', 'sym4', 'coif2', 'haar']
    results = []
    
    try:
        from models.TimeLLM import Model
        
        for wavelet in wavelets:
            try:
                configs = create_test_config(use_swt=True)
                configs.swt_wavelet = wavelet
                
                model = Model(configs)
                model.eval()
                
                # 简单前向传播
                B, T, N = 2, configs.seq_len, configs.enc_in
                x_enc = torch.randn(B, T, N)
                x_mark_enc = torch.randn(B, T, 4)
                x_dec = torch.randn(B, configs.pred_len, N)
                x_mark_dec = torch.randn(B, configs.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                print(f"  {wavelet:8s}: ✓ 正常工作")
                results.append(True)
                
            except Exception as e:
                print(f"  {wavelet:8s}: ✗ 失败 - {e}")
                results.append(False)
        
        if all(results):
            print("✓ 所有小波基测试通过")
            return True
        else:
            print(f"✗ 部分小波基测试失败 ({sum(results)}/{len(results)} 通过)")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_different_levels():
    """测试7: 不同分解层数"""
    print("\n" + "=" * 70)
    print("测试 7: 不同分解层数")
    print("=" * 70)
    
    levels = [1, 2, 3, 4, 5]
    results = []
    
    try:
        from models.TimeLLM import Model
        
        for level in levels:
            try:
                configs = create_test_config(use_swt=True)
                configs.swt_level = level
                
                model = Model(configs)
                model.eval()
                
                # 简单前向传播
                B, T, N = 2, configs.seq_len, configs.enc_in
                x_enc = torch.randn(B, T, N)
                x_mark_enc = torch.randn(B, T, 4)
                x_dec = torch.randn(B, configs.pred_len, N)
                x_mark_dec = torch.randn(B, configs.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                print(f"  Level {level}: ✓ 正常工作")
                results.append(True)
                
            except Exception as e:
                print(f"  Level {level}: ✗ 失败 - {e}")
                results.append(False)
        
        if all(results):
            print("✓ 所有分解层数测试通过")
            return True
        else:
            print(f"✗ 部分分解层数测试失败 ({sum(results)}/{len(results)} 通过)")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_gradient_flow():
    """测试8: 梯度反向传播"""
    print("\n" + "=" * 70)
    print("测试 8: 梯度反向传播")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        
        configs = create_test_config(use_swt=True)
        model = Model(configs)
        model.train()
        
        # 创建测试输入
        B, T, N = 2, configs.seq_len, configs.enc_in
        x_enc = torch.randn(B, T, N)
        x_mark_enc = torch.randn(B, T, 4)
        x_dec = torch.randn(B, configs.pred_len, N)
        x_mark_dec = torch.randn(B, configs.pred_len, 4)
        target = torch.randn(B, configs.pred_len, N)
        
        # 前向传播
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 计算损失
        loss = nn.MSELoss()(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_count = 0
        total_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_count += 1
                if param.grad is not None:
                    grad_count += 1
        
        print(f"  可训练参数数: {total_count}")
        print(f"  有梯度的参数: {grad_count}")
        print(f"  损失值: {loss.item():.6f}")
        
        if grad_count > 0:
            print("✓ 梯度反向传播成功")
            return True
        else:
            print("✗ 没有参数接收到梯度")
            return False
            
    except Exception as e:
        print(f"✗ 梯度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """测试9: 对比 SWT vs 原始 Embedding"""
    print("\n" + "=" * 70)
    print("测试 9: SWT vs 原始 Embedding 对比")
    print("=" * 70)
    
    try:
        from models.TimeLLM import Model
        
        # 创建相同的输入
        B, T, N = 2, 96, 7
        x_enc = torch.randn(B, T, N)
        x_mark_enc = torch.randn(B, T, 4)
        x_dec = torch.randn(B, 96, N)
        x_mark_dec = torch.randn(B, 96, 4)
        
        # 使用 SWT 的模型
        configs_swt = create_test_config(use_swt=True)
        model_swt = Model(configs_swt)
        model_swt.eval()
        
        with torch.no_grad():
            output_swt = model_swt(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 使用原始 Embedding 的模型
        configs_orig = create_test_config(use_swt=False)
        model_orig = Model(configs_orig)
        model_orig.eval()
        
        with torch.no_grad():
            output_orig = model_orig(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 对比结果
        print(f"\nSWT 模型:")
        print(f"  输出形状: {output_swt.shape}")
        print(f"  输出范围: [{output_swt.min():.4f}, {output_swt.max():.4f}]")
        print(f"  输出均值: {output_swt.mean():.4f}")
        print(f"  输出标准差: {output_swt.std():.4f}")
        
        print(f"\n原始模型:")
        print(f"  输出形状: {output_orig.shape}")
        print(f"  输出范围: [{output_orig.min():.4f}, {output_orig.max():.4f}]")
        print(f"  输出均值: {output_orig.mean():.4f}")
        print(f"  输出标准差: {output_orig.std():.4f}")
        
        print(f"\n对比:")
        diff = torch.mean(torch.abs(output_swt - output_orig)).item()
        print(f"  平均绝对差异: {diff:.6f}")
        
        if output_swt.shape == output_orig.shape:
            print("✓ 两种模型输出形状一致，集成成功")
            return True
        else:
            print("✗ 输出形状不一致")
            return False
            
    except Exception as e:
        print(f"✗ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("WaveletEmbed 集成到 TimeLLM 验证测试套件")
    print("=" * 70 + "\n")
    
    tests = [
        ("导入模块", test_import),
        ("创建使用 SWT 的模型", test_model_creation_with_swt),
        ("创建不使用 SWT 的模型", test_model_creation_without_swt),
        ("使用 SWT 的前向传播", test_forward_pass_with_swt),
        ("不使用 SWT 的前向传播", test_forward_pass_without_swt),
        ("不同小波基", test_different_wavelets),
        ("不同分解层数", test_different_levels),
        ("梯度反向传播", test_gradient_flow),
        ("SWT vs 原始 Embedding 对比", test_comparison),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    
    print("\n详细结果:")
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status:8s} - {test_name}")
    
    if passed == total:
        print("\n" + "=" * 70)
        print("🎉 所有测试通过！WaveletEmbed 已成功集成到 TimeLLM！")
        print("=" * 70)
        return True
    else:
        print("\n" + "=" * 70)
        print(f"⚠️  部分测试失败 ({passed}/{total} 通过)")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
