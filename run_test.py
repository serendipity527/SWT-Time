"""
独立测试脚本：加载训练好的模型并在测试集上进行评估
"""
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm

from accelerate import Accelerator
from torch import nn

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import load_content
from utils.metrics import metric

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def parse_args():
    parser = argparse.ArgumentParser(description='SWT-Time Testing')
    
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='ETTh1_96_96', help='model id')
    parser.add_argument('--model_comment', type=str, default='gpu1_test', help='model comment')
    parser.add_argument('--model', type=str, default='TimeLLM', help='model name')
    
    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    
    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='eval batch size')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    
    # SWT parameters
    parser.add_argument('--use_swt', action='store_true', help='use SWT-Patch Embedding', default=False)
    parser.add_argument('--swt_wavelet', type=str, default='db4', help='wavelet type')
    parser.add_argument('--swt_level', type=int, default=3, help='SWT decomposition level')
    parser.add_argument('--use_all_coeffs', action='store_true', help='use all coefficients', default=True)
    
    # DWT Prompt parameters
    parser.add_argument('--use_dwt_prompt', action='store_true', help='use DWT dynamic prompt', default=False)
    parser.add_argument('--dwt_prompt_level', type=int, default=3, help='DWT decomposition level for prompt')
    parser.add_argument('--prompt_compression', type=str, default='balanced', 
                        choices=['minimal', 'balanced', 'detailed'], help='prompt compression level')
    
    return parser.parse_args()


def test(args, accelerator, model, test_loader, criterion, mae_metric):
    """在测试集上评估模型"""
    preds = []
    trues = []
    total_loss = []
    total_mae_loss = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), desc="Testing"):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            
            loss = criterion(outputs, batch_y.to(accelerator.device))
            mae_loss = mae_metric(outputs, batch_y.to(accelerator.device))
            
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 计算评估指标
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mspe': mspe,
        'avg_loss': np.average(total_loss),
        'avg_mae_loss': np.average(total_mae_loss)
    }, preds, trues


def main():
    args = parse_args()
    
    # 初始化 accelerator（测试时不使用 DeepSpeed）
    accelerator = Accelerator()
    
    # 构建 setting 字符串（与训练时一致）
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, 0)  # 使用 iteration 0
    
    checkpoint_path = os.path.join(args.checkpoints, setting + '-' + args.model_comment, 'checkpoint')
    
    accelerator.print(f"="*60)
    accelerator.print(f"正在加载模型: {checkpoint_path}")
    accelerator.print(f"="*60)
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(checkpoint_path):
        accelerator.print(f"错误: 找不到 checkpoint 文件: {checkpoint_path}")
        accelerator.print("\n可用的 checkpoints:")
        if os.path.exists(args.checkpoints):
            for d in os.listdir(args.checkpoints):
                accelerator.print(f"  - {d}")
        return
    
    # 加载测试数据
    test_data, test_loader = data_provider(args, 'test')
    accelerator.print(f"测试集大小: {len(test_data)}")
    
    # 创建模型
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()
    
    args.content = load_content(args)
    
    # 加载模型权重
    accelerator.print("正在加载模型权重...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 使用strict=False允许部分权重不匹配（兼容新旧版本）
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        accelerator.print(f"⚠️  Missing keys: {missing_keys[:5]}...")  # 只显示前5个
    if unexpected_keys:
        accelerator.print(f"⚠️  Unexpected keys: {unexpected_keys[:5]}...")  # 只显示前5个
    
    accelerator.print("模型权重加载成功!")
    
    # 准备模型和数据
    test_loader, model = accelerator.prepare(test_loader, model)
    
    # 定义损失函数
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    
    # 运行测试
    accelerator.print("\n开始测试...")
    metrics, preds, trues = test(args, accelerator, model, test_loader, criterion, mae_metric)
    
    # 打印结果
    accelerator.print("\n" + "="*60)
    accelerator.print("测试结果:")
    accelerator.print("="*60)
    accelerator.print(f"MSE:  {metrics['mse']:.6f}")
    accelerator.print(f"MAE:  {metrics['mae']:.6f}")
    accelerator.print(f"RMSE: {metrics['rmse']:.6f}")
    accelerator.print(f"MAPE: {metrics['mape']:.6f}")
    accelerator.print(f"MSPE: {metrics['mspe']:.6f}")
    accelerator.print("="*60)
    
    # 保存预测结果
    result_dir = './results/' + setting + '-' + args.model_comment
    if not os.path.exists(result_dir) and accelerator.is_local_main_process:
        os.makedirs(result_dir)
    
    if accelerator.is_local_main_process:
        np.save(os.path.join(result_dir, 'pred.npy'), preds)
        np.save(os.path.join(result_dir, 'true.npy'), trues)
        accelerator.print(f"\n预测结果已保存到: {result_dir}")
        
        # 保存指标到文本文件
        with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
            f.write(f"MSE: {metrics['mse']:.6f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"MAPE: {metrics['mape']:.6f}\n")
            f.write(f"MSPE: {metrics['mspe']:.6f}\n")
        accelerator.print(f"指标已保存到: {os.path.join(result_dir, 'metrics.txt')}")


if __name__ == '__main__':
    main()
