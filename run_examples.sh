#!/bin/bash
# Time-LLM 运行示例脚本

echo "=========================================="
echo "Time-LLM 运行示例"
echo "=========================================="
echo ""
echo "请选择要运行的示例:"
echo ""
echo "1. 基础运行（不使用 SWT）"
echo "2. 使用 SWT 运行（推荐）"
echo "3. 快速测试（5 epochs）"
echo "4. 不同预测长度对比"
echo "5. 不同小波基对比"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "运行基础配置（不使用 SWT）..."
        python run_main.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --model_id baseline_96 \
            --model_comment baseline \
            --model TimeLLM \
            --data ETTh1 \
            --root_path ./dataset \
            --data_path ETTh1.csv \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 96 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model 16 \
            --d_ff 32 \
            --batch_size 32 \
            --learning_rate 0.0001 \
            --train_epochs 10 \
            --llm_model GPT2 \
            --llm_dim 768 \
            --llm_layers 6 \
            --patch_len 16 \
            --stride 8
        ;;
    2)
        echo ""
        echo "运行 SWT 配置（推荐）..."
        python run_main.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --model_id swt_96 \
            --model_comment swt_db4 \
            --model TimeLLM \
            --data ETTh1 \
            --root_path ./dataset \
            --data_path ETTh1.csv \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len 96 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model 16 \
            --d_ff 32 \
            --batch_size 32 \
            --learning_rate 0.0001 \
            --train_epochs 10 \
            --llm_model GPT2 \
            --llm_dim 768 \
            --llm_layers 6 \
            --patch_len 16 \
            --stride 8 \
            --use_swt \
            --swt_wavelet db4 \
            --swt_level 3 \
            --use_all_coeffs
        ;;
    3)
        echo ""
        echo "运行快速测试（5 epochs）..."
        python run_main.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --model_id quick_test \
            --model_comment quick \
            --model TimeLLM \
            --data ETTh1 \
            --root_path ./dataset \
            --data_path ETTh1.csv \
            --seq_len 96 \
            --pred_len 96 \
            --d_model 8 \
            --d_ff 16 \
            --batch_size 64 \
            --train_epochs 5 \
            --llm_model GPT2 \
            --llm_layers 2 \
            --use_swt
        ;;
    4)
        echo ""
        echo "运行不同预测长度对比..."
        for pred_len in 96 192 336 720; do
            echo "预测长度: $pred_len"
            python run_main.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --model_id pred${pred_len} \
                --model_comment swt_pred${pred_len} \
                --model TimeLLM \
                --data ETTh1 \
                --seq_len 96 \
                --pred_len $pred_len \
                --train_epochs 10 \
                --llm_model GPT2 \
                --use_swt
        done
        ;;
    5)
        echo ""
        echo "运行不同小波基对比..."
        for wavelet in db4 db6 sym4 coif2 haar; do
            echo "小波基: $wavelet"
            python run_main.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --model_id swt_${wavelet} \
                --model_comment wavelet_${wavelet} \
                --model TimeLLM \
                --data ETTh1 \
                --seq_len 96 \
                --pred_len 96 \
                --train_epochs 10 \
                --llm_model GPT2 \
                --use_swt \
                --swt_wavelet $wavelet \
                --swt_level 3
        done
        ;;
    *)
        echo "无效选项！"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "运行完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  训练日志: tail -f checkpoints/checkpoint_*/log.txt"
echo "  测试结果: cat results/*/metrics.txt"
