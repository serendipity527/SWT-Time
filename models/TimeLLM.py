from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
from layers.WaveletEmbed import WaveletPatchEmbedding  # 添加小波Patch Embedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class WaveletHead(nn.Module):
    """小波系数预测头 - 对称小波域输出
    
    将LLM隐状态投影到小波系数空间，然后通过ISWT重构回时域信号。
    与WaveletPatchEmbedding形成完整的"小波编码-LLM处理-小波解码"架构。
    
    架构优势：
    1. 对称设计：编码(SWT) ↔ 解码(ISWT)
    2. 频域约束：保证输出符合小波理论和频谱特性
    3. 多尺度预测：LLM分别学习不同频段（趋势、细节）
    4. 可解释性：可以分析各频段的预测质量
    
    Args:
        n_vars: 变量数量
        d_model: LLM隐状态维度
        patch_nums: patch数量
        pred_len: 预测长度
        level: 小波分解层数（需与编码器一致）
        wavelet: 小波基函数（需与编码器一致）
        head_dropout: dropout率
    
    Input:
        x: (B, N, d_model, patch_nums) - LLM处理后的隐状态
    
    Output:
        pred: (B, N, pred_len) - 预测的时域信号
    
    工作流程：
        LLM隐状态 (B, N, d_model, patch_nums)
          ↓ 为每个频段独立投影
        小波系数预测:
          - cA_pred: (B, N, pred_len) 低频趋势
          - cD3_pred: (B, N, pred_len) 高频细节
          - cD2_pred: (B, N, pred_len) 中频细节
          - cD1_pred: (B, N, pred_len) 低频细节
          ↓ Stack到频段维度
        (B, N, pred_len, Level+1)
          ↓ ISWT重构
        (B, N, pred_len)  ← 最终时域预测
    """
    
    def __init__(self, n_vars, d_model, patch_nums, pred_len, 
                 level=3, wavelet='db4', head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.level = level
        self.num_bands = level + 1  # 频段数量
        self.wavelet = wavelet
        
        # 为每个频段创建独立的投影层
        # 这允许LLM学习不同频段的不同特性（如趋势 vs 细节）
        self.band_projections = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=-2),  # (B, N, d_model, patch_nums) -> (B, N, d_model*patch_nums)
                nn.Linear(d_model * patch_nums, pred_len),  # -> (B, N, pred_len)
                nn.Dropout(head_dropout)
            )
            for _ in range(self.num_bands)
        ])
        
        # ISWT重构模块
        from layers.WaveletEmbed import ISWTReconstruction
        self.iswt = ISWTReconstruction(wavelet=wavelet, level=level)
        
        print(f"[WaveletHead] 创建小波输出头：{self.num_bands}个频段投影层 + ISWT重构")
        print(f"  - 小波类型: {wavelet}")
        print(f"  - 分解层数: {level}")
        print(f"  - 每个频段参数量: {d_model * patch_nums * pred_len:,}")
        print(f"  - 总参数量: {d_model * patch_nums * pred_len * self.num_bands:,}")
    
    def forward(self, x):
        """
        Args:
            x: (B, N, d_model, patch_nums) - LLM隐状态
        
        Returns:
            pred: (B, N, pred_len) - 预测的时域信号
        """
        B, N, d_model, patch_nums = x.shape
        
        # Step 1: 为每个频段生成预测的小波系数
        wavelet_coeffs = []
        for i, proj in enumerate(self.band_projections):
            # 每个投影层输出该频段的预测系数
            coeff = proj(x)  # (B, N, pred_len)
            wavelet_coeffs.append(coeff)
        
        # Step 2: Stack到频段维度
        # [(B, N, pred_len)] * num_bands -> (B, N, pred_len, num_bands)
        # 顺序: [cA_pred, cD_n_pred, cD_{n-1}_pred, ..., cD_1_pred]
        wavelet_coeffs = torch.stack(wavelet_coeffs, dim=-1)
        
        # Step 3: ISWT重构回时域
        # (B, N, pred_len, num_bands) -> (B, N, pred_len)
        pred = self.iswt(wavelet_coeffs)
        
        return pred


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # 使用WaveletPatchEmbedding替代原始PatchEmbedding
        # 可以通过configs.use_wavelet控制是否启用（默认启用）
        use_wavelet = getattr(configs, 'use_wavelet', True)
        
        if use_wavelet:
            # 小波Patch Embedding：先SWT分解，再Patching
            self.patch_embedding = WaveletPatchEmbedding(
                d_model=configs.d_model,
                patch_len=self.patch_len,
                stride=self.stride,
                wavelet=getattr(configs, 'wavelet', 'db4'),  # 默认db4小波
                level=getattr(configs, 'swt_level', 3),      # 默认3层分解
                dropout=configs.dropout
            )
            print(f"[TimeLLM] 使用 WaveletPatchEmbedding (小波={getattr(configs, 'wavelet', 'db4')}, 层数={getattr(configs, 'swt_level', 3)})")
        else:
            # 原始Patch Embedding
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)
            print("[TimeLLM] 使用 原始 PatchEmbedding")

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 可通过配置选择输出头类型
            use_wavelet_head = getattr(configs, 'use_wavelet_head', False)
            
            if use_wavelet_head:
                # 小波系数输出头：对称小波域架构
                # LLM隐状态 -> 小波系数预测 -> ISWT重构 -> 时域预测
                self.output_projection = WaveletHead(
                    n_vars=configs.enc_in,
                    d_model=self.d_ff,
                    patch_nums=self.patch_nums,
                    pred_len=self.pred_len,
                    level=getattr(configs, 'swt_level', 3),
                    wavelet=getattr(configs, 'wavelet', 'db4'),
                    head_dropout=configs.dropout
                )
                print("[TimeLLM] 使用 WaveletHead 输出层")
                print(f"  - 架构: LLM隐状态 → 小波系数({getattr(configs, 'swt_level', 3)+1}频段) → ISWT重构 → 时域预测")
            else:
                # 原始线性输出头：直接时域映射
                # LLM隐状态 -> 线性层 -> 时域预测
                self.output_projection = FlattenHead(
                    configs.enc_in, self.head_nf, self.pred_len,
                    head_dropout=configs.dropout
                )
                print("[TimeLLM] 使用 FlattenHead 输出层（直接时域映射）")
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
