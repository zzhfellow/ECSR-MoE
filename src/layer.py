import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import warnings
from itertools import accumulate
import os
from loguru import logger  # 导入 loguru

class EnhancedLSTM(torch.nn.Module):
    """
    A wrapper for different recurrent dropout implementations, which
    pytorch currently doesn't support nativly.

    Uses multilayer, bidirectional lstms with dropout between layers
    and time steps in a variational manner.

    "allen" reimplements a lstm with hidden to hidden dropout, thus disabling
    CUDNN. Can only be used in bidirectional mode.
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`

    "drop_connect" uses default implemetation, but monkey patches the hidden to hidden
    weight matrices instead.
    `Regularizing and Optimizing LSTM Language Models
        <https://arxiv.org/abs/1708.02182>`

    "native" ignores dropout and uses the default implementation.
    """

    def __init__(self,
                 lstm_type,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.lstm_type = lstm_type

        if lstm_type == "drop_connect":
            self.provider = WeightDropLSTM(
                input_size,
                hidden_size,
                num_layers,
                ff_dropout,
                recurrent_dropout,
                bidirectional=bidirectional)
        elif lstm_type == "native":
            self.provider = torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True)
        else:
            raise Exception(lstm_type + " is an invalid lstm type")

    # Expects unpacked inputs in format (batch, seq, features)
    def forward(self, inputs, hidden, lengths):
        seq_len = inputs.shape[1]
        if self.lstm_type in ["allen", "native"]:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True)

            output, _ = self.provider(packed, hidden)

            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

            return output
        elif self.lstm_type == "drop_connect":
            return self.provider(inputs, lengths, seq_len)


class WeightDropLSTM(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.locked_dropout = LockedDropout()
        self.lstms = [
            torch.nn.LSTM(
                input_size
                if l == 0 else hidden_size * (1 + int(bidirectional)),
                hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True) for l in range(num_layers)
        ]
        if recurrent_dropout:
            self.lstms = [
                WeightDrop(lstm, ['weight_hh_l0'], dropout=recurrent_dropout)
                for lstm in self.lstms
            ]

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.ff_dropout = ff_dropout
        self.num_layers = num_layers

    def forward(self, input, lengths, seq_len):
        """Expects input in format (batch, seq, features)"""
        output = input
        for lstm in self.lstms:
            output = self.locked_dropout(
                output, batch_first=True, p=self.ff_dropout)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=True, enforce_sorted=False)
            output, _ = lstm(packed, None)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len)

        return output

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_first=False, p=0.5):
        if not self.training or not p:
            return x
        mask_shape = (x.size(0), 1, x.size(2)) if batch_first else (1,
                                                                    x.size(1),
                                                                    x.size(2))

        mask = x.data.new(*mask_shape).bernoulli_(1 - p).div_(1 - p)
        return mask * x



class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        if hasattr(module, "bidirectional") and module.bidirectional:
            self.weights.extend(
                [weight + "_reverse" for weight in self.weights])

        self.dropout = dropout
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')

            w = None
            mask = torch.ones(1, raw_w.size(1))
            if raw_w.is_cuda: mask = mask.to(raw_w.device)
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout, training=self.training)
            w = mask.expand_as(raw_w) * raw_w
            self.module._parameters[name_w] = w

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # Ignore lack of flattening warning
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class FusionGate(nn.Module):
    def __init__(self, hid_size):
        super(FusionGate, self).__init__()
        self.fuse_weight = nn.Parameter(torch.Tensor(hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fuse_weight)

    def forward(self, a, b):
        # Compute fusion coefficients
        fusion_coef = torch.sigmoid(self.fuse_weight)
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor


class NewFusionGate(nn.Module):
    def __init__(self, hid_size):
        super(NewFusionGate, self).__init__()
        self.fuse = nn.Linear(hid_size * 2, hid_size)

    def forward(self, a, b):
        # Concatenate a and b along the last dimension
        concat_ab = torch.cat([a, b], dim=-1)
        # Apply the linear layer
        fusion_coef = torch.sigmoid(self.fuse(concat_ab))
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor







class EmotionAttentionWeight(nn.Module):
    def __init__(self, feature_dim, num_emotions, hidden_dim):
        """
        情感注意力权重生成模块，结合子句对语义特征和情感概率，生成动态权重。

        参数:
            feature_dim (int): 子句对特征的维度（3072）。
            num_emotions (int): 情感类别数量（7）。
            hidden_dim (int): 注意力网络的隐藏层维度。
        """
        super(EmotionAttentionWeight, self).__init__()
        self.feature_dim = feature_dim
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim

        # 投影层：将子句对特征映射到隐藏维度
        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, self.hidden_dim)

        )

        # 情感概率投影层：将情感概率映射到隐藏维度
        self.emotion_projection = nn.Sequential(
            nn.Linear(num_emotions, 128),
            nn.ReLU(),

            nn.Linear(128, hidden_dim)
        )
        # 注意力评分层
        self.attention_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),

            nn.Linear(128, num_emotions)
        )



        # LayerNorm 和 Dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, pair_candidates, emotion_probs, utterance_nums):
        """
        前向传播，生成每对子句在七个情感类别上的注意力权重。

        输入:
            pair_candidates: 子句对特征，[batch_size, num_clauses, num_clauses, feature_dim]
            emotion_probs: 情感概率，[batch_size, num_clauses, num_emotions]
            utterance_nums: 每个对话的子句数量，[batch_size]

        输出:
            attention_weights: 注意力权重，[batch_size, num_clauses, num_clauses, num_emotions]
        """
        batch_size, num_clauses, _, feature_dim = pair_candidates.shape
        device = pair_candidates.device

        # 生成子句有效掩码
        clause_mask = torch.arange(num_clauses, device=device).unsqueeze(0) < utterance_nums.unsqueeze(
            -1)  # [batch_size, num_clauses]
        pair_mask = clause_mask.unsqueeze(-1) & clause_mask.unsqueeze(-2)  # [batch_size, num_clauses, num_clauses]

        # 投影子句对特征
        pair_features = self.feature_projection(pair_candidates)  # [batch_size, num_clauses, num_clauses, hidden_dim]
        pair_features = self.layer_norm(self.dropout(F.relu(pair_features)))

        # 准备情感概率：对触发子句（emo_expanded）的概率取平均
        emo_probs_expanded = emotion_probs.unsqueeze(2).expand(-1, -1, num_clauses,
                                                               -1)  # [batch_size, num_clauses, num_clauses, num_emotions]
        emotion_features = self.emotion_projection(
            emo_probs_expanded)  # [batch_size, num_clauses, num_clauses, hidden_dim]
        emotion_features = self.layer_norm(self.dropout(F.relu(emotion_features)))

        # 注意力机制：结合子句对特征和情感特征
        combined_features = pair_features + emotion_features  # [batch_size, num_clauses, num_clauses, hidden_dim]
        attention_scores = self.attention_scorer(
            combined_features)  # [batch_size, num_clauses, num_clauses, num_emotions]

        # 应用 softmax 得到归一化权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_clauses, num_clauses, num_emotions]

        # 屏蔽无效子句对
        attention_weights = attention_weights * pair_mask.unsqueeze(-1).float()

        return attention_weights

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionMoE(nn.Module):
    def __init__(self, num_emotions, num_experts_per_emotion, num_shared_experts, feature_dim, hidden_dim, top_k, top_m, neutral_weight, target_dir=None, gamma=0.1):
        super().__init__()
        self.num_emotions = num_emotions
        self.num_experts_per_emotion = num_experts_per_emotion
        self.num_shared_experts = num_shared_experts
        self.num_active_emotions = num_emotions - 1
        self.top_k = top_k
        self.top_m = top_m
        self.input_dim = feature_dim * 2
        self.hidden_dim = hidden_dim
        self.neutral_weight = neutral_weight
        self.target_dir = target_dir
        self.gamma = gamma  # Bias update speed
        # self.initial_gamma = initial_gamma
        # self.final_gamma = final_gamma
        # self.total_epochs = total_epochs
        # self.use_epoch_decay = use_epoch_decay
        # self.current_epoch = 0  # 当前 epoch 计数器，由外部训练脚本更新
        self.lambda_sequence = 0.001  # Load balance loss weight
        self.emotion_names = {
            0: 'neutral',
            1: 'anger',
            2: 'disgust',
            3: 'fear',
            4: 'joy',
            5: 'sadness',
            6: 'surprise'
        }
        # 新增：统计变量（只统计非中性情感）
        # pair_counts: [6] 对应 anger ~ surprise
        # activation_sums: [6, 6*4] = [6, 24]，每行对应一个真实情感的所有专家激活和
        self.register_buffer('pair_counts', torch.zeros(self.num_active_emotions, dtype=torch.long))
        self.register_buffer('activation_sums', torch.zeros(self.num_active_emotions,
                                                            self.num_active_emotions * num_experts_per_emotion,
                                                            dtype=torch.float))
        self.stats_enabled = False

        # 应用到专家网络
        self.experts_per_emotion = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_dim, 2048),
                    nn.ReLU(),


                    nn.Linear(2048, 1024),
                    nn.ReLU(),

                    nn.Linear(1024, 512),
                    nn.ReLU(),


                    nn.Linear(512, 2)
                ) for _ in range(num_experts_per_emotion)
            ]) for _ in range(num_emotions)
        ])

        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 2048),
                nn.ReLU(),


                nn.Linear(2048, 1024),
                nn.ReLU(),


                nn.Linear(1024, 512),
                nn.ReLU(),

                nn.Linear(512, 2)
            ) for _ in range(num_shared_experts)
        ])

        self.gating_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, 2048),
                nn.ReLU(),

                nn.Linear(2048, 1024),
                nn.ReLU(),

                nn.Linear(1024, 512),
                nn.ReLU(),


                nn.Linear(512, num_experts_per_emotion)
            ) for _ in range(self.num_active_emotions)
        ])

        self.attention_weight_generator = EmotionAttentionWeight(
            feature_dim=feature_dim * 2,
            num_emotions=num_emotions,
            hidden_dim=hidden_dim
        )

        # Projection layer for residual connection
        self.residual_projection = nn.Sequential(
                nn.Linear(self.input_dim, 2048),
                nn.ReLU(),

                nn.Linear(2048, 1024),
                nn.ReLU(),

                nn.Linear(1024, 512),
                nn.ReLU(),


                nn.Linear(512, 2)
        )

        # Use register_buffer to avoid gradient computation
        self.register_buffer('expert_bias', torch.zeros(num_emotions, num_experts_per_emotion))

    def enable_stats(self):
        self.stats_enabled = True
        self.pair_counts.zero_()
        self.activation_sums.zero_()

    def disable_stats(self):
        self.stats_enabled = False

    def print_activation_stats(self):
        """打印非中性情感样本的专家激活强度统计"""
        if not self.stats_enabled or self.pair_counts.sum() == 0:
            print("【警告】无非中性情感样本统计数据")
            return

        print("\n=== 非中性情感样本专家激活强度统计 ===")
        for true_emo_idx in range(self.num_active_emotions):  # 0~5 → anger~surprise
            true_emo_id = true_emo_idx + 1
            emo_name = self.emotion_names[true_emo_id]
            total_pairs = self.pair_counts[true_emo_idx].item()
            if total_pairs == 0:
                continue

            line = f"【{emo_name} 样本】 (有效pair数: {total_pairs}) "
            for routed_emo_idx in range(self.num_active_emotions):  # 路由到的专家情感
                routed_emo_name = self.emotion_names[routed_emo_idx + 1]
                for exp_idx in range(self.num_experts_per_emotion):
                    global_exp_idx = routed_emo_idx * self.num_experts_per_emotion + exp_idx
                    avg_act = self.activation_sums[true_emo_idx, global_exp_idx].item() / total_pairs
                    line += f"{routed_emo_name}_exp{exp_idx}: {avg_act:.4f} "
            print(line)
        print("============================================\n")

    def update_bias(self, gates_list, emo_idx, device, top_k_indices):
        if not gates_list:
            return 0.0
        for gates in gates_list:
            gate_probs = gates.mean(dim=0)  # Average gating probability per expert for this emotion
            avg_load = gate_probs.sum() / self.num_experts_per_emotion
            load_diff = gate_probs - avg_load  # Positive: overloaded, Negative: underloaded
            with torch.no_grad():
                for idx in range(self.num_experts_per_emotion):
                    if load_diff[idx] > 0:  # Overloaded
                        self.expert_bias[emo_idx + 1, idx] -= self.gamma
                    elif load_diff[idx] < 0:  # Underloaded
                        self.expert_bias[emo_idx + 1, idx] += self.gamma
    def get_current_gamma(self):
        """
        计算当前 epoch 对应的 gamma 值。
        若启用 use_epoch_decay，则从 initial_gamma 线性衰减至 final_gamma。
        否则返回固定 initial_gamma。
        """
        if not self.use_epoch_decay or self.total_epochs <= 0:
            return self.initial_gamma

        progress = min(self.current_epoch / self.total_epochs, 1.0)
        current_gamma = self.initial_gamma + (self.final_gamma - self.initial_gamma) * progress
        return max(current_gamma, self.final_gamma)

    # def update_bias(self, gates_list, emo_idx, device, top_k_indices=None):
    #     """
    #     动态负载均衡偏置更新（无辅助损失）。
    #     使用当前 epoch 计算得到的动态 gamma。
    #     """
    #     if not gates_list:
    #         return 0.0
    #
    #     # 合并当前批次该情感的所有 gating 概率
    #     all_gates = torch.cat(gates_list, dim=0)  # [total_pairs_this_emotion, num_experts_per_emotion]
    #     gate_probs = all_gates.mean(dim=0)  # 每个专家的平均负载 L_q
    #
    #     ideal_load = 1.0 / self.num_experts_per_emotion
    #     load_diff = ideal_load - gate_probs  # 正值：负载过低 → 增大 bias
    #
    #     current_gamma = self.get_current_gamma()  # ← 关键：动态 gamma
    #
    #     with torch.no_grad():
    #         self.expert_bias[emo_idx + 1] += current_gamma * load_diff
    #
    #     return 0.0

    def compute_sequence_loss(self, gates_list):
        if not gates_list:
            return 0.0
        sequence_loss = 0.0
        for gates in gates_list:
            gate_probs = gates.mean(dim=0)
            entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8))
            sequence_loss += entropy
        return sequence_loss / len(gates_list)

    def forward(self, input, emotion_probs, utterance_nums, gold_labels=None):
        """
        input: [batch_size, num_clauses, feature_dim]
        emotion_probs: [batch_size, num_clauses, num_emotions]  (predicted emotion probabilities)
        utterance_nums: tensor or list, length = batch_size
        gold_labels: [batch_size, num_clauses] LongTensor, real emotion labels (0=neutral, 1~6=non-neutral)
        """
        device = input.device
        batch_size, num_clauses = input.shape[:2]
        gates_list = []

        # 在 eval 模式且传入 gold_labels 时自动开启统计（并在第一次调用时清零）
        if not self.training and gold_labels is not None:
            if not self.stats_enabled:
                self.enable_stats()  # 清零 pair_counts 和 activation_sums

        # Construct pair candidates
        emo_expanded = input.unsqueeze(2).expand(-1, -1, num_clauses, -1)
        all_expanded = input.unsqueeze(1).expand(-1, num_clauses, -1, -1)
        pair_candidates = torch.cat([emo_expanded, all_expanded], dim=-1)
        emotion_probs_expanded = emotion_probs.unsqueeze(2).expand(-1, -1, num_clauses, -1)

        # Generate attention weights
        attention_weights = self.attention_weight_generator(pair_candidates, emotion_probs, utterance_nums)

        # Initialize output tensor
        pair_logits = torch.zeros(batch_size, num_clauses, num_clauses, 2, device=device)
        pair_candidates_flat = pair_candidates.view(batch_size * num_clauses * num_clauses, self.input_dim)
        attention_weights_flat = attention_weights.reshape(batch_size * num_clauses * num_clauses, self.num_emotions)
        emotion_probs_flat = emotion_probs_expanded.reshape(batch_size * num_clauses * num_clauses, self.num_emotions)

        # Create clause mask
        clause_mask = torch.arange(num_clauses, device=device).unsqueeze(0) < utterance_nums.unsqueeze(-1)
        clause_mask_flat = clause_mask.unsqueeze(-1).expand(-1, -1, num_clauses).reshape(
            batch_size * num_clauses * num_clauses)
        assert clause_mask_flat.sum() > 0, "clause_mask_flat is all zeros!"

        # Filter active pairs
        active_indices = clause_mask_flat.nonzero(as_tuple=True)[0]
        pair_candidates_active = pair_candidates_flat[active_indices]
        emotion_probs_active = emotion_probs_flat[active_indices]
        attention_weights_active = attention_weights_flat[active_indices]

        # === 修正：提取每个 active pair 的真实情感标签（gold label）===
        active_gold_emos = None
        if not self.training and gold_labels is not None:
            # gold_labels: [B, num_clauses]
            # pair 的情感 utterance 是行索引（emo_expanded 来自 input.unsqueeze(2)）
            gold_emo_for_pairs = gold_labels.unsqueeze(2).expand(-1, -1,
                                                                 num_clauses)  # [B, num_clauses_emo, num_clauses_cause]
            gold_emo_flat = gold_emo_for_pairs.reshape(batch_size * num_clauses * num_clauses)
            active_gold_emos = gold_emo_flat[active_indices]  # [num_active_pairs]

        # Compute shared experts' outputs (once for all pairs, no gating)
        shared_outputs = torch.zeros(len(active_indices), self.num_shared_experts, 2, device=device)
        for shared_idx in range(self.num_shared_experts):
            expert = self.shared_experts[shared_idx]
            shared_outputs[:, shared_idx] = expert(pair_candidates_active)
        shared_outputs_sum = shared_outputs.sum(dim=1)  # [num_active_pairs, 2]

        # Select top-M emotions
        probs_non_neutral = emotion_probs_active[:, 1:]
        _, top_m_indices = probs_non_neutral.topk(self.top_m, dim=-1)

        pair_lists = [[] for _ in range(self.num_active_emotions)]
        for p in range(len(active_indices)):
            for sel_emo_idx in top_m_indices[p]:
                pair_lists[sel_emo_idx.item()].append(p)

        # Initialize expert outputs
        expert_outputs_all = torch.zeros(len(active_indices), self.num_emotions, 2, device=device)

        # Process routed experts per emotion
        for emo_idx in range(self.num_active_emotions):  # emo_idx: 0~5 → anger~surprise
            sub_pair_ids = pair_lists[emo_idx]
            if not sub_pair_ids:
                continue
            sub_indices = torch.tensor(sub_pair_ids, device=device)
            sub_candidates = pair_candidates_active[sub_indices]

            # Gating
            gating_network = self.gating_networks[emo_idx]
            gates = gating_network(sub_candidates)  # [num_sub_pairs, num_experts_per_emotion]
            gates_for_routing = gates + self.expert_bias[emo_idx + 1]
            gates_soft = F.softmax(gates, dim=-1)
            gates_for_routing_soft = F.softmax(gates_for_routing, dim=-1)
            gates_list.append(gates_soft)

            # === 统计专家激活强度（仅非中性真实情感样本）===
            if not self.training and gold_labels is not None and active_gold_emos is not None:
                sub_gold = active_gold_emos[sub_indices]  # [num_sub_pairs]
                valid_mask = (sub_gold >= 1) & (sub_gold <= 6)  # 只统计非中性
                if valid_mask.any():
                    valid_gold = sub_gold[valid_mask] - 1  # 映射到 0~5 索引（anger~surprise）
                    valid_gates = gates_soft[valid_mask]  # [valid_num, num_experts]

                    for local_exp_idx in range(self.num_experts_per_emotion):
                        global_exp_idx = emo_idx * self.num_experts_per_emotion + local_exp_idx
                        acts = valid_gates[:, local_exp_idx]  # [valid_num]
                        for i, true_idx in enumerate(valid_gold):
                            self.activation_sums[true_idx, global_exp_idx] += acts[i].item()

            # Select top-K routed experts
            indep_top_k_gates, indep_top_k_indices = gates_for_routing_soft.topk(self.top_k, dim=-1)
            top_k_gates = indep_top_k_gates / (indep_top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)

            # Compute routed experts' outputs
            expert_outputs = torch.zeros(len(sub_indices), self.top_k, 2, device=device)
            for k in range(self.top_k):
                k_indices = indep_top_k_indices[:, k]
                k_gates = top_k_gates[:, k].unsqueeze(-1)
                for indep_idx in range(self.num_experts_per_emotion):
                    idx_mask = (k_indices == indep_idx)
                    if idx_mask.sum() == 0:
                        continue
                    expert = self.experts_per_emotion[emo_idx][indep_idx]
                    expert_input = sub_candidates[idx_mask]
                    k_gates_subset = k_gates[idx_mask]
                    expert_output = expert(expert_input)
                    indices = idx_mask.nonzero(as_tuple=True)[0]
                    if indices.numel() > 0:
                        expert_outputs[indices, k] = expert_output * k_gates_subset

            expert_outputs_all[sub_indices, emo_idx + 1] = expert_outputs.sum(dim=1)
            self.update_bias([gates_soft], emo_idx, device, indep_top_k_indices)

        # Combine shared and routed expert outputs
        expert_outputs_all[:, 0] = shared_outputs_sum  # neutral slot

        # Apply attention weights
        attention_weights_active = attention_weights_active.unsqueeze(-1)  # [num_active_pairs, num_emotions, 1]
        pair_predictions = (expert_outputs_all * attention_weights_active).sum(dim=1)  # [num_active_pairs, 2]

        # Residual connection
        residual = self.residual_projection(pair_candidates_active)
        pair_predictions = pair_predictions + residual

        # Assign to full tensor
        pair_logits_flat = pair_logits.view(batch_size * num_clauses * num_clauses, 2)
        pair_logits_flat[active_indices] = pair_predictions
        pair_logits = pair_logits_flat.view(batch_size, num_clauses, num_clauses, 2)

        # === 统计非中性样本的 pair 数量 ===
        if not self.training and gold_labels is not None and active_gold_emos is not None:
            non_neutral = (active_gold_emos >= 1) & (active_gold_emos <= 6)
            if non_neutral.any():
                counts = torch.bincount(active_gold_emos[non_neutral] - 1, minlength=self.num_active_emotions)
                self.pair_counts.add_(counts)  # in-place 累加

        return pair_logits, 0.0

