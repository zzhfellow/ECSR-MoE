#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from src.layer import EnhancedLSTM, MultiHeadAttention, FusionGate, NewFusionGate, EmotionMoE
from src.tools import set_seed
import os

from transformers.file_utils import ModelOutput
from dataclasses import dataclass
import torch.nn.functional as F


class TextClassification(nn.Module):
    def __init__(self, cfg, tokenizer,logger=None):
        super(TextClassification, self).__init__()
        self.cfg = cfg
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
        self.logger = logger  # 存储 logger
        self.speaker_embedder = nn.Embedding(len(cfg.speaker_dict), bert_config.hidden_size)
        self.tokenizer = tokenizer

        num_classes = 7 if cfg['emo_cat'] == 'yes' else 2
        num = 2

        self.fusion = NewFusionGate(bert_config.hidden_size * num)

        drop_rate = 0.1

        self.video_linear = nn.Sequential(
            nn.Linear(cfg.video_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.audio_linear = nn.Sequential(
            nn.Linear(cfg.audio_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.emotion_linear = nn.Sequential(
            nn.Linear(bert_config.hidden_size * num, cfg.hid_size),
            nn.ReLU(),
            nn.Linear(cfg.hid_size, num_classes)
        )
        self.cause_linear = nn.Linear(bert_config.hidden_size * num, 2)

        self.dropout = nn.Dropout(cfg['dropout'])

        self.lstm = EnhancedLSTM('drop_connect', bert_config.hidden_size, bert_config.hidden_size, 1, ff_dropout=0.1,
                                 recurrent_dropout=0.1, bidirectional=True)

        att_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)

        self.speaker_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2,
                                                    att_head_size, att_head_size,
                                                    bert_config.attention_probs_dropout_prob)
        self.reply_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2,
                                                  att_head_size, att_head_size,
                                                  bert_config.attention_probs_dropout_prob)
        self.global_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size * 2,
                                                   att_head_size, att_head_size,
                                                   bert_config.attention_probs_dropout_prob)

        self.apply(self._init_esim_weights)

        self.bert = AutoModel.from_pretrained(cfg.bert_path)

        self.emotion_moe = EmotionMoE(
            num_emotions=num_classes,
            num_experts_per_emotion=cfg.num_experts_per_emotion,
            num_shared_experts=cfg.num_shared_experts,
            feature_dim=bert_config.hidden_size * num,
            hidden_dim=cfg.hidden_dim,
            top_k=cfg.top_k,
            top_m=cfg.top_m,
            neutral_weight=cfg.neutral_weight,
            target_dir=cfg.target_dir,
            gamma=cfg.gamma,
            # initial_gamma=cfg.initial_gamma,
            # total_epochs=cfg.epoch_size,  # 自动从 config 传入 8
            # use_epoch_decay=True,

        )

    def _init_esim_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_utt_mask(self, input, utterance_nums, pair_nums, pairs):
        mask = torch.arange(input.shape[1]).unsqueeze(0).to(input.device) < utterance_nums.unsqueeze(-1)
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu = torch.flip(torch.flip(torch.triu(torch.ones_like(mask[0])), [1]), [0])
        for i in range(len(utterance_nums)):
            mask[i] = mask[i] * triu

        batch_size, seq_len = input.shape[:2]

        gold = input.new_zeros((batch_size, seq_len, seq_len), dtype=torch.long)
        for i in range(len(input)):
            if pair_nums[i] == 0:
                continue
            gold[i, [w[0] for w in pairs[i, :pair_nums[i]]], [w[1] for w in pairs[i, :pair_nums[i]]]] = 1
        return mask, gold

    def merge_input(self, input, indices):
        max_utterance_num = max([len(w) for w in indices])
        res = input.new_zeros((len(indices), max_utterance_num, input.shape[-1]))
        for i in range(len(indices)):
            cur_id = indices[i][0][0]
            end_id = indices[i][-1][0]
            cur_lens = 0
            for j in range(cur_id, end_id + 1):
                start = input.new_tensor([w[1] for w in indices[i] if w[0] == j], dtype=torch.long)
                end = input.new_tensor([w[2] - 1 for w in indices[i] if w[0] == j], dtype=torch.long)
                start_rep = torch.gather(input[j], 0, start.unsqueeze(-1).expand(-1, input.shape[-1]))
                end_rep = torch.gather(input[j], 0, end.unsqueeze(-1).expand(-1, input.shape[-1]))
                end = input.new_tensor([w[2] for w in indices[i] if w[0] == j], dtype=torch.long)
                lens = start.shape[0]
                res[i, cur_lens:cur_lens + lens] = end_rep + input[j][0].unsqueeze(0)
                cur_lens += lens
        return res

    def get_emotion(self, logits, utterance_nums, emotion_labels, emo=True):
        mask = torch.arange(logits.shape[1]).unsqueeze(0).to(logits.device) < utterance_nums.unsqueeze(-1)
        mask = mask.to(logits.device)
        activate_loss = mask.view(-1) == 1
        activate_logits = logits.view(-1, logits.shape[-1])[activate_loss]
        activate_gold = emotion_labels.view(-1)[activate_loss]
        if self.cfg.emo_cat == 'yes' and emo:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.5 , 1.5, 1.5, 1.5, 1.5]).to(logits.device))
            # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] + [1.5] * 6).to(logits.device))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(logits.device))
        loss = criterion(activate_logits, activate_gold.long())
        return loss

    def set_mask(self, emotion_logitis, cause_logits, utterance_nums):
        valid_mask = torch.arange(emotion_logitis.shape[1]).unsqueeze(0).to(
            cause_logits.device) < utterance_nums.unsqueeze(-1)
        valid_mask = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)
        emotion_mask = emotion_logitis.argmax(-1) == 1
        cause_mask = cause_logits.argmax(-1) == 1
        joint_mask = emotion_mask.unsqueeze(-1) | cause_mask.unsqueeze(1)
        mask = valid_mask * joint_mask
        return mask

    def align_features(self, feat0, feat1, utterance_nums):
        mask = torch.arange(feat0.shape[1]).unsqueeze(0).to(feat0.device) < utterance_nums.unsqueeze(-1)
        criterion = nn.MSELoss()
        activate_loss = mask.view(-1) == 1
        feat0 = feat0.view(-1, feat0.shape[-1])[activate_loss]
        feat1 = feat1.view(-1, feat1.shape[-1])[activate_loss]
        loss = criterion(feat0, feat1)
        return loss

    def build_hdict(self, input, text, audio, video, utterance_nums):
        res = []
        for i in range(len(utterance_nums)):
            instance = input[i, :utterance_nums[i]]
            res.append(instance)
        res = torch.cat(res, dim=0)
        mm_res = []
        for i in range(len(utterance_nums)):
            instance0 = text[i, :utterance_nums[i]]
            instance1 = audio[i, :utterance_nums[i]]
            instance2 = video[i, :utterance_nums[i]]
            mm_res.extend([instance0, instance1, instance2])
        mm_res = torch.cat(mm_res, dim=0)
        h_dict = {
            'all': res,
            'sub': mm_res
        }
        return h_dict

    def split_sequence(self, input, utterance_nums):
        res = input.new_zeros((len(utterance_nums), max(utterance_nums), input.shape[-1]))
        start = 0
        for i in range(len(utterance_nums)):
            res[i, :utterance_nums[i]] = input[start:start + utterance_nums[i]]
            start += utterance_nums[i]
        return res

    def build_attention(self, sequence_outputs, gmasks=None, smasks=None, rmasks=None):
        rep = self.reply_attention(sequence_outputs, sequence_outputs, sequence_outputs, rmasks)[0]
        thr = self.global_attention(sequence_outputs, sequence_outputs, sequence_outputs, gmasks)[0]
        sp = self.speaker_attention(sequence_outputs, sequence_outputs, sequence_outputs, smasks)[0]
        r = torch.stack((rep, thr, sp), 0)
        r = torch.max(r, 0)[0]
        length = sequence_outputs.shape[1] // 4
        return r[:, : length]

    def forward(self, **kwargs):
        input_ids, input_masks, utterance_nums = [kwargs[w] for w in 'input_ids input_masks utterance_nums'.split()]
        pairs, pair_nums, labels, indices = [kwargs[w] for w in 'pairs pair_nums labels indices'.split()]
        cause_labels, speaker_ids = [kwargs[w] for w in ['cause_labels', 'speaker_ids']]
        audio_features, video_features = [kwargs[w] for w in ['audio_features', 'video_features']]
        gmasks, smasks, rmasks = [kwargs[w] for w in ['gmasks', 'smasks', 'rmasks']]

        input = self.bert(input_ids, attention_mask=input_masks)[0]
        speaker_emb = self.speaker_embedder(speaker_ids)

        text = self.merge_input(input, indices)
        audio = self.audio_linear(audio_features)
        video = self.video_linear(video_features)

        input = text + speaker_emb + audio + video

        input = self.lstm(input, None, utterance_nums.cpu())

        tt = torch.cat((text, text), dim=-1)
        aa = torch.cat((audio, audio), dim=-1)
        vv = torch.cat((video, video), dim=-1)

        sequence_input = torch.cat((input, tt, aa, vv), 1)


        output = self.build_attention(sequence_input, gmasks, smasks, rmasks)
        input = self.fusion(input, output)

        emotion_logits = self.emotion_linear(input)
        emo_loss = self.get_emotion(emotion_logits, utterance_nums, labels, emo=True)

        cause_logits = self.cause_linear(input)
        cause_loss = self.get_emotion(cause_logits, utterance_nums, cause_labels, emo=False)

        ecp_logits, load_balance_loss = self.emotion_moe(
            input,
            emotion_logits,
            utterance_nums,
            gold_labels=labels  # 新增：传入真实情感标签
        )

        ecp_mask, gold_matrix = self.get_utt_mask(input, utterance_nums, pair_nums, pairs)
        activate_loss = ecp_mask.view(-1) == 1
        activate_logits = ecp_logits.view(-1, 2)[activate_loss]
        activate_gold = gold_matrix.view(-1)[activate_loss]
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, self.cfg['loss_weight']]).to(input.device))
        ecp_loss = criterion(activate_logits, activate_gold.long())
        if torch.isnan(ecp_loss):
            ecp_loss = 0

        loss = emo_loss + cause_loss + ecp_loss + load_balance_loss

        return loss, (ecp_logits, emotion_logits, cause_logits, ecp_mask)