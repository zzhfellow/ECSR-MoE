#!/usr/bin/env python

"""
Name: trainer.py
"""

import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
import copy

class MyTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.result_file = os.path.join(config.target_dir, 'result.txt')
        if not os.path.exists(self.result_file):
            with open(self.result_file, 'w', encoding='utf-8') as f:
                f.write('训练结果记录\n\n')

        self.scores = []
        self.lines = []
        self.test_lines = []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        best_test_f1 = 0.0
        best_state_dict = None

        epoch_bar = tqdm(
            range(self.config.epoch_size),
            desc="Overall Training Progress",
            unit="epoch",
            position=0,
            leave=True
        )

        for epoch in epoch_bar:
            self.model.global_epoch = epoch
            self.global_epoch = epoch

            # 更新进度条描述
            epoch_bar.set_description(f"Epoch {epoch + 1}/{self.config.epoch_size}")

            # 训练（内部无任何进度条）
            self.train_step()
            self.re_init()

            # 验证
            valid_score, (valid_res, _) = self.evaluate_step()
            print(f"\n=== Validation ===\n{valid_res}")

            # 测试
            self.re_init()
            test_score, (test_res, test_metrics) = self.evaluate_step(self.test_loader)
            test_f1 = test_metrics['default']
            print(f"\n=== Test ===\n{test_res}")


            self.re_init()
            self.add_instance(valid_score, valid_res, test_score, test_res)

            # 更新 best
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_iter = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                best_res = test_res
                epoch_bar.set_postfix({"Best F1": f"{best_test_f1:.4f}", "Current": f"{test_f1:.4f}"})

            import gc
            gc.collect()  # 清理 Python 垃圾对象
            torch.cuda.empty_cache()  # 清理 PyTorch CUDA 缓存

            # 早停
            if epoch - best_iter > self.config.patience:
                print(f"\nEarly stopping triggered after {self.config.patience} epochs without improvement.")
                break

        epoch_bar.close()
        print(f"\nTraining completed. Best test F1: {best_test_f1:.4f} at epoch {best_iter + 1}")

        return best_test_f1, best_state_dict, best_res, best_iter

    def train_step(self):
        self.model.train()
        losses = []
        total_batches = len(self.train_loader)

        for batch_idx, data in enumerate(self.train_loader, 1):
            loss, _ = self.model(**data)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.optimizer.step()
            if self.config.scheduler is not None:
                self.config.scheduler.step()
            self.model.zero_grad()

            # 可选：每 100 个 batch 打印一次平均 loss，避免完全无声
            if batch_idx % 100 == 0 or batch_idx == total_batches:
                avg_loss = np.mean(losses[-100:])  # 最近 100 个的平均
                print(f"    → Batch {batch_idx}/{total_batches} | Recent avg loss: {avg_loss:.4f}")

        # epoch 结束时打印最终平均 loss
        final_avg_loss = np.mean(losses)
        print(f"  Train Epoch {self.global_epoch} finished | Avg loss: {final_avg_loss:.4f}")

    def evaluate_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader

        for data in dataLoader:
            with torch.no_grad():
                loss, output = self.model(**data)
                self.add_output(data, output)

        score, res = self.report_score()
        gold_count = len(set(self.golds['ecp']))  # 使用golds['ecp']长度


        return score, res

    def final_evaluate(self, epoch=0):
        self.model.eval()
        score, res = self.evaluate_step(self.test_loader)
        print(res[0])
        return score, res

    def re_init(self):
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        self.keys = ['default']

    def add_instance(self, valid_score, valid_res, test_score, test_res):
        self.scores.append(valid_score)
        self.lines.append((valid_res, {'default': valid_score}))
        self.test_lines.append((test_res, {'default': test_score}))

    def get_best(self):
        best_id = np.argmax(self.scores)
        return self.lines[best_id]

    def add_output(self, data, output):
        ecp_predictions, emo_predictions, cause_predictions, masks = output
        predictions = ecp_predictions.argmax(-1).cpu().numpy()
        emo_pred = emo_predictions.argmax(-1).cpu().numpy()
        cause_pred = cause_predictions.argmax(-1).cpu().numpy()
        masks = masks.cpu().numpy()

        for i in range(len(emo_pred)):
            mask = masks[i]
            doc_id = data['doc_ids'][i]
            utt_nums = data['utterance_nums'][i]

            emo_pred_ = emo_pred[i, :utt_nums].tolist()
            emo_gold_ = data['labels'][i, :utt_nums].tolist()
            self.preds['emo'] += emo_pred_
            self.golds['emo'] += emo_gold_

            cause_pred_ = cause_pred[i, :utt_nums].tolist()
            cause_gold_ = data['cause_labels'][i, :utt_nums].tolist()
            self.preds['cause'] += cause_pred_
            self.golds['cause'] += cause_gold_

            pair_num = data['pair_nums'][i]
            prediction = predictions[i] * mask
            pred_pairs = np.where(prediction == 1)
            pred_pairs = [(w, z) for w, z in zip(pred_pairs[0], pred_pairs[1])
                          if emo_pred_[w] == 1 or cause_pred_[z] == 1]
            pred_pairs = [(doc_id, w, z) for w, z in pred_pairs if w >= z]

            self.preds['ecp'] += pred_pairs
            self.golds['ecp'] += [(doc_id, *w) for w in data['pairs'][i][:pair_num].tolist()]

            for _, w, z in pred_pairs:
                emo_idx = emo_pred_[w]
                self.preds[f'ecp_{emo_idx}'].append((doc_id, w, z))
            for w, z in data['pairs'][i][:pair_num].tolist():
                emo_idx = emo_gold_[w]
                self.golds[f'ecp_{emo_idx}'].append((doc_id, w, z))

    def report_score(self):
        index_to_emotion = {v: k for k, v in self.config.label_dict.items()}

        tp = len(set(self.preds['ecp']) & set(self.golds['ecp']))
        fp = len(set(self.preds['ecp']) - set(self.golds['ecp']))
        fn = len(set(self.golds['ecp']) - set(self.preds['ecp']))

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0

        ecp_metrics = {}
        for emo_idx in range(7):
            tp_emo = len(set(self.preds[f'ecp_{emo_idx}']) & set(self.golds[f'ecp_{emo_idx}']))
            fp_emo = len(set(self.preds[f'ecp_{emo_idx}']) - set(self.golds[f'ecp_{emo_idx}']))
            fn_emo = len(set(self.golds[f'ecp_{emo_idx}']) - set(self.preds[f'ecp_{emo_idx}']))

            p_emo = tp_emo / (tp_emo + fp_emo) if tp_emo + fp_emo > 0 else 0
            r_emo = tp_emo / (tp_emo + fn_emo) if tp_emo + fn_emo > 0 else 0
            f_emo = 2 * p_emo * r_emo / (p_emo + r_emo) if p_emo + r_emo > 0 else 0

            ecp_metrics[emo_idx] = {
                'p': p_emo,
                'r': r_emo,
                'f': f_emo,
                'tp': tp_emo,
                'pred': tp_emo + fp_emo,
                'gold': tp_emo + fn_emo
            }

        # 计算 Avg.6（非中性情感 1-6 的加权平均 F1）
        f1_scores_6 = [ecp_metrics[i]['f'] for i in range(1, 7)]
        weights_6 = [ecp_metrics[i]['gold'] for i in range(1, 7)]
        total_weight_6 = sum(weights_6)
        avg_6_f1 = sum(f * w for f, w in zip(f1_scores_6, weights_6)) / total_weight_6 if total_weight_6 > 0 else 0

        # 计算 Avg.4（anger, joy, sadness, surprise）
        f1_scores_4 = [ecp_metrics[i]['f'] for i in [1, 4, 5, 6]]  # anger, joy, sadness, surprise
        weights_4 = [ecp_metrics[i]['gold'] for i in [1, 4, 5, 6]]
        total_weight_4 = sum(weights_4)
        avg_4_f1 = sum(f * w for f, w in zip(f1_scores_4, weights_4)) / total_weight_4 if total_weight_4 > 0 else 0

        gold_emo = [0 if w == 0 else 1 for w in self.golds['emo']]
        pred_emo = [0 if w == 0 else 1 for w in self.preds['emo']]
        emo = precision_recall_fscore_support(gold_emo, pred_emo, average='binary')
        cause = precision_recall_fscore_support(self.golds['cause'], self.preds['cause'], average='binary')

        res = (f"总体 ECP 指标:\n"
               f"Pair Pre. {p * 100:.4f}\t Rec. {r * 100:.4f}\tF1 {f * 100:.4f}\n"
               f"TP {tp}\tPred. {tp + fp}\tGold. {tp + fn}\n"
               f"\n按情感类别 ECP 指标:\n")

        for emo_idx in range(7):
            metrics = ecp_metrics[emo_idx]
            emotion_name = index_to_emotion.get(emo_idx, f"Unknown_{emo_idx}")
            res += (f"情感 {emotion_name} (样本数: {metrics['gold']}):\n"
                    f"Pre. {metrics['p'] * 100:.4f}\t Rec. {metrics['r'] * 100:.4f}\tF1 {metrics['f'] * 100:.4f}\n"
                    f"TP {metrics['tp']}\tPred. {metrics['pred']}\tGold. {metrics['gold']}\n")

        res += (f"\n平均指标:\n"
                f"Avg.6 (非中性情感 1-6) F1: {avg_6_f1 * 100:.4f}\n"
                f"Avg.4 (主要情感 1-4) F1: {avg_4_f1 * 100:.4f}\n"
                f"\n情感分类指标:\n"
                f"Emo: Pre. {emo[0] * 100:.4f}\t Rec. {emo[1] * 100:.4f}\tF1 {emo[2] * 100:.4f}\n"
                f"原因分类指标:\n"
                f"Cause: Pre. {cause[0] * 100:.4f}\t Rec. {cause[1] * 100:.4f}\tF1 {cause[2] * 100:.4f}\n")

        return f, (res, {
            'p': p,
            'r': r,
            'default': f,
            'emo': emo[2],
            'cause': cause[2],
            'avg_6': avg_6_f1,
            'avg_4': avg_4_f1
        })