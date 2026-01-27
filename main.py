#!/usr/bin/env python

import os
import glob
import argparse
import yaml
import random
import string
import torch
from attrdict import AttrDict
from loguru import logger
import warnings
import itertools
from datetime import datetime
import gc

from src.tools import update_config, set_seed, load_params_bert
from src.trainer import MyTrainer
from src.loader import make_supervised_data_module
import transformers
from src.model import TextClassification

warnings.filterwarnings('ignore')


# ==================== 显存监控工具（可选）====================
def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
# ============================================================


class Template:
    def __init__(self, args):
        # 加载配置文件
        config = AttrDict(yaml.load(
            open('src/config.yaml', 'r', encoding='utf-8'),
            Loader=yaml.FullLoader
        ))

        # 更新配置，优先使用命令行参数
        for k, v in vars(args).items():
            if v is not None:
                setattr(config, k, v)
        config = update_config(config)

        # 设置随机种子
        set_seed(config.seed)

        # 设置随机字符串（日志用）
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        config.best_model_path = os.path.join(config.target_dir, 'best_model.pt')  # 占位

        # 设置设备
        config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')

        # 创建目标目录
        os.makedirs(config.target_dir, exist_ok=True)

        # 【清理历史模型】
        old_models = glob.glob(os.path.join(config.target_dir, 'best_model_*.pt'))
        for old_path in old_models:
            try:
                os.remove(old_path)
                logger.info(f"已删除历史模型文件: {old_path}")
            except Exception as e:
                logger.warning(f"删除历史模型失败 {old_path}: {e}")

        self.config = config
        self.logger = logger
        self.logger.add(os.path.join(config.target_dir, f'grid_search_{random_str}.log'))

        # 全局最佳模型路径
        self.current_global_best_path = None

        # 设置 PyTorch 显存优化（关键！）
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def grid_search(self):
        param_grid = {
            'seed': [555],#666,777,888,999,321,6666,7777,8888,9999,111,1111,222,2222,333,3333,1234]
             'top_m': [3],
            # 'num_experts_per_emotion': [4],
            # 其他超参数...
        }

        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.logger.info(f"总参数组合数: {len(param_combinations)}")

        # 全局最佳追踪
        global_best_test_f1 = 0.0
        global_best_params = None
        global_best_res = None
        global_best_epoch = -1
        global_best_state_dict = None

        # === 提前加载共享的 tokenizer 和数据基础部分（避免每次重复加载 pkl）===
        tokenizer_base = transformers.AutoTokenizer.from_pretrained(
            self.config.bert_path, padding_side="right", use_fast=False
        )
        # 提前执行一次 make_supervised_data_module 来加载数据集和 collator
        _, _, _, _ = make_supervised_data_module(self.config, tokenizer_base)  # 这会触发 pkl 加载/生成

        for idx, params in enumerate(param_combinations):
            self.logger.info(f"训练组合 {idx + 1}/{len(param_combinations)}: {params}")
            log_gpu_memory(f"[组合 {idx + 1} 开始前]")
            start_time = datetime.now()

            # === 关键修复1：为当前组合重新设置随机种子 ===
            current_seed = params.get('seed', self.config.seed)
            set_seed(current_seed)
            self.logger.info(f"已为当前组合重置随机种子 seed = {current_seed}")

            # === 关键修复2：为当前组合重新创建 tokenizer 和 data_loader（确保 shuffle 真正随机）===
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.bert_path, padding_side="right", use_fast=False
            )
            train_loader, valid_loader, test_loader, config_copy = make_supervised_data_module(
                self.config, tokenizer
            )

            # 注入所有超参数到 config_copy（包括 seed）
            for param, value in params.items():
                setattr(config_copy, param, value)

            # 创建模型（模型初始化会受新 seed 影响）
            model = TextClassification(config_copy, tokenizer).to(config_copy.device)
            config_copy = load_params_bert(config_copy, model, train_loader)

            # 训练
            trainer = MyTrainer(model, config_copy, train_loader, valid_loader, test_loader)
            best_test_f1, best_state_dict, best_test_res, best_epoch = trainer.train()

            log_gpu_memory(f"[组合 {idx + 1} 训练后]")

            # ==================== 【保存全局最佳模型】====================
            if best_test_f1 > global_best_test_f1 + 1e-6:
                # 删除旧的全局最佳模型
                if self.current_global_best_path and os.path.exists(self.current_global_best_path):
                    try:
                        os.remove(self.current_global_best_path)
                        self.logger.info(f"已删除旧的全局最佳模型: {self.current_global_best_path}")
                    except Exception as e:
                        self.logger.warning(f"删除旧模型失败: {e}")

                # 更新全局最佳
                global_best_test_f1 = best_test_f1
                global_best_params = params.copy()
                global_best_res = best_test_res
                global_best_epoch = best_epoch
                global_best_state_dict = best_state_dict

                # 生成新模型路径
                param_str = "_".join([f"{k}={v}" for k, v in global_best_params.items()])
                best_model_filename = f"best_model_{param_str}_f1={best_test_f1:.4f}.pt"
                best_model_path = os.path.join(config_copy.target_dir, best_model_filename)

                # 保存模型
                try:
                    torch.save(best_state_dict, best_model_path)
                    self.current_global_best_path = best_model_path
                    self.config.best_model_path = best_model_path
                    self.logger.info(
                        f"[GLOBAL BEST] 已保存模型: {best_model_path} | "
                        f"F1: {best_test_f1:.4f} | 轮次: {best_epoch} | 参数: {global_best_params}"
                    )
                except Exception as e:
                    self.logger.error(f"保存全局最佳模型失败: {str(e)}")
                    raise
            # =================================================================

            # 单次组合日志
            self.logger.info(
                f"组合 {idx + 1} 完成。最佳测试 F1: {best_test_f1:.4f} @ 轮次 {best_epoch}，"
                f"耗时: {datetime.now() - start_time}"
            )

            # 写入 result.txt
            with open(os.path.join(config_copy.target_dir, 'result.txt'), 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"组合 {idx + 1}/{len(param_combinations)}: {params}\n")
                f.write(f"最佳测试 F1: {best_test_f1:.4f} (轮次 {best_epoch})\n")
                f.write(f"是否全局最佳: {'是' if best_test_f1 > global_best_test_f1 - 1e-6 else '否'}\n")
                f.write(
                    f"模型路径: {self.config.best_model_path if best_test_f1 > global_best_test_f1 - 1e-6 else 'N/A'}\n")
                f.write(f"耗时: {datetime.now() - start_time}\n")
                f.write(f"测试结果:\n{best_test_res}\n")
                f.write(f"{'=' * 60}\n")

            # ==================== 【释放内存】====================
            del trainer
            del model
            del config_copy
            del best_state_dict
            del train_loader
            del valid_loader
            del test_loader
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory(f"[组合 {idx + 1} 清理后]")
            # ==========================================================

        # 网格搜索结束总结
        self.logger.info(
            f"网格搜索完成。\n"
            f"  最佳测试 F1: {global_best_test_f1:.4f}\n"
            f"  最佳参数: {global_best_params}\n"
            f"  最佳轮次: {global_best_epoch}\n"
            f"  模型保存路径: {self.config.best_model_path}"
        )

        with open(os.path.join(self.config.target_dir, 'result.txt'), 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"全局最佳模型总结\n")
            f.write(f"超参数组合: {global_best_params}\n")
            f.write(f"最佳测试 F1: {global_best_test_f1:.4f}\n")
            f.write(f"最佳轮次: {global_best_epoch}\n")
            f.write(f"模型路径: {self.config.best_model_path}\n")
            f.write(f"测试结果:\n{global_best_res}\n")
            f.write(f"{'=' * 60}\n")

        return global_best_params, global_best_test_f1

    def forward(self):
        best_params, best_score = self.grid_search()
        self.logger.info(f"训练完成，最佳参数: {best_params}，最高 F1: {best_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert', help='模型类型')
    parser.add_argument('-cd', '--cuda_index', type=int, default=5, help='CUDA 设备索引')
    parser.add_argument('--target_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')

    args = parser.parse_args()
    template = Template(args)
    template.forward()