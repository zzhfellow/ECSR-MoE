#!/usr/bin/env python
import os
import yaml
import torch
import pickle as pkl
from attrdict import AttrDict
from loguru import logger
import warnings
import transformers
from src.tools import update_config, set_seed, load_params_bert
from src.trainer import MyTrainer
from src.loader import SupervisedDataset, CollateFN
from src.model import TextClassification  # 添加导入
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def load_existing_data(config):
    """直接加载现有的 .pkl 文件，使用绝对路径"""
    pkl_path = '/hadata/weim/zzh/emotionmoe/data/preprocessed/roberta-base.pkl'  # 硬编码绝对路径
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"预处理数据文件 {pkl_path} 不存在，请检查路径。")

    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)

    # 确保 config 包含必要的字典
    config['label_dict'] = data['label_dict']
    config['speaker_dict'] = data['speaker_dict']

    # 创建数据集
    test_dataset = SupervisedDataset(data, 'test')

    return test_dataset, data['video'], data['audio'], config


def make_supervised_data_module(config, tokenizer):
    """创建测试数据加载器，直接使用现有 .pkl 文件"""
    test_dataset, video_map, audio_map, config = load_existing_data(config)

    # 创建 collator
    data_collator = CollateFN(tokenizer=tokenizer, config=config, video_map=video_map, audio_map=audio_map)

    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=data_collator)

    # 返回 train_loader=None（因为仅评估），valid_loader=None，test_loader 和 config
    return None, None, test_loader, config


def main():
    # 加载配置文件
    config = AttrDict(yaml.load(
        open('src/config.yaml', 'r', encoding='utf-8'),
        Loader=yaml.FullLoader
    ))

    # 更新配置（指定预训练模型路径）
    config.pretrained_model_path = '/hadata/weim/zzh/emotionmoe/data/save/best_model/best_model_seed=555_top_k=2_f1=0.5838.pt'
    config = update_config(config)

    # 设置随机种子和设备
    set_seed(config.seed)
    config.device = torch.device(f'cuda:{config.cuda_index}' if torch.cuda.is_available() else 'cpu')

    # 初始化日志
    logger.add(os.path.join(config.target_dir, 'evaluation.log'))
    logger.info("开始评估最佳模型...")

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.bert_path, padding_side="right", use_fast=False)

    # 创建数据加载器（仅加载测试集）
    train_loader, valid_loader, test_loader, config = make_supervised_data_module(config, tokenizer)

    # 实例化模型
    model = TextClassification(config, tokenizer).to(config.device)

    # 加载最佳模型权重
    if os.path.exists(config.pretrained_model_path):
        pretrained_state_dict = torch.load(config.pretrained_model_path, map_location=config.device)
        # model.load_state_dict(pretrained_state_dict, strict=True)
        model.load_state_dict(pretrained_state_dict, strict=False)
        logger.info(f"成功加载最佳模型权重从 {config.pretrained_model_path}")
    else:
        raise FileNotFoundError(f"模型路径 {config.pretrained_model_path} 不存在，请检查。")

    # 设置优化器（尽管评估不需要，但 trainer 初始化可能需要）
    config = load_params_bert(config, model, test_loader)

    # 创建 trainer 实例（仅使用 test_loader）
    trainer = MyTrainer(model, config, train_loader=None, valid_loader=None, test_loader=test_loader)

    # 执行评估
    score, (res, metrics) = trainer.final_evaluate()
    # === 新增：显式打印 EmotionMoE 专家激活统计 ===
    if hasattr(model, 'emotion_moe'):
        print("\n" + "=" * 60)
        print("EmotionMoE Expert Activation Statistics (Test Set)")
        print("=" * 60)
        model.emotion_moe.print_activation_stats()
        print("=" * 60 + "\n")

    # 原有日志和打印
    logger.info(f"测试集评估结果:\n{res}")
    logger.info(f"默认 F1 分数: {metrics['default'] * 100:.4f}%")
    print(f"测试集评估结果:\n{res}")


if __name__ == '__main__':
    main()