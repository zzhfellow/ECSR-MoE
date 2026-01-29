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
from src.loader import (
    read_data,
    read_video,
    build_speaker_dict,
    SupervisedDataset,
    CollateFN,
)
from src.model import TextClassification  # 添加导入
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def load_existing_data(config):
    """加载预处理数据，如果 pkl 不存在则自动生成（使用 config 中的 label_dict）"""
    pkl_path = '/hadata/weim/zzh/emotionmoe/data/preprocessed/roberta-base.pkl'

    if not os.path.exists(pkl_path):
        logger.warning(f"预处理文件 {pkl_path} 不存在，正在自动生成...")

        # ────────────────────────────────────────────────
        # 以下是生成逻辑（基本复制 loader.py 的第一次生成部分）
        # ────────────────────────────────────────────────
        data = {}

        # 读取原始 txt 数据
        for mode in ['train', 'valid', 'test']:
            txt_path = os.path.join(config.dataset_dir, f'{mode}.txt')
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"原始数据文件不存在：{txt_path}")
            data[mode] = read_data(txt_path)

        # 读取视频和音频特征
        data['video'] = read_video(True)   # video=True → video features
        data['audio'] = read_video(False)  # video=False → audio features

        # label_dict：强制使用 config.yaml 中的写死映射（最重要的一行）
        if 'label_dict' not in config or not config.label_dict:
            raise ValueError("config 中缺少 label_dict，请检查 config.yaml")
        data['label_dict'] = config.label_dict.copy()

        # speaker_dict：从全量数据构建（和 loader.py 一致）
        data['speaker_dict'] = build_speaker_dict(
            data['train'] + data['valid'] + data['test']
        )

        # 保存到 pkl
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as f:
            pkl.dump(data, f)
        logger.info(f"已自动生成预处理文件：{pkl_path}")

    # ────────────────────────────────────────────────
    # 无论原来是否存在，现在 pkl 都应该存在了，安全加载
    # ────────────────────────────────────────────────
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)

    # 可选：校验 label_dict 是否与当前 config 一致（强烈建议保留）
    if data.get('label_dict') != config.get('label_dict'):
        logger.warning(
            "警告：pkl 中的 label_dict 与 config.yaml 不一致！\n"
            f"config  : {config.get('label_dict')}\n"
            f"pkl     : {data.get('label_dict')}\n"
            "已使用 pkl 中的版本继续运行。如有疑虑请删除 pkl 重新生成。"
        )

    # 更新 config（以 pkl 为准，或强制同步 config）
    config['label_dict'] = data['label_dict']
    config['speaker_dict'] = data['speaker_dict']

    # 创建测试集 dataset
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