# 抑制 pkg_resources 弃用警告（来自 lightning 库）
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightning")

import sys
import os

# 关键修复：添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import torch
import numpy as np
from hydra import compose, initialize
from src.models import AMRNodeDualGNN
from src.utils import create_graph_dataset
import glob
import traceback


def get_prediction(feature_path: str, antibiotic: str, isolate_ids: list = None) -> dict:
    """
    根据特征文件夹和抗生素名称进行预测（供 FastAPI 调用）
    Args:
        feature_path: 预处理后的特征文件夹路径（可能为相对路径，如 './data/extracted_unitigs'）
        antibiotic: 抗生素名称，如 "vancomycin"
        isolate_ids: 可选，指定要预测的菌株ID列表
    Returns:
        dict: {"y_proba": [[敏感概率, 耐药概率], ...], "y_pred": [0/1, ...], "isolate_ids": ["id1", ...]}
    """
    # 获取项目根目录（src目录的上一级）
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 相对路径转绝对路径（基于项目根目录）
    if feature_path.startswith('./'):
        feature_path = os.path.join(current_dir, feature_path.lstrip('./'))
    elif not os.path.isabs(feature_path):
        feature_path = os.path.join(current_dir, feature_path)

    # 初始化返回结果（所有字段默认值为列表，绝对不会是None）
    result = {
        "y_proba": [],
        "y_pred": [],
        "isolate_ids": []
    }

    temp_ids_file = None
    try:
        # 1. 构建临时配置
        overrides = [
            f"data.input_dir={feature_path}",
            f"data.antimicrobial={antibiotic}",
            f"data.labels={os.path.join(current_dir, 'data/ast_labels.csv')}",
            f"data.whole_ids={os.path.join(current_dir, 'data/whole.ids')}",
            f"data.predict_ids={os.path.join(current_dir, 'data/predict.ids')}",
            f"adj_matrix.file_path_1={os.path.join(current_dir, 'data/fcgr_adj_matrix.csv')}",
            f"adj_matrix.file_path_2={os.path.join(current_dir, 'data/snps_adj_matrix.csv')}",
            f"trainer.model_checkpoint.dirpath={os.path.join(current_dir, 'experiments/checkpoints')}",
        ]

        # 如果指定了菌株ID，创建临时ids文件
        if isolate_ids and len(isolate_ids) > 0:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ids', delete=False, encoding='utf-8') as f:
                f.write('\n'.join(isolate_ids))
            temp_ids_file = f.name
            overrides.append(f"data.predict_ids={temp_ids_file}")

        # 2. 加载配置和数据集（节点级任务：整个数据集是一个大图）
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="config", overrides=overrides)

        # 正确解包create_graph_dataset返回的元组
        dataset_output = create_graph_dataset(cfg)
        if isinstance(dataset_output, tuple):
            print(f"调试信息：create_graph_dataset返回了包含{len(dataset_output)}个元素的元组")
            data = dataset_output[0]  # 第一个元素是Data对象
            print(f"调试信息：Data对象包含的属性: {list(data.keys())}")
        else:
            data = dataset_output
            print(f"调试信息：create_graph_dataset返回了单个Data对象")

        # ✅ 关键修复1：使用Data对象内置的predict_mask过滤结果
        if hasattr(data, 'predict_mask'):
            predict_mask = data.predict_mask.bool()
            predict_count = int(predict_mask.sum().item())
            print(f"调试信息：predict_mask标记了{predict_count}个需要预测的节点")
            print(f"调试信息：predict_mask形状: {predict_mask.shape}")
        else:
            raise AttributeError("Data对象缺少predict_mask属性，无法确定需要预测的节点")

        # ✅ 关键修复2：使用Data对象内置的isolate_codes获取菌株ID
        if hasattr(data, 'isolate_codes'):
            all_isolate_ids = data.isolate_codes
            print(f"调试信息：Data对象包含{len(all_isolate_ids)}个菌株ID")
        elif hasattr(data, 'ids'):
            all_isolate_ids = data.ids
            print(f"调试信息：使用Data对象的ids属性，共{len(all_isolate_ids)}个")
        else:
            # 降级方案：从predict.ids文件读取
            print("警告：Data对象缺少isolate_codes和ids属性，将从predict.ids文件读取")
            with open(cfg.data.predict_ids, 'r', encoding='utf-8') as f:
                all_isolate_ids = [line.strip() for line in f if line.strip()]

        # 3. 加载模型权重
        checkpoint_dir = cfg.trainer.model_checkpoint.dirpath
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

        if not ckpt_files:
            raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到任何模型权重文件(.ckpt)")

        checkpoint = torch.load(ckpt_files[0], map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        # 过滤掉损失函数相关的参数（避免加载失败）
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('loss_fn')}

        model = AMRNodeDualGNN(
            cfg=cfg,
            c_in=data.x.size(-1),
            class_weight=None
        )
        model.load_state_dict(filtered_dict, strict=False)
        model.eval()
        model.to('cpu')  # 强制使用CPU，避免GPU环境兼容性问题

        # 4. 执行预测（节点级任务：直接传入整个图）
        with torch.no_grad():
            batch = data.to(model.device)

            # 双重保险：给batch添加train_mask属性（避免模型报错）
            if not hasattr(batch, 'train_mask'):
                batch.train_mask = torch.zeros_like(batch.y, dtype=torch.bool)

            output = model(batch, mode="predict")

            # 处理不同的输出格式
            if isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output

            if logits.numel() == 0:
                raise ValueError("模型输出为空，请检查数据和模型配置")

            # ✅ 关键修复3：正确处理单通道sigmoid输出
            if logits.dim() == 1 or logits.size(1) == 1:
                # 模型输出是单值：耐药概率的logit
                drug_resistant_prob = torch.sigmoid(logits).squeeze()
                sensitive_prob = 1 - drug_resistant_prob
                # 转换成 [敏感概率, 耐药概率] 标准格式
                all_probs = torch.stack([sensitive_prob, drug_resistant_prob], dim=1)
            else:
                # 模型输出是双通道：直接用softmax
                all_probs = torch.softmax(logits, dim=1)

            all_preds = torch.argmax(all_probs, dim=1)

            # ✅ 关键修复4：只提取predict_mask标记的节点的结果
            predict_probs = all_probs[predict_mask].cpu().numpy()
            predict_preds = all_preds[predict_mask].cpu().numpy()

            # 提取对应的菌株ID
            if len(all_isolate_ids) == len(predict_mask):
                # all_isolate_ids包含所有节点的ID，只取predict_mask为True的
                predict_isolate_ids = [all_isolate_ids[i] for i in range(len(predict_mask)) if predict_mask[i]]
            else:
                # all_isolate_ids已经是预测节点的ID（从文件读取的情况）
                predict_isolate_ids = all_isolate_ids

        # 5. 整理结果
        result["y_proba"] = predict_probs.tolist()
        result["y_pred"] = predict_preds.tolist()
        result["isolate_ids"] = predict_isolate_ids

        # 验证结果长度一致
        if len(result["y_proba"]) != len(result["isolate_ids"]):
            print(f"警告：预测结果数量({len(result['y_proba'])})与菌株ID数量({len(result['isolate_ids'])})不一致")
            print(f"这可能意味着isolate_codes属性的长度与节点数量不匹配")
        else:
            print(f"✅ 预测完成！共预测{len(result['y_proba'])}个菌株")

        return result

    except Exception as e:
        print("=" * 80)
        print("预测函数内部错误详情：")
        traceback.print_exc()
        print("=" * 80)
        raise RuntimeError(f"预测失败: {str(e)}") from e

    finally:
        # 清理临时文件
        if temp_ids_file and os.path.exists(temp_ids_file):
            try:
                os.unlink(temp_ids_file)
            except:
                pass


# 本地测试入口（从项目根目录执行：uv run python src/predict.py）
if __name__ == "__main__":
    try:
        result = get_prediction("./data/extracted_unitigs", "vancomycin")
        print("✅ 本地测试成功！")
        print(f"最终预测样本数: {len(result['y_proba'])}")
        print(f"最终菌株ID数: {len(result['isolate_ids'])}")
        print("前10个预测结果:")
        for i in range(min(10, len(result['y_proba']))):
            print(f"  菌株: {result['isolate_ids'][i]}, "
                  f"敏感概率: {result['y_proba'][i][0]:.4f}, "
                  f"耐药概率: {result['y_proba'][i][1]:.4f}, "
                  f"预测结果: {'耐药' if result['y_pred'][i] == 1 else '敏感'}")

        # 统计预测结果分布
        resistant_count = sum(result['y_pred'])
        sensitive_count = len(result['y_pred']) - resistant_count
        print(f"\n预测结果统计:")
        print(f"  耐药菌株: {resistant_count}个 ({resistant_count / len(result['y_pred']) * 100:.1f}%)")
        print(f"  敏感菌株: {sensitive_count}个 ({sensitive_count / len(result['y_pred']) * 100:.1f}%)")

    except Exception as e:
        print(f"❌ 本地测试失败: {e}")