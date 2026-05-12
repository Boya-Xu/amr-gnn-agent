# src/explain_api.py
import glob
import os
import tempfile
import torch
import numpy as np
from captum.attr import IntegratedGradients
from hydra import compose, initialize
from src.models import GNNModel
from src.utils import create_graph_dataset


def get_explanation(feature_path: str, antibiotic: str, isolate_ids: list = None, n_steps: int = 50) -> dict:
    """
    计算 Integrated Gradients 特征重要性

    Args:
        feature_path: 特征文件夹路径
        antibiotic: 抗生素名称
        isolate_ids: 可选，指定要解释的菌株ID列表
        n_steps: IG 积分步数

    Returns:
        dict: {
            "isolate_ids": [...],
            "attributions": [...],
            "attribution_shape": [...]
        }
    """
    import torch
    import traceback

    # 项目根目录的绝对路径
    PROJECT_ROOT = "D:/project/resistance-prediction/amr-gnn-agent"

    # 1. 构建临时配置（只覆盖需要动态传入的参数）
    overrides = [
        f"data.input_dir={feature_path}",
        f"data.antimicrobial={antibiotic}",
        f"trainer.model_checkpoint.dirpath={PROJECT_ROOT}/experiments/checkpoints",
    ]

    temp_ids_file = None
    if isolate_ids:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ids', delete=False) as f:
            f.write('\n'.join(isolate_ids))
            temp_ids_file = f.name
        overrides.append(f"data.predict_ids={temp_ids_file}")

    try:
        # 2. 加载 Hydra 配置
        print("1. 加载 Hydra 配置...")
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="explain", overrides=overrides)
        print("   ✓ 配置加载成功")

        # 3. 创建数据集
        print("2. 创建数据集...")
        dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
        isolate_codes = dataset.isolate_codes
        print(f"   ✓ 数据集创建成功，节点数: {dataset.x.shape[0]}")

        # 4. 确定设备
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"   使用设备: {device}")

        # 5. 加载模型
        print("3. 加载模型...")
        model_kwargs = {}
        if cfg.gnn.layer_name in ["GAT", "GATv2"]:
            model_kwargs["heads"] = cfg.gnn.GAT.heads
        elif cfg.gnn.layer_name == "TransformerConv":
            model_kwargs["heads"] = cfg.gnn.TransformerConv.heads

        model = GNNModel(cfg=cfg, c_in=dataset.x.size(-1), **model_kwargs)

        checkpoint_dir = cfg.trainer.model_checkpoint.dirpath
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        print(f"   找到权重文件: {ckpt_files[0]}")

        checkpoint = torch.load(ckpt_files[0], map_location=device)
        model_weights = {
            k.lstrip("model").lstrip("."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(model_weights)
        model.eval()
        model.to(device)
        print("   ✓ 模型加载成功")

        # 6. 准备数据
        print("4. 准备数据...")
        x = dataset.x.to(device)
        edge_index_1 = dataset.edge_index_1.to(device)
        edge_index_2 = dataset.edge_index_2.to(device)
        print(f"   特征形状: {x.shape}")

        # 7. 计算 Integrated Gradients
        print("5. 计算 Integrated Gradients...")
        dl = IntegratedGradients(model)

        # 处理 baseline
        if hasattr(cfg.explainer, 'baseline') and cfg.explainer.baseline and cfg.explainer.baseline.endswith(".pt"):
            baseline = torch.load(cfg.explainer.baseline).to(device)
        else:
            baseline = None  # Captum 默认使用零基线

        attribution = dl.attribute(
            x,
            target=None,
            baselines=baseline,
            additional_forward_args=(edge_index_1, edge_index_2),
            internal_batch_size=min(x.size(0), 32),
            n_steps=n_steps,
        )
        print("   ✓ IG 计算完成")

        # 8. 整理结果
        attributions_np = attribution.detach().cpu().numpy()

        # 按样本汇总特征重要性（取均值）
        sample_attributions = np.mean(attributions_np, axis=1).tolist()

        return {
            "isolate_ids": list(isolate_codes),
            "attributions": sample_attributions,
            "attribution_shape": list(attributions_np.shape)
        }

    except Exception as e:
        print("=" * 50)
        print("get_explanation 内部错误:")
        traceback.print_exc()
        print("=" * 50)
        raise

    finally:
        if temp_ids_file and os.path.exists(temp_ids_file):
            os.unlink(temp_ids_file)