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
import traceback


def get_explanation(feature_path: str, antibiotic: str, isolate_ids: list = None, n_steps: int = 50) -> dict:
    """
    计算 Integrated Gradients 特征重要性（优化版：只计算指定菌株的子图）
    
    Args:
        feature_path: 特征文件夹路径（可以是相对路径或绝对路径）
        antibiotic: 抗生素名称
        isolate_ids: 可选，指定要解释的菌株ID列表（如果不指定，只返回第一个）
        n_steps: IG 积分步数（默认 50，可适当降低到 20-30 提升速度）
    
    Returns:
        dict: {
            "isolate_ids": [菌株ID列表],
            "attributions": [[特征重要性分数列表], ...],  # 每个菌株对应一个列表，长度为特征维度
            "attribution_shape": [len(isolate_ids), 特征维度]
        }
    """
    import torch
    import traceback

    # 动态获取项目根目录（假设当前文件在 src/ 下，项目根目录为 src 的上一级）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. 构建临时配置（只覆盖需要动态传入的参数）
    overrides = [
        f"data.input_dir={feature_path}",
        f"data.antimicrobial={antibiotic}",
        f"trainer.model_checkpoint.dirpath={os.path.join(project_root, 'experiments/checkpoints')}",
    ]

    temp_ids_file = None
    if isolate_ids:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ids', delete=False) as f:
            f.write('\n'.join(isolate_ids))
            temp_ids_file = f.name
        overrides.append(f"data.predict_ids={temp_ids_file}")

    try:
        # 2. 加载 Hydra 配置（使用 explain.yaml，内部路径应为相对路径）
        print("1. 加载 Hydra 配置...")
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="explain", overrides=overrides)
        print("   ✓ 配置加载成功")

        # 3. 创建数据集
        print("2. 创建数据集...")
        dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
        isolate_codes = dataset.isolate_codes
        print(f"   ✓ 数据集创建成功，总节点数: {dataset.x.shape[0]}")

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

        # 6. 确定要解释的节点索引（核心优化：只计算用户指定的菌株）
        print("4. 确定要解释的菌株...")
        if isolate_ids:
            target_indices = []
            for target_id in isolate_ids:
                try:
                    idx = list(isolate_codes).index(target_id)
                    target_indices.append(idx)
                except ValueError:
                    print(f"   警告: 菌株 {target_id} 不在数据集中，已跳过")
            
            if not target_indices:
                target_indices = [0]
                print("   警告: 未找到指定菌株，将使用第一个菌株")
        else:
            target_indices = [0]
            print("   未指定菌株，将只计算第一个菌株")
        
        print(f"   将计算 {len(target_indices)} 个菌株的 attribution")
        
        # 7. 获取所有节点特征和边索引（全图）
        x_full = dataset.x.to(device)
        edge_index_1 = dataset.edge_index_1.to(device)
        edge_index_2 = dataset.edge_index_2.to(device)
        
        # 构建子图（包含目标节点及其一阶邻居）
        target_set = set(target_indices)
        
        # 找出所有与目标节点相邻的节点（一阶邻居）
        neighbor_indices = set()
        for edge in range(edge_index_1.shape[1]):
            src = edge_index_1[0, edge].item()
            dst = edge_index_1[1, edge].item()
            if src in target_set or dst in target_set:
                neighbor_indices.add(src)
                neighbor_indices.add(dst)
        
        # 合并目标节点和邻居节点
        subgraph_indices = list(target_set.union(neighbor_indices))
        subgraph_indices.sort()
        
        # 创建原始索引到子图索引的映射
        old_to_new = {old: new for new, old in enumerate(subgraph_indices)}
        
        # 提取子图节点特征
        x = x_full[subgraph_indices]
        
        # 重新映射边索引到子图
        def remap_edge_index(edge_index):
            edges = []
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                if src in old_to_new and dst in old_to_new:
                    edges.append([old_to_new[src], old_to_new[dst]])
            if edges:
                return torch.tensor(edges, device=device).t().contiguous()
            else:
                return torch.zeros((2, 0), dtype=torch.long, device=device)
        
        edge_index_1_sub = remap_edge_index(edge_index_1)
        edge_index_2_sub = remap_edge_index(edge_index_2)
        
        # 获取子图中目标节点的新索引
        new_target_indices = [old_to_new[idx] for idx in target_indices]
        
        print(f"   子图节点数: {x.shape[0]}, 目标节点数: {len(new_target_indices)}")
        
        # 8. 计算 Integrated Gradients（只对子图中的目标节点）
        print(f"5. 计算 Integrated Gradients (n_steps={n_steps})...")
        dl = IntegratedGradients(model)

        # 处理 baseline
        if hasattr(cfg.explainer, 'baseline') and cfg.explainer.baseline and cfg.explainer.baseline.endswith(".pt"):
            baseline = torch.load(cfg.explainer.baseline).to(device)
        else:
            baseline = torch.zeros_like(x)
            print("   使用零基线")

        attribution_full = dl.attribute(
            x,
            target=None,
            baselines=baseline,
            additional_forward_args=(edge_index_1_sub, edge_index_2_sub),
            internal_batch_size=min(x.size(0), 16),
            n_steps=n_steps,
        )
        print("   ✓ IG 计算完成")

        # 9. 只返回目标节点的 attribution（完整特征重要性向量）
        attribution = attribution_full[new_target_indices]
        attributions_np = attribution.detach().cpu().numpy()   # shape: (len(target_indices), feature_dim)

        # 获取对应的 isolate_ids
        result_isolate_ids = [isolate_codes[idx] for idx in target_indices]

        # 直接返回完整向量，不对特征维度做任何压缩
        return {
            "isolate_ids": result_isolate_ids,
            "attributions": attributions_np.tolist(),          # 每个菌株的完整特征分数列表
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