# src/explain_api.py
import glob
import os
import pickle
import hashlib
import tempfile
import torch
import numpy as np
from functools import lru_cache
from collections import OrderedDict
from captum.attr import IntegratedGradients
from hydra import compose, initialize
from src.models import GNNModel
from src.utils import create_graph_dataset
import traceback
from typing import List, Optional, Dict, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationCache:
    def __init__(self, max_size: int = 32, cache_dir: str = ".explanation_cache"):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self.memory_cache = OrderedDict()
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, feature_path: str, antibiotic: str, isolate_ids: Tuple[str, ...], n_steps: int) -> str:
        key_str = f"{feature_path}_{antibiotic}_{isolate_ids}_{n_steps}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get(self, feature_path: str, antibiotic: str, isolate_ids: Tuple[str, ...], n_steps: int) -> Optional[Dict]:
        cache_key = self._get_cache_key(feature_path, antibiotic, isolate_ids, n_steps)
        if cache_key in self.memory_cache:
            logger.debug(f"使用内存缓存结果: {cache_key[:8]}...")
            self.memory_cache.move_to_end(cache_key)
            return self.memory_cache[cache_key]
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                logger.debug(f"使用磁盘缓存结果: {cache_key[:8]}...")
                self._add_to_memory_cache(cache_key, result)
                return result
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")
        return None

    def set(self, feature_path: str, antibiotic: str, isolate_ids: Tuple[str, ...], n_steps: int, result: Dict):
        cache_key = self._get_cache_key(feature_path, antibiotic, isolate_ids, n_steps)
        self._add_to_memory_cache(cache_key, result)
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _add_to_memory_cache(self, cache_key: str, result: Dict):
        if cache_key in self.memory_cache:
            self.memory_cache.move_to_end(cache_key)
        else:
            if len(self.memory_cache) >= self.max_size:
                self.memory_cache.popitem(last=False)
            self.memory_cache[cache_key] = result

    def clear(self):
        self.memory_cache.clear()
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))


_explanation_cache = ExplanationCache(max_size=32, cache_dir=".explanation_cache")
_global_model_cache = {}


def get_global_model(feature_path: str, antibiotic: str, project_root: str):
    cache_key = (feature_path, antibiotic)
    if cache_key in _global_model_cache:
        logger.info("使用全局缓存的模型")
        return _global_model_cache[cache_key]
    logger.info("首次加载模型")
    cfg, model, dataset, device = load_model_and_config(feature_path, antibiotic, project_root)
    _global_model_cache[cache_key] = (cfg, model, dataset, device)
    return cfg, model, dataset, device


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_k_hop_neighbors(edge_index: torch.Tensor, target_nodes: Set[int], k: int) -> Set[int]:
    if k <= 0:
        return set(target_nodes)
    neighbors = set(target_nodes)
    current_layer = set(target_nodes)
    for _ in range(k):
        next_layer = set()
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src in current_layer and dst not in neighbors:
                next_layer.add(dst)
            if dst in current_layer and src not in neighbors:
                next_layer.add(src)
        neighbors.update(next_layer)
        current_layer = next_layer
        if not next_layer:
            break
    return neighbors


def remap_edge_index_vectorized(edge_index: torch.Tensor, old_to_new: Dict[int, int]) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    max_idx = edge_index.max().item()
    mapper = torch.full((max_idx + 1,), -1, dtype=torch.long, device=edge_index.device)
    for old_idx, new_idx in old_to_new.items():
        mapper[old_idx] = new_idx
    mapped_edges = mapper[edge_index]
    valid_mask = (mapped_edges[0] >= 0) & (mapped_edges[1] >= 0)
    valid_edges = mapped_edges[:, valid_mask]
    if valid_edges.shape[1] == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    return valid_edges


def validate_inputs(feature_path: str, antibiotic: str, isolate_ids: Optional[List[str]], n_steps: int):
    if not feature_path:
        raise ValueError("特征路径不能为空")
    if not antibiotic:
        raise ValueError("抗生素名称不能为空")
    if n_steps < 1:
        raise ValueError(f"n_steps必须大于0，当前值: {n_steps}")
    if isolate_ids is not None and not isinstance(isolate_ids, list):
        raise ValueError(f"isolate_ids必须是列表类型")


def load_model_and_config(feature_path: str, antibiotic: str, project_root: str, isolate_ids: Optional[List[str]] = None):
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
        logger.info("加载Hydra配置...")
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="explain", overrides=overrides)
        logger.info("配置加载成功")
        logger.info("✓ 配置加载成功")
        # 强制修正所有路径，彻底解决Hydra路径解析问题
        cfg.data.labels = os.path.join(project_root, "data", "ast_labels.csv")
        cfg.data.whole_ids = os.path.join(project_root, "data", "whole.ids")
        cfg.data.val_ids = os.path.join(project_root, "data", "val.ids")
        cfg.data.train_ids = os.path.join(project_root, "data", "train.ids")
        cfg.data.predict_ids = os.path.join(project_root, "data", "predict.ids")
        cfg.adj_matrix.file_path_1 = os.path.join(project_root, "data", "fcgr_adj_matrix.csv")
        cfg.adj_matrix.file_path_2 = os.path.join(project_root, "data", "snps_adj_matrix.csv")

        # 创建数据集
        logger.info("创建数据集...")
        dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
        logger.info(f"数据集创建成功，总节点数: {dataset.x.shape[0]}")
        device = get_device()
        logger.info(f"使用设备: {device}")
        logger.info("加载模型...")
        model_kwargs = {}
        if cfg.gnn.layer_name in ["GAT", "GATv2"]:
            model_kwargs["heads"] = cfg.gnn.GAT.heads
        elif cfg.gnn.layer_name == "TransformerConv":
            model_kwargs["heads"] = cfg.gnn.TransformerConv.heads
        model = GNNModel(cfg=cfg, c_in=dataset.x.size(-1), **model_kwargs)
        checkpoint_dir = cfg.trainer.model_checkpoint.dirpath
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到检查点文件")
        logger.info(f"找到权重文件: {ckpt_files[0]}")
        checkpoint = torch.load(ckpt_files[0], map_location=device)
        model_weights = {k.lstrip("model").lstrip("."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(model_weights)
        model.eval()
        model.to(device)
        logger.info("模型加载成功")
        return cfg, model, dataset, device
    finally:
        if temp_ids_file and os.path.exists(temp_ids_file):
            os.unlink(temp_ids_file)


def build_subgraph(dataset, target_indices: List[int], num_gnn_layers: int, device: torch.device) -> Tuple:
    logger.info("构建子图...")
    x_full = dataset.x.to(device)
    edge_index_1 = dataset.edge_index_1.to(device)
    edge_index_2 = dataset.edge_index_2.to(device)
    target_set = set(target_indices)
    neighbors_1 = get_k_hop_neighbors(edge_index_1, target_set, num_gnn_layers)
    neighbors_2 = get_k_hop_neighbors(edge_index_2, target_set, num_gnn_layers)
    all_nodes = target_set.union(neighbors_1).union(neighbors_2)
    subgraph_indices = sorted(list(all_nodes))
    old_to_new = {old: new for new, old in enumerate(subgraph_indices)}
    x_sub = x_full[subgraph_indices]
    edge_index_1_sub = remap_edge_index_vectorized(edge_index_1, old_to_new)
    edge_index_2_sub = remap_edge_index_vectorized(edge_index_2, old_to_new)
    new_target_indices = [old_to_new[idx] for idx in target_indices]
    logger.info(f"子图节点数: {x_sub.shape[0]} (原始: {x_full.shape[0]})")
    logger.info(f"目标节点数: {len(new_target_indices)}")
    return (x_sub, edge_index_1_sub, edge_index_2_sub, new_target_indices, old_to_new, subgraph_indices)


def compute_integrated_gradients(model, x, edge_index_1, edge_index_2, target_indices, n_steps=20, baseline=None, internal_batch_size=16):
    logger.info(f"计算Integrated Gradients (n_steps={n_steps})...")
    if baseline is None:
        baseline = torch.zeros_like(x)
        logger.info("使用零基线")
    ig = IntegratedGradients(model)
    actual_batch_size = min(x.size(0), internal_batch_size)
    attribution_full = ig.attribute(x, target=None, baselines=baseline, additional_forward_args=(edge_index_1, edge_index_2), internal_batch_size=actual_batch_size, n_steps=n_steps)
    attribution_target = attribution_full[target_indices]
    logger.info("IG计算完成")
    return attribution_target


def get_explanation(feature_path: str, antibiotic: str, isolate_ids: Optional[List[str]] = None, n_steps: int = 20, use_cache: bool = True) -> Dict:
    import time
    start_time = time.time()
    validate_inputs(feature_path, antibiotic, isolate_ids, n_steps)
    isolate_ids_tuple = tuple(isolate_ids) if isolate_ids else ()
    if use_cache:
        cached_result = _explanation_cache.get(feature_path, antibiotic, isolate_ids_tuple, n_steps)
        if cached_result is not None:
            logger.info("使用结果缓存")
            return cached_result
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        cfg, model, dataset, device = get_global_model(feature_path, antibiotic, project_root)
        isolate_codes = dataset.isolate_codes

        logger.info("确定要解释的菌株...")
        if isolate_ids:
            target_indices = []
            found_ids = []
            missing_ids = []
            for target_id in isolate_ids:
                try:
                    idx = list(isolate_codes).index(target_id)
                    target_indices.append(idx)
                    found_ids.append(target_id)
                except ValueError:
                    missing_ids.append(target_id)
                    logger.warning(f"菌株 {target_id} 不在数据集中，已跳过")
            if missing_ids:
                logger.warning(f"跳过的菌株: {missing_ids}")
            if not target_indices:
                logger.error(f"所有指定的菌株都不在数据集中: {isolate_ids}")
                return {
                    "isolate_ids": [],
                    "attributions": [],
                    "attribution_shape": [0, 0],
                    "error": f"指定的菌株都不在数据集中: {isolate_ids}",
                    "metadata": {"computation_time_seconds": time.time() - start_time, "status": "failed"}
                }
            isolate_ids = found_ids
        else:
            target_indices = [0]
            found_ids = [isolate_codes[0]]
            logger.info("未指定菌株，将使用第一个菌株")

        logger.info(f"将计算 {len(target_indices)} 个菌株的归因")
        num_gnn_layers = getattr(cfg.gnn, 'num_layers', 2)
        x_sub, edge_index_1_sub, edge_index_2_sub, new_target_indices, old_to_new, subgraph_indices = build_subgraph(dataset, target_indices, num_gnn_layers, device)
        baseline = None
        if hasattr(cfg.explainer, 'baseline') and cfg.explainer.baseline:
            baseline_path = cfg.explainer.baseline
            if baseline_path.endswith(".pt") and os.path.exists(baseline_path):
                baseline = torch.load(baseline_path).to(device)
                if baseline.shape[0] == dataset.x.shape[0]:
                    baseline = baseline[subgraph_indices]
                logger.info(f"使用自定义基线: {baseline_path}")
        attribution_target = compute_integrated_gradients(model, x_sub, edge_index_1_sub, edge_index_2_sub, new_target_indices, n_steps, baseline)
        attributions_np = attribution_target.detach().cpu().numpy()
        result_isolate_ids = found_ids if isolate_ids else [isolate_codes[0]]
        computation_time = time.time() - start_time
        result = {
            "isolate_ids": result_isolate_ids,
            "attributions": attributions_np.tolist(),
            "attribution_shape": list(attributions_np.shape),
            "metadata": {
                "num_subgraph_nodes": x_sub.shape[0],
                "num_total_nodes": dataset.x.shape[0],
                "num_gnn_layers": num_gnn_layers,
                "computation_time_seconds": computation_time,
                "device": str(device),
                "n_steps": n_steps,
            }
        }
        if use_cache:
            _explanation_cache.set(feature_path, antibiotic, isolate_ids_tuple, n_steps, result)
        logger.info(f"解释完成，耗时: {computation_time:.2f}秒")
        return result
    except Exception as e:
        logger.error(f"get_explanation 内部错误: {e}")
        return {
            "isolate_ids": isolate_ids if isolate_ids else [],
            "attributions": [],
            "attribution_shape": [0, 0],
            "error": str(e),
            "metadata": {"computation_time_seconds": time.time() - start_time, "status": "failed"}
        }


def get_explanation_batch(feature_path: str, antibiotic: str, isolate_ids_list: List[List[str]], n_steps: int = 20, use_cache: bool = True) -> List[Dict]:
    results = []
    for i, isolate_ids in enumerate(isolate_ids_list):
        logger.info(f"处理批次 {i+1}/{len(isolate_ids_list)}")
        result = get_explanation(feature_path, antibiotic, isolate_ids, n_steps, use_cache)
        results.append(result)
    return results


def clear_cache():
    _explanation_cache.clear()
    _global_model_cache.clear()
    logger.info("所有缓存已清空")


def get_cache_stats() -> Dict:
    memory_size = len(_explanation_cache.memory_cache)
    disk_size = 0
    if os.path.exists(_explanation_cache.cache_dir):
        disk_size = len(os.listdir(_explanation_cache.cache_dir))
    return {
        "memory_cache_size": memory_size,
        "disk_cache_size": disk_size,
        "model_cache_size": len(_global_model_cache),
        "cache_dir": _explanation_cache.cache_dir
    }
