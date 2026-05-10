import os
import tempfile
import torch
import torch_geometric.loader as geom_loader
from hydra import compose, initialize
from src.models import AMRNodeDualGNN
from src.utils import create_graph_dataset
import glob

def get_prediction(feature_path: str, antibiotic: str, isolate_ids: list = None) -> dict:
    """
    根据特征文件夹和抗生素名称进行预测（供 FastAPI 调用）

    Args:
        feature_path: 预处理后的特征文件夹路径（可能为相对路径，如 './data/extracted_unitigs'）
        antibiotic: 抗生素名称，如 "vancomycin"
        isolate_ids: 可选，指定要预测的菌株ID列表

    Returns:
        dict: {"y_proba": [...], "y_pred": [...], "isolate_ids": [...]}
    """
    # ========== 新增：将相对路径转换为绝对路径（基于当前文件所在的项目根目录） ==========
    if feature_path.startswith('./'):
        # 获取当前文件（即该脚本）所在目录，假设脚本放在项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        feature_path = os.path.join(current_dir, feature_path.lstrip('./'))
    elif not os.path.isabs(feature_path):
        # 如果是其他形式的相对路径，可类似处理
        current_dir = os.path.dirname(os.path.abspath(__file__))
        feature_path = os.path.join(current_dir, feature_path)
    # =================================================================================

    # 1. 构建临时配置
    overrides = [
        f"data.input_dir={feature_path}",
        f"data.antimicrobial={antibiotic}",
        "data.labels=data/ast_labels.csv",
        "data.whole_ids=data/whole.ids",
        "adj_matrix.file_path_1=data/fcgr_adj_matrix.csv",
        "adj_matrix.file_path_2=data/snps_adj_matrix.csv",
        "trainer.model_checkpoint.dirpath=experiments/checkpoints",
    ]

    temp_ids_file = None
    if isolate_ids:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ids', delete=False) as f:
            f.write('\n'.join(isolate_ids))
            temp_ids_file = f.name
        overrides.append(f"data.predict_ids={temp_ids_file}")

    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name="config", overrides=overrides)

        dataset, adj_mat_1, adj_mat_2 = create_graph_dataset(cfg)
        loader = geom_loader.DataLoader([dataset], batch_size=1)

        checkpoint_dir = cfg.trainer.model_checkpoint.dirpath
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        checkpoint = torch.load(ckpt_files[0], map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('loss_fn')}

        model = AMRNodeDualGNN(
            cfg=cfg,
            c_in=dataset.x.size(-1),
            class_weight=None
        )
        model.load_state_dict(filtered_dict, strict=False)
        model.eval()

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(model.device)
                output = model(batch)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
                if logits.numel() == 0:
                    continue
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        if all_probs:
            y_proba = np.concatenate(all_probs, axis=0).tolist()
            y_pred = np.concatenate(all_preds, axis=0).tolist()
        else:
            y_proba = []
            y_pred = []

        return {
            "y_proba": y_proba,
            "y_pred": y_pred,
            "isolate_ids": []
        }
    finally:
        if temp_ids_file and os.path.exists(temp_ids_file):
            os.unlink(temp_ids_file)