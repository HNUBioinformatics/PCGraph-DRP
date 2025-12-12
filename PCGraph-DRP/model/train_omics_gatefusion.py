import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import warnings
from torch.serialization import safe_globals
from numpy.core.multiarray import scalar
warnings.filterwarnings('ignore')

from model_omics_gatefusion import HierarchicalGNNFeatureExtractor, PathwayCellGNN

# ============ 日志 ============
def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_log_file = output_dir / "drug_response_training.log"
    metrics_log_file = output_dir / "training_metrics.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(main_log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    metrics_logger = logging.getLogger('metrics')
    metrics_handler = logging.FileHandler(metrics_log_file, encoding='utf-8')
    metrics_formatter = logging.Formatter('%(asctime)s - %(message)s')
    metrics_handler.setFormatter(metrics_formatter)
    if not metrics_logger.handlers:
        metrics_logger.addHandler(metrics_handler)
        metrics_logger.addHandler(logging.StreamHandler())
    metrics_logger.setLevel(logging.INFO)
    return logging.getLogger(__name__), metrics_logger

# ============ 工具类 ============
class IC50Normalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    def fit(self, train_values):
        self.scaler.fit(np.array(train_values).reshape(-1, 1))
        self.is_fitted = True
        return self
    def transform(self, values):
        if not self.is_fitted: raise ValueError("Normalizer not fitted")
        return self.scaler.transform(np.array(values).reshape(-1, 1)).flatten()
    def inverse_transform(self, normalized_values):
        if not self.is_fitted: raise ValueError("Normalizer not fitted")
        return self.scaler.inverse_transform(np.array(normalized_values).reshape(-1, 1)).flatten()

# ============ 损失 ============
class RobustRegressionLoss(nn.Module):
    def __init__(self, delta=1.0, mse_weight=0.4, mae_weight=0.3, huber_weight=0.3):
        super().__init__()
        self.delta, self.mse_weight, self.mae_weight, self.huber_weight = delta, mse_weight, mae_weight, huber_weight
    def forward(self, p, t):
        return (self.mse_weight * F.mse_loss(p, t) +
                self.mae_weight * F.l1_loss(p, t) +
                self.huber_weight * F.smooth_l1_loss(p, t, beta=self.delta))

# ============ 读三组学 ============
def load_omics_csv(omics_path: str):
    df = pd.read_csv(omics_path)
    cell_col = df.columns[0]
    df[cell_col] = df[cell_col].astype(str)
    mut_cols = [c for c in df.columns if c.endswith('_mutation')]
    cnv_cols = [c for c in df.columns if c.endswith('_cnv')]
    rna_cols = [c for c in df.columns if c.endswith('_rna')]
    omics = {}
    for _, row in df.iterrows():
        name = str(row[cell_col])
        omics[name] = {
            'mut': row[mut_cols].values.astype(np.float32),
            'cnv': row[cnv_cols].values.astype(np.float32),
            'rna': row[rna_cols].values.astype(np.float32),
        }
    return omics, len(mut_cols), len(cnv_cols), len(rna_cols)

# ============ 数据集与组装 ============
class DrugResponseDataset(Dataset):
    def __init__(self, cell_extractor, drug_features_path, response_data_path,
                 omics_dict, scalers, valid_cell_lines=None,
                 ic50_normalizer=None, is_training=False):
        self.cell_extractor = cell_extractor
        self.ic50_normalizer = ic50_normalizer
        self.is_training = is_training
        self.graph_cache = {}
        self.drug_features_df = pd.read_csv(drug_features_path)
        self.response_df = pd.read_csv(response_data_path)
        self.drug_id_to_features = {
            row.iloc[0]: torch.from_numpy(row.iloc[1:].values.astype(np.float32))
            for _, row in self.drug_features_df.iterrows()
        }
        self.omics_dict = omics_dict
        self.scalers = scalers

        temp_samples, raw_responses = [], []
        for _, row in self.response_df.iterrows():
            cell_id = row.get('cell_line_id')
            cell_line_name = self.cell_extractor.cell_line_id_to_name.get(cell_id)
            drug_id = row['drug_id']
            response = float(row.get('IC50', row.get('response_value', np.nan)))
            if (cell_line_name in self.omics_dict) and (drug_id in self.drug_id_to_features) and np.isfinite(response):
                if valid_cell_lines and cell_line_name not in valid_cell_lines:
                    continue
                temp_samples.append({
                    'cell_line': cell_line_name,
                    'drug_id': drug_id,
                    'cell_line_id': cell_id,
                    'response': response
                })
                raw_responses.append(response)

        # IC50 标准化
        normalized_responses = raw_responses
        if self.ic50_normalizer and raw_responses:
            if self.is_training:
                self.ic50_normalizer.fit(raw_responses)
            normalized_responses = self.ic50_normalizer.transform(raw_responses) if self.ic50_normalizer.is_fitted else raw_responses

        self.samples = []
        for i, s in enumerate(temp_samples):
            od = self.omics_dict[s['cell_line']]
            x_mut = torch.from_numpy(od['mut'].astype(np.float32))  # mutation 通常不标准化
            x_cnv = torch.from_numpy(self.scalers['cnv'].transform(od['cnv'].reshape(1, -1)).astype(np.float32)).squeeze(0)
            x_rna = torch.from_numpy(self.scalers['rna'].transform(od['rna'].reshape(1, -1)).astype(np.float32)).squeeze(0)
            self.samples.append({
                'cell_line': s['cell_line'],
                'drug_id': s['drug_id'],
                'cell_line_id': s['cell_line_id'],
                'response': normalized_responses[i],
                'mut': x_mut, 'cnv': x_cnv, 'rna': x_rna
            })

        logging.info(f"Dataset initialized (is_training: {self.is_training}): {len(self.samples)} valid samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cname = s['cell_line']
        if cname not in self.graph_cache:
            self.graph_cache[cname] = self.cell_extractor.create_single_cell_graph(cname)
        return {
            'cell_graph': self.graph_cache[cname],
            'drug_features': self.drug_id_to_features[s['drug_id']],
            'mut': s['mut'], 'cnv': s['cnv'], 'rna': s['rna'],
            'response': torch.tensor([s['response']], dtype=torch.float32),
            'cell_line_id': s['cell_line_id'],
            'drug_id': s['drug_id']
        }

def collate_drug_response_batch(batch):
    return {
        'cell_graphs': Batch.from_data_list([item['cell_graph'] for item in batch]),
        'drug_features': torch.stack([item['drug_features'] for item in batch]),
        'mut': torch.stack([item['mut'] for item in batch]),
        'cnv': torch.stack([item['cnv'] for item in batch]),
        'rna': torch.stack([item['rna'] for item in batch]),
        'responses': torch.cat([item['response'] for item in batch]),
        'cell_line_ids': [item['cell_line_id'] for item in batch],
        'drug_ids': [item['drug_id'] for item in batch]
    }

# ============ 三组学编码 + 门控融合 ============
class OmicsEncoderFuser(nn.Module):
    def __init__(self, in_dim_mut=308, in_dim_cnv=308, in_dim_rna=308, H=128, p_drop=0.1):
        super().__init__()
        def mlp(din, dout):
            return nn.Sequential(
                nn.Linear(din, 512), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(512, dout), nn.ReLU()
            )
        self.mlp_mut = mlp(in_dim_mut, H)
        self.mlp_cnv = mlp(in_dim_cnv, H)
        self.mlp_rna = mlp(in_dim_rna, H)
        self.alpha_head = nn.Sequential(
            nn.Linear(3*H, H), nn.ReLU(),
            nn.Linear(H, 3)  # -> α_mut, α_cnv, α_rna
        )
        self.norm = nn.LayerNorm(H)

    def forward(self, x_mut, x_cnv, x_rna):
        h_mut = self.mlp_mut(x_mut)
        h_cnv = self.mlp_cnv(x_cnv)
        h_rna = self.mlp_rna(x_rna)
        logits = self.alpha_head(torch.cat([h_mut, h_cnv, h_rna], dim=1))
        alpha = torch.softmax(logits, dim=1)                      # [B,3]
        z = alpha[:,0:1]*h_mut + alpha[:,1:2]*h_cnv + alpha[:,2:3]*h_rna
        return self.norm(z), (h_mut, h_cnv, h_rna, alpha)

# ============ 模型（mid + late + Head） ============
class DrugResponsePredictor(nn.Module):
    def __init__(self, cell_gnn: nn.Module, drug_feature_dim: int,
                 H_cond=128, fusion_hidden_dims=(256, 128, 64), dropout=0.2,
                 fusion_type='gate'):
        super().__init__()
        self.cell_gnn = cell_gnn
        self.fusion_type = fusion_type
        cell_out_dim = getattr(cell_gnn, "output_dim", cell_gnn.hidden_dim)

        self.omics_fuser = OmicsEncoderFuser(H=H_cond)  # 308/308/308 -> H
        self.cell_norm = nn.LayerNorm(cell_out_dim)
        self.drug_norm = nn.LayerNorm(drug_feature_dim)

        # late：对齐 + 门控残差/加和/拼接
        if fusion_type in ('add', 'gate'):
            self.z_proj = nn.Linear(H_cond, cell_out_dim)
        else:
            self.z_proj = nn.Identity()

        if fusion_type == 'gate':
            self.fuse_gate = nn.Sequential(
                nn.Linear(cell_out_dim + cell_out_dim, cell_out_dim), nn.ReLU(),
                nn.Linear(cell_out_dim, cell_out_dim), nn.Sigmoid()
            )
            fused_cell_dim = cell_out_dim
        elif fusion_type == 'add':
            self.fuse_gate = nn.Identity()
            fused_cell_dim = cell_out_dim
        elif fusion_type == 'concat':
            fused_cell_dim = cell_out_dim + H_cond
        else:
            raise ValueError("fusion_type must be 'gate' | 'add' | 'concat'")

        layers, prev_dim = [], fused_cell_dim + drug_feature_dim
        for hidden_dim in fusion_hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, batched_cell_graphs, drug_features, x_mut, x_cnv, x_rna):
        # 1) 三组学编码 + 门控融合 -> z_cond
        z_cond, _ = self.omics_fuser(x_mut, x_cnv, x_rna)

        # 2) mid-fusion：在 Pathway→Cell 中用 z_cond 调制
        # 需要你的 PathwayCellGNN.forward(data, omics_cond) 支持这个参数（见我之前给的文件）
        h_cell = self.cell_gnn(batched_cell_graphs, z_cond)
        h_cell = self.cell_norm(h_cell)

        # 3) late-fusion：门控残差/加和/拼接
        if self.fusion_type == 'concat':
            fused_cell = torch.cat([h_cell, z_cond], dim=1)
        else:
            z_hat = self.z_proj(z_cond)
            if self.fusion_type == 'gate':
                g = self.fuse_gate(torch.cat([h_cell, z_hat], dim=1))
                fused_cell = h_cell + g * z_hat
            else:  # add
                fused_cell = h_cell + z_hat

        # 4) 拼药物特征 → 预测 IC50
        drug_p = self.drug_norm(drug_features)
        return self.head(torch.cat([fused_cell, drug_p], dim=1)).squeeze(-1)

# ============ 指标 ============
def safe_correlation(x, y):
    return 0.0 if np.std(x) == 0 or np.std(y) == 0 else np.nan_to_num(np.corrcoef(x, y)[0, 1])
def safe_spearman(x, y):
    return (0.0, 1.0) if np.std(x) == 0 or np.std(y) == 0 else np.nan_to_num(spearmanr(x, y))

# ============ 训练/验证/测试 ============
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Training'):
        optimizer.zero_grad()
        predictions = model(
            batch['cell_graphs'].to(device),
            batch['drug_features'].to(device),
            batch['mut'].to(device), batch['cnv'].to(device), batch['rna'].to(device)
        )
        loss = criterion(predictions, batch['responses'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def validate(model, loader, criterion, device, normalizer=None):
    model.eval()
    val_loss = 0
    preds, targs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            p = model(
                batch['cell_graphs'].to(device),
                batch['drug_features'].to(device),
                batch['mut'].to(device), batch['cnv'].to(device), batch['rna'].to(device)
            )
            t = batch['responses'].to(device)
            val_loss += criterion(p, t).item()
            preds.extend(p.cpu().numpy())
            targs.extend(t.cpu().numpy())
    mp = normalizer.inverse_transform(preds) if (normalizer and normalizer.is_fitted) else preds
    mt = normalizer.inverse_transform(targs) if (normalizer and normalizer.is_fitted) else targs
    return {
        'loss': val_loss / len(loader) if len(loader) > 0 else 0,
        'r2': r2_score(mt, mp),
        'mse': mean_squared_error(mt, mp),
        'rmse': np.sqrt(mean_squared_error(mt, mp)),
        'mae': mean_absolute_error(mt, mp),
        'pearson': safe_correlation(mt, mp),
        'spearman': safe_spearman(mt, mp)[0]
    }

def evaluate(model, loader, device, normalizer=None):
    model.eval()
    preds, targs, cell_ids, drug_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            p = model(
                batch['cell_graphs'].to(device),
                batch['drug_features'].to(device),
                batch['mut'].to(device), batch['cnv'].to(device), batch['rna'].to(device)
            )
            t = batch['responses']
            preds.extend(p.cpu().numpy())
            targs.extend(t.numpy())
            cell_ids.extend(batch['cell_line_ids'])
            drug_ids.extend(batch['drug_ids'])
    final_preds = normalizer.inverse_transform(preds) if (normalizer and normalizer.is_fitted) else preds
    final_targs = normalizer.inverse_transform(targs) if (normalizer and normalizer.is_fitted) else targs
    metrics = {
        'r2': r2_score(final_targs, final_preds),
        'mse': mean_squared_error(final_targs, final_preds),
        'rmse': np.sqrt(mean_squared_error(final_targs, final_preds)),
        'mae': mean_absolute_error(final_targs, final_preds),
        'pearson': safe_correlation(final_targs, final_preds),
        'spearman': safe_spearman(final_targs, final_preds)[0]
    }
    results_df = pd.DataFrame({
        'cell_line_id': cell_ids,
        'drug_id': drug_ids,
        'predicted_ic50': final_preds,
        'actual_ic50': final_targs
    })
    return metrics, results_df

# ============ 训练主流程 ============
def train_end_to_end_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    logger, metrics_logger = setup_logging(output_dir)
    logger.info(f"Using device: {device}")

    # 初始化细胞特征提取器
    logger.info("Initializing cell line feature extractor...")
    cell_extractor = HierarchicalGNNFeatureExtractor(output_dir=str(output_dir))
    cell_extractor.load_all_data(
        pathway_activity_path=args.pathway_activity_path,
        pathway_jaccard_path=args.pathway_jaccard_path,
        pathway_static_features_path=args.pathway_static_features_path,
        cell_line_id_map_path=args.cell_line_id_map_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
    )

    # 读取三组学
    omics_dict, d_mut, d_cnv, d_rna = load_omics_csv(args.omics_path)
    assert d_mut == 308 and d_cnv == 308 and d_rna == 308, f"Expect 308 per omics, got {d_mut},{d_cnv},{d_rna}"
    logger.info(f"Loaded omics for {len(omics_dict)} cell lines (mut/cnv/rna = 308 each)")

    # 拟合 scaler（只用训练细胞系）
    rna_scaler = StandardScaler()
    cnv_scaler = StandardScaler()
    train_cells = [c for c in cell_extractor.train_cell_lines if c in omics_dict]
    if len(train_cells) == 0:
        raise ValueError("No training cells found in omics_dict. Check cell line naming.")
    rna_scaler.fit(np.vstack([omics_dict[c]['rna'] for c in train_cells]))
    cnv_scaler.fit(np.vstack([omics_dict[c]['cnv'] for c in train_cells]))
    scalers = {'rna': rna_scaler, 'cnv': cnv_scaler}
    logger.info("Fitted RNA/CNV scalers on training cells")

    # 维度
    pathway_feat_dim = cell_extractor.pathway_static_features.shape[1]
    cellline_feat_dim = len(cell_extractor.pathways)
    drug_feature_dim = pd.read_csv(args.train_drug_features_path).shape[1] - 1

    # GNN（带 mid-fusion 接口）
    cell_gnn = PathwayCellGNN(
        pathway_feat_dim=pathway_feat_dim,
        cellline_feat_dim=cellline_feat_dim,
        hidden_dim=args.cell_hidden_dim,
        output_dim=args.cell_output_dim,
        use_unweighted_pathway_gcn=args.use_unweighted_pathway_gcn,
        omics_dim=128  # 与 z_cond 维度一致
    ).to(device)

    logger.info(f"Cell GNN output dim: {cell_gnn.output_dim}, Drug feature dim: {drug_feature_dim}")

    # 模型
    model = DrugResponsePredictor(
        cell_gnn=cell_gnn,
        drug_feature_dim=drug_feature_dim,
        H_cond=128,
        fusion_hidden_dims=args.fusion_hidden_dims,
        dropout=args.dropout,
        fusion_type=args.fusion_type  # 推荐 'gate'
    ).to(device)
    logger.info(f"Total model params: {sum(p.numel() for p in model.parameters()):,}, Fusion type: {args.fusion_type}")

    # 数据集
    ic50_normalizer = IC50Normalizer()
    train_dataset = DrugResponseDataset(
        cell_extractor=cell_extractor,
        drug_features_path=args.train_drug_features_path,
        response_data_path=args.train_data_path,
        omics_dict=omics_dict, scalers=scalers,
        valid_cell_lines=set(cell_extractor.train_cell_lines),
        ic50_normalizer=ic50_normalizer,
        is_training=True
    )
    val_dataset = DrugResponseDataset(
        cell_extractor=cell_extractor,
        drug_features_path=args.val_drug_features_path,
        response_data_path=args.val_data_path,
        omics_dict=omics_dict, scalers=scalers,
        valid_cell_lines=set(cell_extractor.val_cell_lines),
        ic50_normalizer=ic50_normalizer,
        is_training=False
    )
    test_dataset = DrugResponseDataset(
        cell_extractor=cell_extractor,
        drug_features_path=args.test_drug_features_path,
        response_data_path=args.test_data_path,
        omics_dict=omics_dict, scalers=scalers,
        valid_cell_lines=set(cell_extractor.test_cell_lines),
        ic50_normalizer=ic50_normalizer,
        is_training=False
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_drug_response_batch, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_drug_response_batch, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_drug_response_batch, num_workers=args.num_workers)

    # 优化器与训练
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    criterion = RobustRegressionLoss(args.huber_delta, args.mse_weight, args.mae_weight, args.huber_weight)

    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = output_dir / 'best_model_fusion.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device, ic50_normalizer)
        scheduler.step(val_metrics['loss'])

        metrics_logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"R²: {val_metrics['r2']:.4f} | RMSE: {val_metrics['rmse']:.4f} | MAE: {val_metrics['mae']:.4f} | "
            f"Pearson: {val_metrics['pearson']:.4f} | Spearman: {val_metrics['spearman']:.4f}"
        )

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'ic50_normalizer': ic50_normalizer,
                'scalers': scalers,  # 保存 RNA/CNV 标准化器
                'cell_gnn_config': {
                    'pathway_feat_dim': pathway_feat_dim,
                    'cellline_feat_dim': cellline_feat_dim,
                    'hidden_dim': args.cell_hidden_dim,
                    'output_dim': args.cell_output_dim,
                    'use_unweighted_pathway_gcn': args.use_unweighted_pathway_gcn,
                    'omics_dim': 128
                },
                'drug_feature_dim': drug_feature_dim,
                'fusion_type': args.fusion_type
            }, best_model_path)
            logger.info(f"Epoch {epoch}: Best model saved to {best_model_path}")
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # 加载最优模型测试
    logger.info("Loading best model for testing...")
    with safe_globals([IC50Normalizer, StandardScaler, scalar]):
        checkpoint = torch.load(best_model_path, weights_only=False)

    cell_gnn_loaded = PathwayCellGNN(**checkpoint['cell_gnn_config']).to(device)
    model_loaded = DrugResponsePredictor(
        cell_gnn=cell_gnn_loaded,
        drug_feature_dim=checkpoint['drug_feature_dim'],
        H_cond=128,
        fusion_hidden_dims=args.fusion_hidden_dims,
        dropout=args.dropout,
        fusion_type=checkpoint['fusion_type']
    ).to(device)
    model_loaded.load_state_dict(checkpoint['model_state_dict'])

    test_metrics, test_results_df = evaluate(model_loaded, test_loader, device, checkpoint['ic50_normalizer'])
    logger.info("--- Test Results ---")
    for metric, value in test_metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    test_results_df.to_csv(output_dir / 'test_predictions_fusion.csv', index=False)
    logger.info(f"Test predictions saved to {output_dir / 'test_predictions_fusion.csv'}")

    return model_loaded, test_metrics

# ============ 参数 ============
def parse_args():
    parser = argparse.ArgumentParser(description='Drug Response Prediction (Cell GNN + Omics + Drug)')

    # 路径参数
    parser.add_argument('--base_path', type=str, default="../PCGraph-DRP", help='Base data path')
    parser.add_argument('--output_dir', type=str, default="../PCGraph-DRP/results_fusion_omics_mid_late/", help='Output directory')
    parser.add_argument('--train_data_path', type=str, default="data/splits/train_data_random.csv")
    parser.add_argument('--val_data_path', type=str, default="data/splits/val_data_random.csv")
    parser.add_argument('--test_data_path', type=str, default="data/splits/test_data_random.csv")
    parser.add_argument('--train_drug_features_path', type=str, default="results/drug_features/train_drug_fused_features3.csv")
    parser.add_argument('--val_drug_features_path', type=str, default="results/drug_features/val_drug_fused_features3.csv")
    parser.add_argument('--test_drug_features_path', type=str, default="results/drug_features/test_drug_fused_features3.csv")
    parser.add_argument('--pathway_activity_path', type=str, default="pathway_network/pathway_activity_ssgsea_714.csv")
    parser.add_argument('--pathway_jaccard_path', type=str, default="pathway_network/pathway_jaccard_matrix.csv")
    parser.add_argument('--pathway_static_features_path', type=str, default="pathway_network/pathway_ESM2_features_scaled.csv")
    parser.add_argument('--cell_line_id_map_path', type=str, default="data/processed/cell_line_id_map.csv")
    parser.add_argument('--omics_path', type=str, default="data/raw/reduced_omics_data.csv",
                        help='三组学表路径（首列cell，后面 gene*_mutation/gene*_cnv/gene*_rna）')

    # 模型参数
    parser.add_argument('--cell_hidden_dim', type=int, default=128)
    parser.add_argument('--cell_output_dim', type=int, default=128)
    parser.add_argument('--use_unweighted_pathway_gcn', action='store_true')
    parser.add_argument('--fusion_hidden_dims', type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--fusion_type', type=str, default='gate', choices=['concat', 'add', 'gate'])

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--huber_delta', type=float, default=1.0)
    parser.add_argument('--mse_weight', type=float, default=0.4)
    parser.add_argument('--mae_weight', type=float, default=0.3)
    parser.add_argument('--huber_weight', type=float, default=0.3)

    args = parser.parse_args()
    # 路径补全
    if args.base_path:
        for arg in vars(args):
            if 'path' in arg and getattr(args, arg) and not os.path.isabs(getattr(args, arg)):
                setattr(args, arg, os.path.join(args.base_path, getattr(args, arg)))
    return args

# ============ 入口 ============
def main():
    args = parse_args()
    torch.manual_seed(3407)
    np.random.seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_end_to_end_model(args)

if __name__ == "__main__":
    main()