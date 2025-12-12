# HierarchicalGNNFeatureExtractor.py
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

logger = logging.getLogger(__name__)

# --------- Utils (segment ops) ---------
# These functions are kept as they are used by the remaining gate
def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-16) -> torch.Tensor:
    if src.numel() == 0 or index.numel() == 0:
        return torch.zeros_like(src)
    
    assert index.min().item() >= 0, "segment_softmax: index has negative entries."
    assert index.max().item() < num_segments, f"segment_softmax: index out of range, max={int(index.max())}, num_segments={num_segments}"
    seg_max = scatter(src, index, dim=0, dim_size=num_segments, reduce='max')
    src_stable = src - seg_max.index_select(0, index)
    exp = torch.exp(src_stable)
    seg_sum = scatter(exp, index, dim=0, dim_size=num_segments, reduce='sum')
    out = exp / (seg_sum.index_select(0, index) + eps)
    return out


def segment_standardize(src: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-6) -> torch.Tensor:
    if src.numel() == 0:
        return torch.zeros_like(src)
    seg_sum = scatter(src, index, dim=0, dim_size=num_segments, reduce='sum')
    seg_count = scatter(torch.ones_like(src), index, dim=0, dim_size=num_segments, reduce='sum').clamp_min(1.)
    seg_mean = seg_sum / seg_count
    diff = src - seg_mean.index_select(0, index)
    seg_var = scatter(diff * diff, index, dim=0, dim_size=num_segments, reduce='mean')
    seg_std = torch.sqrt(seg_var + eps)
    z = diff / (seg_std.index_select(0, index) + eps)
    return z


# --------- Feature Extractor (Simplified) ---------
class HierarchicalGNNFeatureExtractor:
    def __init__(self,
                 output_dir: str = "./HeteroConv_features/",
                 pp_min_threshold: float = 0.01):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pathway-pathway config
        self.pp_min_threshold = float(pp_min_threshold)

        # Scalers (only for pathway activity)
        self.ssgsea_scaler = StandardScaler()

        # Mappings and caches
        self.pathway_to_idx: Dict[str, int] = {}
        self.cell_line_name_to_id: Dict[str, int] = {}
        self.cell_line_id_to_name: Dict[int, str] = {}

        # Loaded data placeholders
        self.pathways: List[str] = []

        # Optional containers for PP structure
        self.pathway_jaccard: Optional[pd.DataFrame] = None
        self.pathway_pp_edges: Optional[pd.DataFrame] = None

        # Prepared PP tensors (reused across graphs)
        self._pp_edge_index: Optional[torch.Tensor] = None
        self._pp_edge_weight: Optional[torch.Tensor] = None

    # ---------- IO and preprocessing ----------
    def _validate_and_clean_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        logger.info(f"Validating dataset: {name}")
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found null values in {name}, filling with 0 or mean.")
            if name in ['pathway_activity']:
                df = df.fillna(0.0)
            elif name in ['pathway_static_features']:
                df = df.fillna(df.mean(numeric_only=True))
            else:
                df = df.fillna(0)
        return df

    def _load_pp_structure(self, pp_path: str):
        logger.info(f"Loading pathway PP structure from: {pp_path}")
        raw = pd.read_csv(pp_path)
        cols = {c.lower().strip() for c in raw.columns}
        if {'src_pathway', 'dst_pathway'}.issubset(cols):
            rename_map = {c: c.lower().strip() for c in raw.columns}
            raw = raw.rename(columns=rename_map)
            if 'weight' not in raw.columns:
                raw['weight'] = 1.0
            raw['src_pathway'] = raw['src_pathway'].astype(str).str.strip()
            raw['dst_pathway'] = raw['dst_pathway'].astype(str).str.strip()
            self.pathway_pp_edges = self._validate_and_clean_dataframe(raw, 'pathway_pp_edges')
            self.pathway_jaccard = None
        else:
            mat = pd.read_csv(pp_path, index_col=0)
            mat.index = mat.index.astype(str).str.strip()
            mat.columns = mat.columns.astype(str).str.strip()
            self.pathway_jaccard = self._validate_and_clean_dataframe(mat, 'pathway_jaccard')
            self.pathway_pp_edges = None

    def load_all_data(
            self,
            pathway_activity_path: str,
            pathway_jaccard_path: str,
            pathway_static_features_path: str,
            cell_line_id_map_path: str,
            train_data_path: str,
            val_data_path: str,
            test_data_path: str,
    ):
        logger.info("Loading all data files (Pathway & Cell Line only)...")

        # Load only necessary tables
        self.pathway_activity = pd.read_csv(pathway_activity_path, index_col=0)
        self.pathway_activity = self._validate_and_clean_dataframe(self.pathway_activity, 'pathway_activity')

        self._load_pp_structure(pathway_jaccard_path)

        self.pathway_static_features = pd.read_csv(pathway_static_features_path, index_col=0)
        self.pathway_static_features = self._validate_and_clean_dataframe(self.pathway_static_features,'pathway_static_features')

        self.cell_line_id_map = pd.read_csv(cell_line_id_map_path)
        self.train_data = pd.read_csv(train_data_path)
        self.val_data = pd.read_csv(val_data_path)
        self.test_data = pd.read_csv(test_data_path)

        # Build mappings and preprocess
        self._create_mappings()
        self._extract_data_splits()
        self._preprocess_data()

        logger.info(f"Data loaded: {len(self.pathways)} pathways")
        logger.info(
            f"Train: {len(self.train_cell_lines)}, Val: {len(self.val_cell_lines)}, Test: {len(self.test_cell_lines)}")

    def _create_mappings(self):
        self.pathways = list(self.pathway_static_features.index)
        self.pathway_to_idx = {p: i for i, p in enumerate(self.pathways)}

        self.cell_line_name_to_id = dict(zip(self.cell_line_id_map['cell_line'], self.cell_line_id_map.iloc[:, 1]))
        self.cell_line_id_to_name = dict(zip(self.cell_line_id_map.iloc[:, 1], self.cell_line_id_map['cell_line']))

    def _extract_data_splits(self):
        def get_cell_lines_from_ids(cell_ids):
            valid_names = [self.cell_line_id_to_name.get(cid) for cid in cell_ids.unique() if pd.notna(cid)]
            return [name for name in valid_names if name in self.pathway_activity.index]

        self.train_cell_lines = get_cell_lines_from_ids(self.train_data['cell_line_id'])
        self.val_cell_lines = get_cell_lines_from_ids(self.val_data['cell_line_id'])
        self.test_cell_lines = get_cell_lines_from_ids(self.test_data['cell_line_id'])

    def _prepare_pp_edges(self):
        rows, cols, wts = [], [], []
        if self.pathway_pp_edges is not None:
            for _, r in self.pathway_pp_edges.iterrows():
                if r['src_pathway'] in self.pathway_to_idx and r['dst_pathway'] in self.pathway_to_idx:
                    rows.append(self.pathway_to_idx[r['src_pathway']])
                    cols.append(self.pathway_to_idx[r['dst_pathway']])
                    wts.append(r.get('weight', 1.0))
        elif self.pathway_jaccard is not None:
            jdf = self.pathway_jaccard.reindex(index=self.pathways, columns=self.pathways).fillna(0.0)
            jacc = jdf.values
            np.fill_diagonal(jacc, 0.0)
            ii, jj = np.where(jacc > self.pp_min_threshold)
            rows, cols, wts = ii.tolist(), jj.tolist(), jacc[ii, jj].tolist()

        if rows:
            self._pp_edge_index = torch.tensor([rows, cols], dtype=torch.long)
            self._pp_edge_weight = torch.tensor(wts, dtype=torch.float32)
        else:
            self._pp_edge_index = torch.empty((2, 0), dtype=torch.long)
            self._pp_edge_weight = torch.empty((0,), dtype=torch.float32)

    def _preprocess_data(self):
        logger.info("Preprocessing data with split normalization (Pathway & Cell Line only)...")
        train_ssgsea = self.pathway_activity.reindex(self.train_cell_lines)[self.pathways].fillna(0.0).values
        self.ssgsea_scaler.fit(train_ssgsea)
        logger.info("ssGSEA feature normalizer fitted")
        self._prepare_pp_edges()

    # ---------- Graph building ----------
    def create_single_cell_graph(self, cell_line: str) -> HeteroData:
        data = HeteroData()
        try:
            # Pathway features (static) [P, Dp]
            pf = self.pathway_static_features.reindex(self.pathways).fillna(0.0).values.astype(np.float32)
            data['pathway'].x = torch.from_numpy(pf)

            # Cellline feature (ssGSEA normalized) [1, P]
            ssgsea = self.pathway_activity.loc[cell_line, self.pathways].fillna(0.0).values
            ssgsea = self.ssgsea_scaler.transform([ssgsea])[0].astype(np.float32)
            data['cellline'].x = torch.from_numpy(ssgsea).unsqueeze(0)

            # Edges with weights
            self._add_edges_to_graph(data, cell_line)

        except Exception as e:
            logger.error(f"Failed to create graph for {cell_line}: {e}")
            return self._create_minimal_graph()
        return data

    def _create_minimal_graph(self) -> HeteroData:
        data = HeteroData()
        num_pathways = len(self.pathways) if self.pathways else 50
        pathway_feat_dim = self.pathway_static_features.shape[1] if hasattr(self, 'pathway_static_features') else 32
        data['pathway'].x = torch.zeros(num_pathways, pathway_feat_dim)
        data['cellline'].x = torch.zeros(1, num_pathways)
        data['pathway', 'similar_to', 'pathway'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['pathway', 'activates', 'cellline'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['cellline', 'has_pathway', 'pathway'].edge_index = torch.empty((2, 0), dtype=torch.long)
        return data

    def _add_edges_to_graph(self, data: HeteroData, cell_line: str):
        # 1) Pathway<->Pathway edges
        data['pathway', 'similar_to', 'pathway'].edge_index = self._pp_edge_index
        data['pathway', 'similar_to', 'pathway'].edge_weight = self._pp_edge_weight

        # 2) Pathway->Cellline (ssGSEA scores)
        pc_ei = [[p_idx for p_idx in range(len(self.pathways))], [0] * len(self.pathways)]
        pc_w = self.pathway_activity.loc[cell_line, self.pathways].fillna(0.0).values
        data['pathway', 'activates', 'cellline'].edge_index = torch.tensor(pc_ei, dtype=torch.long)
        data['pathway', 'activates', 'cellline'].edge_weight = torch.tensor(pc_w, dtype=torch.float32)

        # 3) Reverse edges
        data['cellline', 'has_pathway', 'pathway'].edge_index = data[
            'pathway', 'activates', 'cellline'].edge_index.flip(0)
        data['cellline', 'has_pathway', 'pathway'].edge_weight = data['pathway', 'activates', 'cellline'].edge_weight


# --------- Gates and Encoders (Simplified) ---------
class PathwayToCellGate(nn.Module):
    def __init__(self, pathway_dim: int, cell_dim: int, output_dim: int):
        super().__init__()
        self.pathway_proj = nn.Linear(pathway_dim, output_dim)
        self.attention = nn.Sequential(
            nn.Linear(pathway_dim + cell_dim, output_dim), nn.Tanh(), nn.Linear(output_dim, 1)
        )
        self.gate = nn.Sequential(nn.Linear(pathway_dim, output_dim), nn.Sigmoid())
        self.alpha_prior = nn.Parameter(torch.tensor(0.1))

    def forward(self, pathway_h: torch.Tensor, cell_feat: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        num_cells = cell_feat.size(0)
        pathway_src, cell_dst = edge_index
        p_feat = pathway_h[pathway_src]
        c_feat = cell_feat[cell_dst]
        att = self.attention(torch.cat([p_feat, c_feat], dim=-1)).squeeze(-1)
        ew_norm = segment_standardize(edge_weight, cell_dst, num_segments=num_cells)
        att = att + self.alpha_prior * ew_norm
        weights = segment_softmax(att, cell_dst, num_segments=num_cells)
        gated = self.gate(p_feat) * self.pathway_proj(p_feat)
        messages = gated * weights.unsqueeze(-1)
        cell_h = scatter(messages, cell_dst, dim=0, dim_size=num_cells, reduce='sum')
        return cell_h


# --------- NEW Simplified GNN Model ---------
class PathwayCellGNN(nn.Module):

    def __init__(self,
                 pathway_feat_dim: int,
                 cellline_feat_dim: int,
                 hidden_dim: int = 256,
                 output_dim: int = 256,
                 use_unweighted_pathway_gcn: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_unweighted_pathway_gcn = use_unweighted_pathway_gcn

        # Pathway feature projection layer
        self.pathway_proj = nn.Linear(pathway_feat_dim, hidden_dim) if pathway_feat_dim != hidden_dim else nn.Identity()
        self.pathway_norm = nn.LayerNorm(hidden_dim)

        # GCN for pathways
        self.pathway_conv = GCNConv(hidden_dim, hidden_dim)

        # Gate for pathway -> cell aggregation
        self.pathway_to_cell_gate = PathwayToCellGate(hidden_dim, cellline_feat_dim, output_dim)

    def forward(self, data: HeteroData) -> torch.Tensor:
        # 1) Initial pathway features from static data (e.g., ESM2)
        pathway_h_init = self.pathway_proj(data['pathway'].x)

        # 2) Pathway GCN
        edge_index = data['pathway', 'similar_to', 'pathway'].edge_index
        if edge_index.size(1) > 0:
            edge_weight = data[
                'pathway', 'similar_to', 'pathway'].edge_weight if not self.use_unweighted_pathway_gcn else None
            pathway_h_conv = F.relu(self.pathway_conv(pathway_h_init, edge_index, edge_weight=edge_weight))
            pathway_h = self.pathway_norm(pathway_h_init + pathway_h_conv)
        else:
            pathway_h = self.pathway_norm(pathway_h_init)

        # 3) Pathway -> Cell aggregation
        cell_features = self.pathway_to_cell_gate(
            pathway_h,
            data['cellline'].x,
            data['pathway', 'activates', 'cellline'].edge_index,
            data['pathway', 'activates', 'cellline'].edge_weight,
        )

        # Ensure correct output shape
        num_cells = data['cellline'].x.size(0)
        if cell_features.size(0) != num_cells:
            raise ValueError(f"Output embedding count {cell_features.size(0)} does not match cell count {num_cells}")

        return cell_features
