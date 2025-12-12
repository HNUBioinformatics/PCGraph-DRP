#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict, Any, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ESM2_FEATURES_PATH = r"../PCGraph-DRP/pathway_network/pathway_ESM2_features_scaled.csv"
PATHWAY_FILE_PATH = r"../PCGraph-DRP/data/raw/filtered_pathways_714.csv"
OUTPUT_PATH = r"../PCGraph-DRP/pathway_network/pathway_node_features.csv"

MIN_VARIANCE = 0.9
MIN_DIM = 128
MAX_DIM = 256
RANDOM_STATE = 42


def load_pathway_gene_mapping(pathway_file: str) -> Dict[str, Set[str]]:
    try:
        logger.info(f"Loading pathway-gene mapping: {pathway_file}")
        df = pd.read_csv(pathway_file)
        
        if 'genes' not in df.columns or 'name' not in df.columns:
            logger.warning(f"Pathway file format mismatch, trying first two columns: {df.columns[:2].tolist()}")
            gene_col = df.columns[0]
            name_col = df.columns[1]
        else:
            gene_col = 'genes'
            name_col = 'name'
        
        pathway_to_genes = {}
        for _, row in df.iterrows():
            pathway_name = str(row[name_col]).strip()
            if pd.isna(row[gene_col]):
                continue
                
            gene_list = str(row[gene_col]).split('|')
            genes = {g.strip() for g in gene_list if g.strip()}
            
            if genes:
                pathway_to_genes[pathway_name] = genes
        
        logger.info(f"Successfully loaded gene mapping for {len(pathway_to_genes)} pathways")
        
        gene_counts = [len(genes) for genes in pathway_to_genes.values()]
        logger.info(f"Pathway gene counts: min={min(gene_counts)}, max={max(gene_counts)}, average={np.mean(gene_counts):.1f}")
        
        return pathway_to_genes
    except Exception as e:
        logger.error(f"Failed to load pathway-gene mapping: {e}")
        raise


def load_esm2_gene_features(esm2_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading gene-level ESM2 features: {esm2_path}")
        esm2_features = pd.read_csv(esm2_path, index_col=0)
        
        esm2_features.index = esm2_features.index.astype(str)
        
        logger.info(f"ESM2 feature dimensions: {esm2_features.shape}, containing {len(esm2_features)} genes")
        
        esm2_cols = [col for col in esm2_features.columns if 'esm2_' in col or 'emb' in col]
        if not esm2_cols:
            logger.warning("No ESM2 feature columns found, trying to use all columns")
            esm2_cols = esm2_features.columns.tolist()
        
        logger.info(f"Number of ESM2 feature columns: {len(esm2_cols)}")
        
        return esm2_features
    except Exception as e:
        logger.error(f"Failed to load ESM2 features: {e}")
        raise


def aggregate_gene_features_to_pathway(
    pathway_to_genes: Dict[str, Set[str]], 
    gene_features: pd.DataFrame,
    aggregate_methods: List[str] = ['mean', 'max', 'min', 'std']
) -> Dict[str, pd.DataFrame]:
    try:
        logger.info(f"Starting to aggregate gene features to pathway features, using methods: {aggregate_methods}")
        
        feature_cols = gene_features.columns
        logger.info(f"Gene feature dimensions: {gene_features.shape}")
        
        total_genes = set(gene_features.index)
        matched_genes_count = 0
        total_pathway_genes = 0
        
        aggregated_features = {method: [] for method in aggregate_methods}
        pathway_names = []
        pathway_gene_counts = []
        matched_gene_counts = []
        
        for pathway_name, pathway_genes in pathway_to_genes.items():
            matched_genes = list(pathway_genes.intersection(total_genes))
            matched_gene_counts.append(len(matched_genes))
            total_pathway_genes += len(pathway_genes)
            
            if not matched_genes:
                for method in aggregate_methods:
                    aggregated_features[method].append(pd.Series([0.0] * len(feature_cols), index=feature_cols, name=pathway_name))
                pathway_names.append(pathway_name)
                pathway_gene_counts.append(len(pathway_genes))
                continue
                
            pathway_gene_features = gene_features.loc[matched_genes]
            
            for method in aggregate_methods:
                if method == 'mean':
                    agg_feat = pathway_gene_features.mean(axis=0)
                elif method == 'max':
                    agg_feat = pathway_gene_features.max(axis=0)
                elif method == 'min':
                    agg_feat = pathway_gene_features.min(axis=0)
                elif method == 'std':
                    if len(matched_genes) > 1:
                        agg_feat = pathway_gene_features.std(axis=0)
                    else:
                        agg_feat = pd.Series([0.0] * len(feature_cols), index=feature_cols)
                else:
                    raise ValueError(f"Unsupported aggregation method: {method}")
                
                agg_feat.name = pathway_name
                aggregated_features[method].append(agg_feat)
            
            pathway_names.append(pathway_name)
            pathway_gene_counts.append(len(pathway_genes))
        
        result = {}
        for method in aggregate_methods:
            df = pd.DataFrame(aggregated_features[method])
            df.index = pathway_names
            result[method] = df
        
        match_rate = matched_genes_count / total_pathway_genes * 100 if total_pathway_genes > 0 else 0
        logger.info(f"Total pathway genes: {total_pathway_genes}, matched genes: {matched_genes_count}, match rate: {match_rate:.2f}%")
        
        pathway_info = pd.DataFrame({
            'pathway': pathway_names,
            'total_genes': pathway_gene_counts,
            'matched_genes': matched_gene_counts,
            'match_rate': [count/total*100 if total > 0 else 0 for count, total in zip(matched_gene_counts, pathway_gene_counts)]
        })
        
        match_rate_bins = [0, 20, 40, 60, 80, 100]
        match_rate_counts = pd.cut(pathway_info['match_rate'], bins=match_rate_bins).value_counts().sort_index()
        logger.info(f"Pathway gene match rate distribution:")
        for i, count in enumerate(match_rate_counts):
            logger.info(f"  {match_rate_bins[i]}-{match_rate_bins[i+1]}%: {count} pathways")
        
        result['info'] = pathway_info
        return result
    
    except Exception as e:
        logger.error(f"Failed to aggregate gene features: {e}")
        raise


def combine_aggregated_features(aggregated_features: Dict[str, pd.DataFrame], 
                               methods: List[str] = ['mean', 'max', 'min', 'std']) -> pd.DataFrame:
    try:
        pathways = set(aggregated_features[methods[0]].index)
        for method in methods[1:]:
            if set(aggregated_features[method].index) != pathways:
                raise ValueError(f"Pathway set of aggregation method {method} is different from other methods")
        
        combined_df = pd.DataFrame(index=aggregated_features[methods[0]].index)
        
        for method in methods:
            df = aggregated_features[method]
            renamed_cols = {col: f"{method}_{col}" for col in df.columns}
            combined_df = pd.concat([combined_df, df.rename(columns=renamed_cols)], axis=1)
        
        logger.info(f"ESM2 feature dimensions after combining {len(methods)} aggregation methods: {combined_df.shape}")
        return combined_df
    
    except Exception as e:
        logger.error(f"Failed to combine aggregated features: {e}")
        raise


def analyze_pca_dimensions(data: np.ndarray, max_components: int = None) -> Dict[str, Any]:
    if max_components is None or max_components >= data.shape[1]:
        max_components = min(data.shape[1] - 1, 300)
    
    pca = PCA().fit(data)
    
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    dims_for_thresholds = {}
    
    for t in thresholds:
        try:
            dim = np.where(cum_var_ratio >= t)[0][0] + 1
            dims_for_thresholds[t] = dim
            logger.info(f"Dimensions required to retain {t*100:.0f}% variance: {dim}")
        except IndexError:
            dims_for_thresholds[t] = data.shape[1]
            logger.info(f"Even using all {data.shape[1]} dimensions cannot reach {t*100:.0f}% variance")
    
    preset_dims = [64, 128, 192, 256]
    preset_dims = [d for d in preset_dims if d < data.shape[1]]
    preset_var_ratio = {dim: float(cum_var_ratio[dim-1]) for dim in preset_dims}
    
    for dim, var in preset_var_ratio.items():
        logger.info(f"{dim}-dimensional PCA explains variance: {var*100:.2f}%")
    
    results = {
        "cum_var_ratio": cum_var_ratio,
        "dims_for_thresholds": dims_for_thresholds,
        "preset_var_ratio": preset_var_ratio
    }
    
    return results


def determine_optimal_dim(pca_analysis: Dict[str, Any], 
                         min_variance: float, 
                         min_dim: int = 64, 
                         max_dim: int = 256) -> int:
    dims = pca_analysis["dims_for_thresholds"]
    target_dim = dims.get(min_variance, None)
    
    if target_dim is None:
        closest_threshold = min(dims.keys(), key=lambda x: abs(x - min_variance))
        target_dim = dims[closest_threshold]
        logger.info(f"No exact dimension found to meet {min_variance*100:.0f}% variance, selecting dimension for closest {closest_threshold*100:.0f}% variance: {target_dim}")
    
    target_dim = max(min_dim, min(max_dim, target_dim))
    logger.info(f"Final selected dimension for dimensionality reduction: {target_dim}")
    
    return target_dim


def reduce_dimension(combined_features: pd.DataFrame, 
                    min_variance: float = 0.9, 
                    min_dim: int = 128,
                    max_dim: int = 256,
                    random_state: int = 42) -> Tuple[pd.DataFrame, int]:
    try:
        data = combined_features.fillna(0)
        
        logger.info(f"Analyzing PCA dimensionality reduction, feature dimensions: {data.shape}")
        pca_analysis = analyze_pca_dimensions(data.values)
        
        optimal_dim = determine_optimal_dim(
            pca_analysis, 
            min_variance=min_variance, 
            min_dim=min_dim,
            max_dim=max_dim
        )
        
        logger.info(f"Performing PCA dimensionality reduction to {optimal_dim} dimensions...")
        pca = PCA(n_components=optimal_dim, random_state=random_state)
        reduced_data = pca.fit_transform(data.values)
        
        reduced_df = pd.DataFrame(
            reduced_data,
            index=data.index,
            columns=[f'esm2_pca_{i}' for i in range(optimal_dim)]
        )
        
        final_explained_var = sum(pca.explained_variance_ratio_) * 100
        logger.info(f"Information retained after dimensionality reduction: {final_explained_var:.2f}%")
        
        return reduced_df, optimal_dim
    
    except Exception as e:
        logger.error(f"Failed to perform dimensionality reduction: {e}")
        raise


def save_features(combined_df: pd.DataFrame, output_path: str, meta_data: Dict[str, Any] = None, 
                 pathway_info: Optional[pd.DataFrame] = None) -> None:
    try:
        output_dir = os.path.dirname(os.path.abspath(output_path)) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        combined_df.to_csv(output_path)
        logger.info(f"ESM2 pathway features saved to: {output_path}")
        
        if meta_data:
            import json
            meta_path = os.path.join(output_dir, "esm2_feature_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            logger.info(f"Feature metadata saved to: {meta_path}")
        
        if pathway_info is not None:
            pathway_path = os.path.join(output_dir, "pathway_gene_match_info.csv")
            pathway_info.to_csv(pathway_path, index=False)
            logger.info(f"Pathway gene match information saved to: {pathway_path}")
    
    except Exception as e:
        logger.error(f"Failed to save features: {e}")
        raise


def main():
    try:
        logger.info(f"Starting ESM2 pathway feature extraction process...")
        logger.info(f"Pathway file: {PATHWAY_FILE_PATH}")
        logger.info(f"ESM2 gene feature file: {ESM2_FEATURES_PATH}")
        logger.info(f"Output file path: {OUTPUT_PATH}")
        logger.info(f"PCA target variance explanation rate: {MIN_VARIANCE}")
        logger.info(f"PCA dimension range: {MIN_DIM}-{MAX_DIM}")
        
        pathway_to_genes = load_pathway_gene_mapping(PATHWAY_FILE_PATH)
        
        gene_esm2_features = load_esm2_gene_features(ESM2_FEATURES_PATH)
        
        aggregation_methods = ['mean', 'max', 'min', 'std']
        aggregated_features = aggregate_gene_features_to_pathway(
            pathway_to_genes, 
            gene_esm2_features,
            aggregation_methods
        )
        
        combined_esm2_features = combine_aggregated_features(
            aggregated_features, 
            aggregation_methods
        )
        
        reduced_esm2_features, optimal_dim = reduce_dimension(
            combined_esm2_features,
            min_variance=MIN_VARIANCE,
            min_dim=MIN_DIM,
            max_dim=MAX_DIM,
            random_state=RANDOM_STATE
        )
        
        final_features = reduced_esm2_features
        logger.info(f"Final ESM2 pathway feature dimensions: {final_features.shape}")
        
        pathway_info = aggregated_features.get('info')
        meta_data = {
            "num_pathways": len(final_features),
            "esm2_original_dim": gene_esm2_features.shape[1],
            "esm2_aggregated_dim": combined_esm2_features.shape[1],
            "esm2_reduced_dim": optimal_dim,
            "total_features_dim": final_features.shape[1],
            "min_variance": MIN_VARIANCE,
            "aggregation_methods": aggregation_methods,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_features(
            final_features, 
            OUTPUT_PATH, 
            meta_data,
            pathway_info
        )
        
        logger.info("ESM2 pathway feature extraction completed!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())