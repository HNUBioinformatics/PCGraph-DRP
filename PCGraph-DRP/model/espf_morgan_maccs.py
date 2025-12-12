import os
import argparse
import numpy as np
import pandas as pd
import pickle
import codecs
import re
import sys

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
except ImportError:
    print("Warning: RDKit library is not installed or found, some functions may be limited. Please install it with 'pip install rdkit-pypi'.")
    class Chem:
        @staticmethod
        def MolFromSmiles(smiles): return None if smiles == "" else "dummy"
    class AllChem:
        @staticmethod
        def GetMorganFingerprintAsBitVect(*args, **kwargs): return [0]*2048
    class MACCSkeys:
        @staticmethod
        def GenMACCSKeys(mol): return [0]*166

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("Warning: scikit-learn library is not installed or found. Please install it with 'pip install scikit-learn'.")
    class PCA:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, data): return data
        def transform(self, data): return data

OUTPUT_DIR = "../PCGraph-DRP/results"
SPLITS_DIR = "../PCGraph-DRP/data/splits"
SMILES_FILE = "../PCGraph-DRP/data/raw/cleaned_drugs_smiles.csv"
DRUG_ID_MAP_FILE = "../PCGraph-DRP/data/processed/drug_id_map.csv"
ESPF_VOCAB_PATH = '../PCGraph-DRP/data/raw/drug_codes_chembl_freq_1500.txt'
ESPF_SUBWORD_MAP = '../PCGraph-DRP/data/raw/subword_units_map_chembl_freq_1500.csv'
SPLIT_METHOD = 'random'

np.random.seed(3407)
try:
    import torch
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
        torch.backends.cudnn.deterministic = True
except ImportError:
    pass

def normalize_drug_name(name):
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = re.sub(r'[^\w\-]', '', name)
    name = re.sub(r'\s+', '', name)
    return name

def load_drug_data_by_cell_split():
    suffix = "random" if SPLIT_METHOD == "random" else "cell_based"
    print(f"Loading data using {SPLIT_METHOD} split method...")
    
    datasets = {}
    for name in ['train', 'val', 'test']:
        file_path = os.path.join(SPLITS_DIR, f'{name}_data_{suffix}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data split file not found: {file_path}")
        datasets[name] = pd.read_csv(file_path)
    print(f"Data volume - Train:{len(datasets['train'])}, Validation:{len(datasets['val'])}, Test:{len(datasets['test'])}")
    
    print(f"\n=== Loading SMILES file: {SMILES_FILE} ===")
    smiles_df = pd.read_csv(SMILES_FILE)
    print(f"SMILES file original column names: {smiles_df.columns.tolist()}")
    required_smiles_cols = ['drug_name', 'smiles']
    if not all(col in smiles_df.columns for col in required_smiles_cols):
        raise ValueError(f"SMILES file missing required columns! Need {required_smiles_cols}, actual columns are {smiles_df.columns.tolist()}")
    smiles_df['drug_name_norm'] = smiles_df['drug_name'].apply(normalize_drug_name)
    print(f"Total drugs in SMILES file: {len(smiles_df)}")
    print(f"Number of non-empty SMILES in SMILES file: {smiles_df[smiles_df['smiles'].notna() & (smiles_df['smiles'] != '')].shape[0]}")

    print(f"\n=== Loading drug ID mapping file: {DRUG_ID_MAP_FILE} ===")
    drug_id_map_df = pd.read_csv(DRUG_ID_MAP_FILE)
    print(f"Drug ID mapping file original column names: {drug_id_map_df.columns.tolist()}")
    required_id_cols = ['drug', 'id']
    if not all(col in drug_id_map_df.columns for col in required_id_cols):
        raise ValueError(f"Drug ID mapping file missing required columns! Need {required_id_cols}, actual columns are {drug_id_map_df.columns.tolist()}")
    drug_id_map_df.rename(columns={required_id_cols[0]: 'drug_name', required_id_cols[1]: 'drug_id'}, inplace=True)
    drug_id_map_df['drug_name_norm'] = drug_id_map_df['drug_name'].apply(normalize_drug_name)
    print(f"Total drugs in drug ID mapping file: {len(drug_id_map_df)}")

    print(f"\n=== Merging SMILES and drug ID mapping ===")
    merged_df = pd.merge(
        drug_id_map_df[['drug_id', 'drug_name', 'drug_name_norm']],
        smiles_df[['drug_name', 'drug_name_norm', 'smiles']],
        on='drug_name_norm',
        how='left',
        suffixes=('_idmap', '_smiles')
    )
    print(f"Total drugs after merging: {len(merged_df)}")
    merged_df['smiles_valid'] = merged_df['smiles'].apply(
        lambda x: isinstance(x, str) and x != '' and Chem.MolFromSmiles(x) is not None
    )
    valid_matched = merged_df[merged_df['smiles_valid']]
    print(f"Number of drugs with successful matching and valid SMILES: {len(valid_matched)}")
    
    drug_id_to_smiles = {}
    for _, row in valid_matched.iterrows():
        drug_id = str(row['drug_id'])
        drug_id_to_smiles[drug_id] = row['smiles']
    print(f"Number of drug_id_to_smiles mappings constructed: {len(drug_id_to_smiles)}")

    id_to_drug = {str(row['drug_id']): row['drug_name'] for _, row in drug_id_map_df.iterrows()}
    result = {}
    for name, data in datasets.items():
        unique_drug_ids = data['drug_id'].unique().astype(str)
        print(f"\nNumber of unique drug IDs in {name} dataset: {len(unique_drug_ids)}")
        smiles_list = [drug_id_to_smiles.get(drug_id, "") for drug_id in unique_drug_ids]
        valid_smiles_count = sum(1 for s in smiles_list if s != "" and Chem.MolFromSmiles(s) is not None)
        print(f"Number of valid SMILES in {name} dataset: {valid_smiles_count}")
        
        result[name] = pd.DataFrame({
            'drug_id': unique_drug_ids,
            'drug_name': [id_to_drug.get(drug_id, f"Unknown_{drug_id}") for drug_id in unique_drug_ids],
            'smiles': smiles_list,
            'smiles_valid': [s != "" and Chem.MolFromSmiles(s) is not None for s in smiles_list]
        })

    all_drugs = pd.concat(result.values()).drop_duplicates(subset=['drug_id'])
    result['all'] = all_drugs
    result['has_valid_smiles'] = len(drug_id_to_smiles) > 0
    print(f"\nTotal number of unique drugs: {len(all_drugs)}")
    print(f"Number of valid SMILES among all drugs: {all_drugs['smiles_valid'].sum()}")
    
    return result

def extract_morgan_features(drug_data, output_path, radius=2, nBits=2048):
    valid_smiles_count = drug_data['smiles_valid'].sum()
    use_random = valid_smiles_count == 0
    if use_random:
        print("Warning: No valid SMILES found, will create random Morgan fingerprints")
    else:
        print(f"Extracting Morgan fingerprints using valid SMILES, total {valid_smiles_count} drugs")
    
    fingerprints = []
    valid_drug_ids = []
    failed_drugs = []
    
    for _, row in drug_data.iterrows():
        drug_id = row['drug_id']
        try:
            if not row['smiles_valid']:
                seed = hash(str(drug_id)) % 1000000
                np.random.seed(seed)
                fp = np.random.randint(0, 2, nBits)
            else:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol is None:
                    failed_drugs.append(drug_id)
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            
            fingerprints.append(list(fp))
            valid_drug_ids.append(drug_id)
        except Exception as e:
            print(f"Error processing drug ID {drug_id}: {e}")
            failed_drugs.append(drug_id)
    
    if not fingerprints:
        raise ValueError("Failed to generate any Morgan fingerprints!")
    
    fingerprints_array = np.array(fingerprints)
    result_df = pd.DataFrame({f'Morgan_bit_{i+1}': fingerprints_array[:, i] for i in range(nBits)})
    result_df.insert(0, 'drug_id', valid_drug_ids)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Morgan fingerprints saved to: {output_path}")
    print(f"Successfully processed {len(valid_drug_ids)} drugs, failed {len(failed_drugs)} drugs")
    return result_df

def extract_maccs_features(drug_data, output_path):
    valid_smiles_count = drug_data['smiles_valid'].sum()
    use_random = valid_smiles_count == 0
    if use_random:
        print("Warning: No valid SMILES found, will create random MACCS fingerprints")
    else:
        print(f"Extracting MACCS fingerprints using valid SMILES, total {valid_smiles_count} drugs")
    
    fingerprints = []
    valid_drug_ids = []
    failed_drugs = []
    
    for _, row in drug_data.iterrows():
        drug_id = row['drug_id']
        try:
            if not row['smiles_valid']:
                seed = hash(str(drug_id)) % 1000000
                np.random.seed(seed)
                fp = np.random.randint(0, 2, 166)
            else:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol is None:
                    failed_drugs.append(drug_id)
                    continue
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                fp = np.array(maccs_fp)
            
            fingerprints.append(fp)
            valid_drug_ids.append(drug_id)
        except Exception as e:
            print(f"Error processing drug ID {drug_id}: {e}")
            failed_drugs.append(drug_id)
    
    if not fingerprints:
        raise ValueError("Failed to generate any MACCS fingerprints!")
    
    fingerprints_array = np.array(fingerprints)
    result_df = pd.DataFrame({f'MACCS_bit_{i+1}': fingerprints_array[:, i] for i in range(166)})
    result_df.insert(0, 'drug_id', valid_drug_ids)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    if failed_drugs:
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        base_dir = os.path.dirname(output_path)
        with open(os.path.join(base_dir, f"{base_name}_failed_drugs.pkl"), 'wb') as f:
            pickle.dump(failed_drugs, f)
    
    print(f"MACCS fingerprints saved to: {output_path}")
    print(f"Successfully processed {len(valid_drug_ids)} drugs, failed {len(failed_drugs)} drugs")
    return result_df

def extract_espf_features(drug_data, output_path, vocab_path, subword_map_path):
    try:
        from subword_nmt.apply_bpe import BPE
    except ImportError:
        print("Error: 'subword_nmt' package not found. Please run 'pip install subword-nmt'.")
        return None
    
    if not all(os.path.exists(path) for path in [vocab_path, subword_map_path]):
        print(f"ESPF files do not exist: {vocab_path} or {subword_map_path}")
        return None
    
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv(subword_map_path)
    words2idx_d = dict(zip(sub_csv['index'].values, range(len(sub_csv))))
    
    drug_espf = {}
    false_espf = []
    
    for _, row in drug_data.iterrows():
        if not row['smiles_valid']:
            false_espf.append(row['drug_id'])
            continue
        
        try:
            tokens = dbpe.process_line(row['smiles']).split()
            indices = np.asarray([words2idx_d[token] for token in tokens])
            vector = np.zeros(len(words2idx_d))
            vector[indices] = 1
            drug_espf[row['drug_id']] = vector
        except Exception as e:
            print(f"ESPF encoding failed for drug ID {row['drug_id']}: {e}")
            false_espf.append(row['drug_id'])
    
    if not drug_espf:
        print("Warning: No ESPF features encoded successfully")
        return None
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    base_dir = os.path.dirname(output_path)
    
    with open(output_path, 'wb') as f:
        pickle.dump(drug_espf, f)
    with open(os.path.join(base_dir, f"{base_name}_false_encodings.pkl"), 'wb') as f:
        pickle.dump(false_espf, f)
    valid_drug_ids = list(drug_espf.keys())
    features_array = np.array([drug_espf[drug_id] for drug_id in valid_drug_ids])
    result_df = pd.DataFrame(features_array, columns=[f"espf_feature_{i}" for i in range(features_array.shape[1])])
    result_df.insert(0, "drug_id", valid_drug_ids)
    csv_output = os.path.join(base_dir, f"{base_name}.csv")
    result_df.to_csv(csv_output, index=False)
    
    print(f"ESPF features saved to: {csv_output}")
    print(f"Successfully encoded {len(valid_drug_ids)} drugs, failed {len(false_espf)} drugs")
    return result_df

def concatenate_and_pca_fusion(features_dict, fusion_dim=256, is_train=False, pca_model=None):
    common_ids = set(features_dict[list(features_dict.keys())[0]].index)
    for k in features_dict:
        common_ids.intersection_update(features_dict[k].index)
    
    common_ids = sorted(list(common_ids))
    if not common_ids:
        print("Warning: No common drug IDs found for feature fusion")
        return np.array([]), pca_model, common_ids
    print(f"Found {len(common_ids)} common drugs for fusion")

    aligned_features = []
    for k in features_dict:
        aligned_df = features_dict[k].loc[common_ids]
        aligned_features.append(aligned_df.values)

    concatenated_features = np.hstack(aligned_features)
    feature_names = list(features_dict.keys())
    print(f"Concatenated feature types: {feature_names}")
    print(f"Dimensions of each feature: {[f.shape[1] for f in aligned_features]}")
    print(f"Total dimension after concatenation: {concatenated_features.shape[1]}")
    
    n_samples, n_features = concatenated_features.shape
    actual_fusion_dim = min(fusion_dim, min(n_samples, n_features))
    
    if actual_fusion_dim != fusion_dim:
        print(f"Warning: PCA target dimension adjusted from {fusion_dim} to {actual_fusion_dim} (limited by sample/feature count)")
    
    if is_train:
        if actual_fusion_dim >= n_features:
            print(f"Feature dimension({n_features}) ≤ target dimension({actual_fusion_dim}), skipping PCA")
            return concatenated_features, None, common_ids
        
        print("Training PCA model and reducing dimension...")
        pca = PCA(n_components=actual_fusion_dim)
        reduced_features = pca.fit_transform(concatenated_features)
        print(f"PCA dimension reduction completed: {n_features} → {reduced_features.shape[1]}")
        print(f"PCA cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        return reduced_features, pca, common_ids
    else:
        if pca_model is None:
            print("Warning: No pre-trained PCA model, returning original concatenated features")
            return concatenated_features, None, common_ids
        reduced_features = pca_model.transform(concatenated_features)
        print(f"Reducing dimension using pre-trained PCA model: {n_features} → {reduced_features.shape[1]}")
        return reduced_features, pca_model, common_ids

def process_features_by_dataset_split(output_dir, 
                                      morgan_radius, 
                                      morgan_nbits,
                                      espf_vocab_path,
                                      espf_subword_map,
                                      fusion_dim,
                                      skip_espf,
                                      skip_maccs,
                                      feature_types,
                                      **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    
    valid_feature_types = ['morgan', 'maccs', 'espf']
    feature_types = [ft for ft in feature_types if ft in valid_feature_types]
    if not feature_types:
        feature_types = ['morgan', 'maccs']
    print(f"Enabled feature types: {feature_types}")
    
    drug_data = load_drug_data_by_cell_split()
    has_valid_smiles = drug_data.get('has_valid_smiles', False)
    espf_files_exist = all(os.path.exists(path) for path in [espf_vocab_path, espf_subword_map])
    
    pca_model = None
    if os.path.exists(os.path.join(output_dir, "pca_model3.pkl")) and not has_valid_smiles:
        with open(os.path.join(output_dir, "pca_model3.pkl"), 'rb') as f:
            pca_model = pickle.load(f)

    for dataset_name in ['train', 'val', 'test']:
        print(f"\n" + "="*50)
        print(f"=== Processing {dataset_name} dataset ===")
        print("="*50)
        dataset_drugs = drug_data[dataset_name]
        if len(dataset_drugs) == 0:
            print(f"Warning: No drug data in {dataset_name} dataset, skipping")
            continue
        
        features_dfs = {}
        
        if 'morgan' in feature_types:
            print(f"\n1. Extracting Morgan fingerprints...")
            path = os.path.join(output_dir, f"{dataset_name}_drug_morgan_features3.csv")
            df = extract_morgan_features(dataset_drugs, path, radius=morgan_radius, nBits=morgan_nbits)
            if df is not None:
                features_dfs['morgan'] = df.set_index('drug_id')
        
        if 'maccs' in feature_types and not skip_maccs:
            print(f"\n2. Extracting MACCS fingerprints...")
            path = os.path.join(output_dir, f"{dataset_name}_drug_maccs_features3.csv")
            df = extract_maccs_features(dataset_drugs, path)
            if df is not None:
                features_dfs['maccs'] = df.set_index('drug_id')
        
        if 'espf' in feature_types and not skip_espf and has_valid_smiles and espf_files_exist:
            print(f"\n3. Extracting ESPF features...")
            path = os.path.join(output_dir, f"{dataset_name}_drug_espf_encoding3.pkl")
            df = extract_espf_features(dataset_drugs, path, espf_vocab_path, espf_subword_map)
            if df is not None:
                features_dfs['espf'] = df.set_index('drug_id')
        
        if len(features_dfs) > 0:
            print(f"\n4. Feature fusion (total {len(features_dfs)} feature types)...")
            if len(features_dfs) == 1:
                feature_type, feature_df = list(features_dfs.items())[0]
                print(f"Only one feature type({feature_type}), using it directly as fusion result")
                fused_df = feature_df.reset_index()
                fused_df.columns = ['drug_id'] + [f'fused_feature_{i}' for i in range(1, len(fused_df.columns))]
            else:
                is_train = (dataset_name == 'train')
                fused_features, pca_model, common_ids = concatenate_and_pca_fusion(
                    features_dfs, fusion_dim, is_train, pca_model)
                if len(fused_features) == 0:
                    print("Warning: Feature fusion failed, skipping")
                    continue
                fused_df = pd.DataFrame(
                    fused_features,
                    columns=[f'fused_feature_{i}' for i in range(fused_features.shape[1])]
                )
                fused_df.insert(0, 'drug_id', common_ids)
                if is_train and pca_model is not None:
                    pca_path = os.path.join(output_dir, "pca_model3.pkl")
                    with open(pca_path, 'wb') as f:
                        pickle.dump(pca_model, f)
                    print(f"PCA model saved to: {pca_path}")
            
            fusion_path = os.path.join(output_dir, f"{dataset_name}_drug_fused_features3.csv")
            fused_df.to_csv(fusion_path, index=False)
            print(f"Fused features saved to: {fusion_path}")
        else:
            print(f"Warning: No valid features for {dataset_name} dataset, skipping fusion")
    
    print("\n" + "="*50)
    print("=== All dataset feature processing completed ===")
    generate_feature_report(output_dir)

def generate_feature_report(output_dir):
    report_lines = ["\n=== Drug Feature Extraction Statistical Report ==="]
    report_lines.append(f"Report generation time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for dataset_name in ['train', 'val', 'test']:
        report_lines.append(f"\n【{dataset_name} Dataset】")
        feature_files = {
            'Morgan Fingerprints': f"{dataset_name}_drug_morgan_features3.csv",
            'MACCS Fingerprints': f"{dataset_name}_drug_maccs_features3.csv",
            'ESPF Features': f"{dataset_name}_drug_espf_encoding3.csv",
            'Fused Features': f"{dataset_name}_drug_fused_features3.csv"
        }
        
        for feat_name, filename in feature_files.items():
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    drug_count = len(df)
                    feat_dim = len(df.columns) - 1 if 'drug_id' in df.columns else 0
                    report_lines.append(f"  {feat_name}: {drug_count} drugs, {feat_dim}-dimensional features")
                except Exception as e:
                    report_lines.append(f"  {feat_name}: File exists but failed to read - {str(e)[:50]}")
            else:
                report_lines.append(f"  {feat_name}: Not generated")
    
    report_path = os.path.join(output_dir, "feature_extraction_report3.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    for line in report_lines:
        print(line)
    print(f"\nReport saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Drug Feature Extraction and Fusion Tool')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Feature output directory')
    parser.add_argument('--morgan_radius', type=int, default=2, help='Morgan fingerprint radius (default 2)')
    parser.add_argument('--morgan_nbits', type=int, default=2048, help='Morgan fingerprint bits (default 2048)')
    parser.add_argument('--skip_espf', action='store_true', help='Skip ESPF feature extraction')
    parser.add_argument('--skip_maccs', action='store_true', help='Skip MACCS feature extraction')
    parser.add_argument('--espf_vocab_path', type=str, default=ESPF_VOCAB_PATH, help='ESPF vocabulary path')
    parser.add_argument('--espf_subword_map', type=str, default=ESPF_SUBWORD_MAP, help='ESPF subword map path')
    parser.add_argument('--fusion_dim', type=int, default=256, help='Target dimension for feature fusion (default 256)')
    parser.add_argument('--feature_types', nargs='+', default=['morgan', 'maccs', 'espf'], 
                        choices=['morgan', 'maccs', 'espf'], help='Feature types to extract')
    
    args = parser.parse_args()
    print(f"==== Drug Feature Extraction and Fusion Tool ====")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature types: {args.feature_types}")
    print(f"Morgan fingerprint: radius {args.morgan_radius}, bits {args.morgan_nbits}")
    print(f"Fused feature dimension: {args.fusion_dim}")
    print(f"Skip ESPF: {args.skip_espf}, Skip MACCS: {args.skip_maccs}")
    
    process_features_by_dataset_split(**vars(args))

if __name__ == "__main__":
    main()