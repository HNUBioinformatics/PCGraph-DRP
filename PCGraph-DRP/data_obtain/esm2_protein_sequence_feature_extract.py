import torch
import pandas as pd
import numpy as np
from transformers import EsmTokenizer, EsmModel
import gc
from tqdm import tqdm
import warnings
import os
import psutil
import json
warnings.filterwarnings('ignore')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

class ESM2_650M_Extractor:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", max_length=1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        self._check_system_resources()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        try:
            print(f"load ESM2-650M model...")
            
            cache_dir = r"../PCGraph-DRP/data/models/esm2_650M_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            print("1/2 Loading Tokenizer...")
            self.tokenizer = EsmTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                resume_download=True
            )
            
            print("2/2 Loading Model (This may take a few minutes)...")
            self.model = EsmModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                resume_download=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            
            self.model.eval()
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            print("ESM2-650M loaded successfully!")
            self._print_model_info()
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Trying fallback solution...")
            self._load_fallback_model()
        
        self.max_length = max_length
        self.feature_cache = {}
        
        self.failed_sequences = []
        self.truncated_sequences = []
        self.error_stats = {
            'memory_error': 0,
            'tokenization_error': 0,
            'model_error': 0,
            'unknown_error': 0,
            'empty_sequence': 0,
            'truncated': 0
        }
        
        self.truncation_config = {
            'strategy': 'smart',
            'max_length': 2000,
            'preserve_domains': True
        }
        
    def _check_system_resources(self):
        cpu_memory = psutil.virtual_memory()
        print(f"System Memory: {cpu_memory.total / 1e9:.1f}GB (Available: {cpu_memory.available / 1e9:.1f}GB)")
        
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory = gpu_props.total_memory / 1e9
            print(f"GPU: {gpu_props.name}")
            print(f"GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 6:
                print("Warning: GPU memory may be insufficient to run the 650M model")
    
    def _print_model_info(self):
        config = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n=== Model Information ===")
        print(f"Model Name: ESM2-650M")
        print(f"Number of Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Feature Dimension: {config.hidden_size}D")
        print(f"Max Sequence Length: {config.max_position_embeddings}")
        print(f"Truncation Strategy: {self.truncation_config['strategy']}")
        print(f"Truncation Length: {self.truncation_config['max_length']}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"Current GPU Memory: {allocated:.2f}GB (Cached: {cached:.2f}GB)")
    
    def _load_fallback_model(self):
        print("Loading fallback model: ESM2-150M")
        fallback_model = "facebook/esm2_t30_150M_UR50D"
        cache_dir = r"../PCGraph-DRP/models/esm2_fallback_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tokenizer = EsmTokenizer.from_pretrained(fallback_model, cache_dir=cache_dir)
        self.model = EsmModel.from_pretrained(
            fallback_model, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()
    
    def _get_known_domains(self, gene_name):
        known_domains = {
            'LRRK2': {
                'kinase_domain': (1874, 2138),
                'roc_domain': (1335, 1510),
                'cor_domain': (1511, 1878)
            },
            'MTOR': {
                'kinase_domain': (2159, 2549),
                'heat_repeats': (1000, 2000)
            },
            'ROS1': {
                'kinase_domain': (2010, 2347),
                'transmembrane': (1900, 1950)
            },
            'MYO9A': {
                'motor_domain': (1, 800),
                'iq_motifs': (800, 1200),
                'rho_gap': (1200, 1500)
            },
            'ANK3': {
                'ank_repeats': (1000, 3000),
                'spectrin_binding': (3500, 4000)
            }
        }
        return known_domains.get(gene_name, {})
    
    def _smart_truncate(self, sequence, gene_name, max_length=2000):
        if len(sequence) <= max_length:
            return sequence, 'no_truncation'
        
        domains = self._get_known_domains(gene_name)
        
        if domains and self.truncation_config['preserve_domains']:
            important_regions = []
            for domain_name, (start, end) in domains.items():
                if start < len(sequence) and end <= len(sequence):
                    important_regions.append((start, end, domain_name))
            
            if important_regions:
                important_regions.sort(key=lambda x: 0 if 'kinase' in x[2].lower() else 1)
                
                truncated_seq = ""
                used_length = 0
                preserved_domains = []
                
                for start, end, domain_name in important_regions:
                    domain_seq = sequence[start:end]
                    if used_length + len(domain_seq) <= max_length:
                        truncated_seq += domain_seq
                        used_length += len(domain_seq)
                        preserved_domains.append(domain_name)
                    else:
                        remaining_space = max_length - used_length
                        if remaining_space > 100:
                            truncated_seq += domain_seq[:remaining_space]
                            preserved_domains.append(f"{domain_name}_partial")
                        break
                
                if len(truncated_seq) >= 50:
                    strategy = f"domain_preservation_{'+'.join(preserved_domains)}"
                    return truncated_seq, strategy
        
        if len(sequence) > 4000:
            start_part = sequence[:int(max_length * 0.6)]
            end_part = sequence[-int(max_length * 0.4):]
            truncated_seq = start_part + end_part
            strategy = 'two_ends'
        elif len(sequence) > 3000:
            truncated_seq = sequence[:max_length]
            strategy = 'n_terminal'
        else:
            start_pos = (len(sequence) - max_length) // 2
            truncated_seq = sequence[start_pos:start_pos + max_length]
            strategy = 'middle'
        
        return truncated_seq, strategy
    
    def _sliding_window_features(self, sequence, gene_name, window_size=1800, overlap=200):
        if len(sequence) <= window_size:
            return self._extract_single_sequence(sequence, gene_name)
        
        features_list = []
        step_size = window_size - overlap
        
        for i in range(0, len(sequence) - window_size + 1, step_size):
            window = sequence[i:i + window_size]
            window_features = self._extract_single_sequence(window, f"{gene_name}_window_{i}")
            if window_features is not None:
                features_list.append(window_features[0])
        
        if features_list:
            avg_features = np.mean(features_list, axis=0)
            return avg_features.reshape(1, -1)
        
        return None
    
    def _extract_single_sequence(self, sequence, gene_name):
        try:
            seq_hash = hash(sequence)
            if seq_hash in self.feature_cache:
                return self.feature_cache[seq_hash]
            
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                    
                    attention_mask = inputs['attention_mask']
                    seq_mask = attention_mask[:, 1:-1]
                    seq_embeddings = last_hidden_states[:, 1:-1, :]
                    
                    if seq_mask.sum() > 0:
                        masked_embeddings = seq_embeddings * seq_mask.unsqueeze(-1)
                        features = masked_embeddings.sum(dim=1) / seq_mask.sum(dim=1).unsqueeze(-1)
                    else:
                        features = last_hidden_states[:, 0, :]
                    
                    features_np = features.cpu().numpy().astype(np.float32)
                    self.feature_cache[seq_hash] = features_np
                    
                    return features_np
                    
        except Exception as e:
            print(f"Sequence extraction error {gene_name}: {str(e)[:100]}")
            return None
    
    def extract_sequence_features(self, sequence, gene_name=None, use_mean_pooling=True):
        try:
            if not sequence or len(sequence.strip()) == 0:
                self.error_stats['empty_sequence'] += 1
                self.failed_sequences.append({
                    'gene_name': gene_name,
                    'error_type': 'empty_sequence',
                    'error_message': 'Empty sequence',
                    'sequence_length': 0
                })
                print(f" {gene_name}: Empty sequence")
                return None
            
            original_length = len(sequence)
            truncation_strategy = 'none'
            
            if original_length > 2048:
                if self.truncation_config['strategy'] == 'sliding_window':
                    print(f" {gene_name}: Processing ultra-long sequence with sliding window (Length: {original_length})")
                    return self._sliding_window_features(sequence, gene_name)
                else:
                    sequence, truncation_strategy = self._smart_truncate(
                        sequence, gene_name, self.truncation_config['max_length']
                    )
                    
                    self.error_stats['truncated'] += 1
                    self.truncated_sequences.append({
                        'gene_name': gene_name,
                        'original_length': original_length,
                        'truncated_length': len(sequence),
                        'truncation_strategy': truncation_strategy,
                        'reduction_ratio': f"{(1 - len(sequence)/original_length)*100:.1f}%"
                    })
                    
                    print(f"  {gene_name}: Smart truncation {original_length} -> {len(sequence)} ({truncation_strategy})")
            
            if len(sequence) > self.max_length - 2:
                sequence = sequence[:self.max_length-2]
                if truncation_strategy == 'none':
                    truncation_strategy = 'regular_truncation'
            
            return self._extract_single_sequence(sequence, gene_name)
                        
        except torch.cuda.OutOfMemoryError:
            self.error_stats['memory_error'] += 1
            self.failed_sequences.append({
                'gene_name': gene_name,
                'error_type': 'memory_error',
                'error_message': 'GPU out of memory',
                'sequence_length': original_length
            })
            print(f" {gene_name}: GPU out of memory")
            
            torch.cuda.empty_cache()
            gc.collect()
            return None
                    
        except Exception as e:
            self.error_stats['unknown_error'] += 1
            self.failed_sequences.append({
                'gene_name': gene_name,
                'error_type': 'unknown_error',
                'error_message': str(e)[:100],
                'sequence_length': original_length if 'original_length' in locals() else len(sequence)
            })
            print(f" {gene_name}: {str(e)[:100]}")
            return None
    
    def process_batch_smart(self, sequences, gene_names, initial_batch_size=4):
        features_list = []
        current_batch_size = initial_batch_size
        
        seq_with_info = [(i, seq, name) for i, (seq, name) in enumerate(zip(sequences, gene_names))]
        seq_with_info.sort(key=lambda x: len(x[1]) if x[1] else 0)
        
        print(f"Start processing {len(sequences)} sequences")
        if seq_with_info:
            print(f"Sequence length range: {len(seq_with_info[0][1]) if seq_with_info[0][1] else 0} - {len(seq_with_info[-1][1]) if seq_with_info[-1][1] else 0}")
        
        i = 0
        pbar = tqdm(total=len(seq_with_info), desc="Extracting features")
        
        while i < len(seq_with_info):
            try:
                batch_end = min(i + current_batch_size, len(seq_with_info))
                batch_data = seq_with_info[i:batch_end]
                
                batch_features = []
                for idx, seq, gene_name in batch_data:
                    features = self.extract_sequence_features(seq, gene_name)
                    if features is not None:
                        batch_features.append((idx, features[0]))
                    else:
                        zero_features = np.zeros(self.model.config.hidden_size, dtype=np.float32)
                        batch_features.append((idx, zero_features))
                
                features_list.extend(batch_features)
                i = batch_end
                pbar.update(len(batch_data))
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if memory_used > 0.9 and current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size - 1)
                    elif memory_used < 0.7 and current_batch_size < 8:
                        current_batch_size += 1
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                else:
                    i += 1
        
        pbar.close()
        
        features_list.sort(key=lambda x: x[0])
        features_array = np.array([f[1] for f in features_list])
        
        print(f" Feature extraction completed: {features_array.shape}")
        self._print_summary()
        
        return features_array
    
    def _print_summary(self):
        total_errors = sum(self.error_stats.values()) - self.error_stats['truncated']
        total_truncated = self.error_stats['truncated']
        
        print(f"\n=== Processing Summary ===")
        
        if total_truncated > 0:
            print(f"  Successfully truncated: {total_truncated} sequences")
            print("\nTruncation details:")
            for truncated in self.truncated_sequences[:5]:
                gene_name = truncated['gene_name']
                original = truncated['original_length']
                truncated_len = truncated['truncated_length']
                strategy = truncated['truncation_strategy']
                ratio = truncated['reduction_ratio']
                print(f"  • {gene_name}: {original} -> {truncated_len} ({strategy}, reduced {ratio})")
            
            if len(self.truncated_sequences) > 5:
                print(f"  ... {len(self.truncated_sequences) - 5} more truncated sequences")
        
        if total_errors > 0:
            print(f"\n Processing failed: {total_errors} sequences")
            for error_type, count in self.error_stats.items():
                if count > 0 and error_type != 'truncated':
                    print(f"  {error_type}: {count}")
        
        if total_errors == 0 and total_truncated == 0:
            print(" All sequences processed successfully!")
    
    def save_features(self, features, gene_names, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        features_file = os.path.join(output_dir, "pathway_protein_esm2_650M_features.npy")
        np.save(features_file, features)
        
        feature_df = pd.DataFrame({'gene_name': gene_names})
        for i in range(features.shape[1]):
            feature_df[f'esm2_650M_dim_{i}'] = features[:, i]
        
        csv_file = os.path.join(output_dir, "pathway_protein_esm2_650M_features.csv")
        feature_df.to_csv(csv_file, index=False)
        
        if self.truncated_sequences:
            truncation_report_file = os.path.join(output_dir, "truncation_report.csv")
            truncation_df = pd.DataFrame(self.truncated_sequences)
            truncation_df.to_csv(truncation_report_file, index=False)
        
        if self.failed_sequences:
            error_report_file = os.path.join(output_dir, "failed_sequences_report.csv")
            error_df = pd.DataFrame(self.failed_sequences)
            error_df.to_csv(error_report_file, index=False)
        
        metadata = {
            'model_name': 'facebook/esm2_t33_650M_UR50D',
            'model_size': '650M parameters',
            'feature_dimension': int(features.shape[1]),
            'num_sequences': int(features.shape[0]),
            'truncation_config': self.truncation_config,
            'processing_stats': {
                'total_processed': len(gene_names),
                'successfully_truncated': len(self.truncated_sequences),
                'failed': len(self.failed_sequences),
                'success_rate': f"{((len(gene_names) - len(self.failed_sequences)) / len(gene_names) * 100):.2f}%"
            },
            'error_statistics': self.error_stats,
            'extraction_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = os.path.join(output_dir, "esm2_650M_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== File saving completed ===")
        print(f"Feature file: {features_file}")
        print(f"CSV file: {csv_file}")
        if self.truncated_sequences:
            print(f"Truncation report: {truncation_report_file}")
        if self.failed_sequences:
            print(f"Error report: {error_report_file}")
        print(f"Metadata: {metadata_file}")
        
        return features_file, csv_file, metadata_file

def main():
    data_path = r"../PCGraph-DRP/pathway_network/all_pathway_protein_sequences.csv"
    output_dir = r"../PCGraph-DRP/pathway_network"
    
    print(" Loading data...")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f" Data loading failed: {e}")
        return
    
    if 'gene_name' not in df.columns or 'sequence' not in df.columns:
        print(f" Missing required columns. Current columns: {df.columns.tolist()}")
        return
    
    original_count = len(df)
    df = df.dropna(subset=['gene_name'])
    
    valid_sequences = df['sequence'].dropna()
    if len(valid_sequences) > 0:
        seq_lengths = valid_sequences.str.len()
        print(f"\nSequence length analysis:")
        print(f"  Valid sequences: {len(valid_sequences)}/{len(df)}")
        print(f"  Average length: {seq_lengths.mean():.1f}")
        print(f"  Median: {seq_lengths.median():.1f}")
        print(f"  Range: {seq_lengths.min()} - {seq_lengths.max()}")
        print(f"  Need truncation (>2048): {(seq_lengths > 2048).sum()} sequences")
        
        long_sequences = df[df['sequence'].str.len() > 2048]
        if not long_sequences.empty:
            print(f"\nSequences needing truncation:")
            for _, row in long_sequences.iterrows():
                gene_name = row['gene_name']
                seq_len = len(row['sequence'])
                print(f"  • {gene_name}: {seq_len} aa")
    
    print("\n Initializing ESM2-650M...")
    extractor = ESM2_650M_Extractor(
        model_name="facebook/esm2_t33_650M_UR50D",
        max_length=1024
    )
    
    print("\n Starting feature extraction...")
    sequences = df['sequence'].fillna('').tolist()
    gene_names = df['gene_name'].tolist()
    
    features = extractor.process_batch_smart(sequences, gene_names, initial_batch_size=2)
    
    print("\n Saving results...")
    feature_file, csv_file, metadata_file = extractor.save_features(
        features, gene_names, output_dir
    )
    
    print(f"\n Task completed!")
    print(f"Total sequences: {len(sequences)}")
    print(f"Successfully processed: {len(sequences) - len(extractor.failed_sequences)}")
    print(f"Smart truncated: {len(extractor.truncated_sequences)}")
    print(f"Processing failed: {len(extractor.failed_sequences)}")
    success_rate = ((len(sequences) - len(extractor.failed_sequences)) / len(sequences) * 100)
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Feature dimension: {features.shape[1]}D")
    print(f"File size: {os.path.getsize(feature_file) / 1e6:.1f}MB")
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {max_memory:.2f}GB")

if __name__ == "__main__":
    main()