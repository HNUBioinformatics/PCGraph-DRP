import pandas as pd
import requests
import time
from typing import Optional, Set

def get_uniprot_sequence(gene_name: str, organism: str = "9606") -> Optional[str]:
    try:
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        query = f"gene:{gene_name} AND organism_id:{organism} AND reviewed:true"
        
        params = {
            'query': query,
            'format': 'fasta',
            'size': 5,
            'sort': 'length desc'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200 and response.text.strip():
            fasta_records = response.text.strip().split('>')[1:]
            sequences = []
            
            for record in fasta_records:
                if not record.strip():
                    continue
                parts = record.split('\n', 1)
                if len(parts) < 2:
                    continue
                title, sequence_lines = parts
                sequence = ''.join(sequence_lines.split('\n')).strip()
                if sequence:
                    sequences.append((title, sequence))
            
            if sequences:
                sequences.sort(key=lambda x: len(x[1]), reverse=True)
                longest_title, longest_sequence = sequences[0]
                print(f"  Selected longest sequence (length: {len(longest_sequence)}), source: {longest_title.split()[0]}")
                return longest_sequence
        
        return try_alternative_search(gene_name)
        
    except Exception as e:
        print(f"Error querying gene {gene_name}: {e}")
        return None

def try_alternative_search(gene_name: str) -> Optional[str]:
    strategies = [
        (f"gene_exact:{gene_name}", "Exact gene name match"),
        (f"protein_name:{gene_name}", "Protein name match"),
        (f"gene_synonym:{gene_name}", "Gene synonym match"),
        (f"gene:{gene_name}*", "Fuzzy gene name match"),
        (f"accession:{gene_name}", "Accession number match"),
    ]
    
    for strategy, desc in strategies:
        try:
            params = {
                'query': f"{strategy} AND organism_id:9606 AND reviewed:true",
                'format': 'fasta',
                'size': 3,
                'sort': 'length desc'
            }
            
            response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params)
            
            if response.status_code == 200 and response.text.strip():
                fasta_records = response.text.strip().split('>')[1:]
                sequences = []
                
                for record in fasta_records:
                    if not record.strip():
                        continue
                    parts = record.split('\n', 1)
                    if len(parts) < 2:
                        continue
                    title, sequence_lines = parts
                    sequence = ''.join(sequence_lines.split('\n')).strip()
                    if sequence:
                        sequences.append((title, sequence))
                
                if sequences:
                    sequences.sort(key=lambda x: len(x[1]), reverse=True)
                    longest_title, longest_sequence = sequences[0]
                    print(f"  Alternative strategy[{desc}] found longest sequence (length: {len(longest_sequence)})")
                    return longest_sequence
                    
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  Alternative strategy[{desc}] error: {e}")
            continue
    
    return None

def is_valid_protein_sequence(sequence: str) -> bool:
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    return all(c in valid_amino_acids for c in sequence.upper())

def extract_genes_from_omics_columns(omics_df: pd.DataFrame) -> Set[str]:
    feature_columns = omics_df.columns[1:]
    genes = set()
    suffixes = ['_mutation', '_cnv', '_rna']
    
    for col in feature_columns:
        for suffix in suffixes:
            if col.endswith(suffix):
                gene = col[:-len(suffix)]
                if gene:
                    genes.add(gene)
                break
    
    return genes

def get_unique_protein_sequences(omics_file_path: str, output_file_path: str):
    try:
        omics_df = pd.read_csv(omics_file_path, nrows=0)
        print(f"Successfully read omics data file, total {len(omics_df.columns)} columns (first column is cell line)")
    except Exception as e:
        print(f"Failed to read omics data file: {e}")
        return
    
    unique_genes = extract_genes_from_omics_columns(omics_df)
    total_unique = len(unique_genes)
    
    if total_unique == 0:
        print("No genes extracted from omics data column names, program exits")
        return
    
    print(f"Extracted {total_unique} unique genes from omics data")
    
    result_data = []
    
    for idx, gene in enumerate(sorted(unique_genes), 1):
        print(f"\nQuerying gene {gene} ({idx}/{total_unique})")
        
        sequence = get_uniprot_sequence(gene)
        
        sequence_length = len(sequence) if sequence else 0
        sequence_valid = is_valid_protein_sequence(sequence) if sequence else False
        
        result_data.append({
            'gene': gene,
            'sequence': sequence if sequence else '',
            'sequence_length': sequence_length,
            'sequence_valid': sequence_valid
        })
        
        if sequence:
            print(f"  ✓ Sequence found, length: {sequence_length}, valid: {sequence_valid}")
        else:
            print(f"  ✗ Sequence not found")
        
        time.sleep(0.5)
    
    result_df = pd.DataFrame(result_data)
    
    try:
        result_df.to_csv(output_file_path, index=False)
        print(f"\nResults saved to: {output_file_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")
        return
    
    matched_count = sum(1 for seq in result_df['sequence'] if seq)
    unmatched_count = total_unique - matched_count
    invalid_sequences = sum(1 for valid in result_df['sequence_valid'] if not valid)
    
    print(f"\n=== Matching Result Statistics ===")
    print(f"Total unique genes: {total_unique}")
    print(f"Successfully matched: {matched_count}")
    print(f"Failed to match: {unmatched_count}")
    print(f"Invalid sequences: {invalid_sequences}")
    
    problematic_genes = []
    if unmatched_count > 0:
        problematic_genes.extend(result_df[result_df['sequence'] == '']['gene'].tolist())
    if invalid_sequences > 0:
        problematic_genes.extend(result_df[result_df['sequence_valid'] == False]['gene'].tolist())
    
    if problematic_genes:
        problem_file = output_file_path.replace('.csv', '_problematic.txt')
        with open(problem_file, 'w') as f:
            for gene in problematic_genes:
                f.write(f"{gene}\n")
        print(f"\nList of problematic genes (unmatched or invalid sequences) saved to: {problem_file}")

if __name__ == "__main__":
    omics_input_file = r"../PCGraph-DRP/data/raw/reduced_omics_data.csv"
    sequence_output_file = r"../PCGraph-DRP/pathway_network/omics_protein_sequences.csv"
    
    print("Starting to extract genes from omics data and retrieve protein sequences...")
    get_unique_protein_sequences(omics_input_file, sequence_output_file)
    
    print("\nProcessing completed!")