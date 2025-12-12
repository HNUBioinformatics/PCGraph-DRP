import pandas as pd
import requests
import time
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

PATHWAY_FILE = r"../PCGraph-DRP/data/raw/filtered_pathways_714.csv"
OUTPUT_DIR = r"../PCGraph-DRP/pathway_network"

def extract_unique_proteins(file_path):
    print(f"Step 1: Extracting unique protein (gene) names from {os.path.basename(file_path)}...")
    
    try:
        df = pd.read_csv(file_path, header=None)
        gene_column = df.iloc[:, 0]
    except FileNotFoundError:
        print(f"Error: Input file not found, please check the path: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    unique_proteins = set()
    
    for gene_string in gene_column.dropna():
        genes = str(gene_string).strip().split('|')
        unique_proteins.update(gene for gene in genes if gene)
        
    sorted_proteins = sorted(list(unique_proteins))
    print(f"Successfully extracted {len(sorted_proteins)} unique protein names.")
    return sorted_proteins

def get_uniprot_sequence(protein_name, organism_id="9606"):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    
    query_reviewed = f"(gene:{protein_name}) AND (organism_id:{organism_id}) AND (reviewed:true)"
    params_reviewed = {
        'query': query_reviewed,
        'format': 'fasta',
        'size': 1,
        'sort': 'length desc'
    }
    
    try:
        response = requests.get(base_url, params=params_reviewed, timeout=10)
        if response.status_code == 200 and response.text.strip():
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                return ''.join(lines[1:])
    except requests.exceptions.RequestException as e:
        print(f"  - Warning: Network request failed when querying {protein_name}: {e}")

    query_all = f"(gene:{protein_name}) AND (organism_id:{organism_id})"
    params_all = {
        'query': query_all,
        'format': 'fasta',
        'size': 1,
        'sort': 'length desc'
    }

    try:
        time.sleep(0.1)
        response = requests.get(base_url, params=params_all, timeout=10)
        if response.status_code == 200 and response.text.strip():
            lines = response.text.strip().split('\n')
            if len(lines) > 1:
                return ''.join(lines[1:])
    except requests.exceptions.RequestException as e:
        print(f"  - Warning: Network request failed during the second query for {protein_name}: {e}")
            
    return None

def main():
    protein_list = extract_unique_proteins(PATHWAY_FILE)
    
    if protein_list is None:
        return

    print("\nStep 2: Starting to retrieve protein sequences from UniProt database (This may take some time)...")
    
    results = []
    failed_proteins = []
    
    for protein in tqdm(protein_list, desc="Retrieving sequences"):
        sequence = get_uniprot_sequence(protein)
        
        if sequence:
            results.append({'protein_name': protein, 'sequence': sequence})
        else:
            failed_proteins.append(protein)
        
        time.sleep(0.2) 

    print("Sequence retrieval completed!")

    print("\nStep 3: Saving results...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_file_path = os.path.join(OUTPUT_DIR, "all_pathway_protein_sequences.csv")
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file_path, index=False, encoding='utf-8')
    
    failed_file_path = os.path.join(OUTPUT_DIR, "failed_protein_list.txt")
    if failed_proteins:
        with open(failed_file_path, 'w', encoding='utf-8') as f:
            for protein_name in failed_proteins:
                f.write(f"{protein_name}\n")

    print("\n--- Task Summary ---")
    print(f"Total number of unique proteins: {len(protein_list)}")
    print(f"Number of sequences successfully retrieved: {len(results)}")
    print(f"Number of failed sequence retrievals: {len(failed_proteins)}")
    print(f"Successfully retrieved sequences saved to: {output_file_path}")
    if failed_proteins:
        print(f"List of proteins with failed retrieval saved to: {failed_file_path}")
    print("--- All tasks completed ---")

if __name__ == "__main__":
    main()