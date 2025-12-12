import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def load_pathway_data(file_path):
    print("Reading pathway data...")
    df = pd.read_csv(file_path)
    print(f"Read {len(df)} pathways")
    pathway_genes = {}
    for idx, row in df.iterrows():
        genes = set(row['genes'].split('|'))
        pathway_genes[row['name']] = genes
    print(f"Conversion completed, total {len(pathway_genes)} pathways")
    return pathway_genes, df

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def calculate_jaccard_matrix(pathway_genes):
    pathway_names = list(pathway_genes.keys())
    n_pathways = len(pathway_names)
    print(f"Calculating {n_pathways} x {n_pathways} Jaccard similarity matrix...")
    jaccard_matrix = np.zeros((n_pathways, n_pathways))
    for i, pathway1 in enumerate(pathway_names):
        for j, pathway2 in enumerate(pathway_names):
            if i == j:
                jaccard_matrix[i][j] = 1.0
            elif i < j:
                similarity = jaccard_similarity(
                    pathway_genes[pathway1], 
                    pathway_genes[pathway2]
                )
                jaccard_matrix[i][j] = similarity
                jaccard_matrix[j][i] = similarity

        if (i + 1) % 100 == 0:
            print(f"Completed calculation for {i + 1}/{n_pathways} pathways")
    jaccard_df = pd.DataFrame(jaccard_matrix, 
                             index=pathway_names, 
                             columns=pathway_names)
    print("Jaccard matrix calculation completed!")
    return jaccard_df

def analyze_jaccard_statistics(jaccard_matrix):
    upper_triangle = jaccard_matrix.values[np.triu_indices_from(jaccard_matrix.values, k=1)]
    print("\n=== Jaccard Coefficient Statistical Analysis ===")
    print(f"Total pathway pairs: {len(upper_triangle)}")
    print(f"Mean similarity: {upper_triangle.mean():.4f}")
    print(f"Standard deviation: {upper_triangle.std():.4f}")
    print(f"Maximum similarity: {upper_triangle.max():.4f}")
    print(f"Minimum similarity: {upper_triangle.min():.4f}")
    print(f"Median: {np.median(upper_triangle):.4f}")
    print(f"\nSimilarity distribution:")
    print(f"Pathway pairs with similarity = 0: {(upper_triangle == 0).sum()} ({(upper_triangle == 0).mean()*100:.1f}%)")
    print(f"Pathway pairs with similarity > 0.1: {(upper_triangle > 0.1).sum()} ({(upper_triangle > 0.1).mean()*100:.1f}%)")
    print(f"Pathway pairs with similarity > 0.3: {(upper_triangle > 0.3).sum()} ({(upper_triangle > 0.3).mean()*100:.1f}%)")
    print(f"Pathway pairs with similarity > 0.5: {(upper_triangle > 0.5).sum()} ({(upper_triangle > 0.5).mean()*100:.1f}%)")

def find_most_similar_pathways(jaccard_matrix, top_n=10):
    print(f"\n=== Top {top_n} Most Similar Pathway Pairs ===")
    pathway_names = jaccard_matrix.index.tolist()
    similar_pairs = []
    for i in range(len(pathway_names)):
        for j in range(i + 1, len(pathway_names)):
            similarity = jaccard_matrix.iloc[i, j]
            similar_pairs.append({
                'pathway1': pathway_names[i],
                'pathway2': pathway_names[j],
                'jaccard_similarity': similarity
            })
    similar_pairs_df = pd.DataFrame(similar_pairs)
    top_pairs = similar_pairs_df.nlargest(top_n, 'jaccard_similarity')
    for idx, row in top_pairs.iterrows():
        print(f"{row['pathway1']} <-> {row['pathway2']}: {row['jaccard_similarity']:.4f}")
    
    return top_pairs

def visualize_jaccard_matrix(jaccard_matrix, sample_size=50):
    print(f"\nGenerating visualization plots (sampling {sample_size} pathways)...")
    if len(jaccard_matrix) > sample_size:
        sampled_pathways = np.random.choice(jaccard_matrix.index, sample_size, replace=False)
        sample_matrix = jaccard_matrix.loc[sampled_pathways, sampled_pathways]
    else:
        sample_matrix = jaccard_matrix
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    sns.heatmap(sample_matrix, cmap='viridis', cbar=True, 
                square=True, ax=axes[0,0], 
                xticklabels=False, yticklabels=False)
    axes[0,0].set_title('Jaccard Similarity Coefficient Heatmap Between Pathways')
    upper_triangle = sample_matrix.values[np.triu_indices_from(sample_matrix.values, k=1)]
    axes[0,1].hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].set_xlabel('Jaccard Similarity Coefficient')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Similarity Coefficient Distribution Histogram')
    distance_matrix = 1 - sample_matrix.values
    condensed_distances = pdist(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')
    dendrogram(linkage_matrix, ax=axes[1,0], leaf_rotation=90)
    axes[1,0].set_title('Pathway Hierarchical Clustering Dendrogram')
    sorted_similarities = np.sort(upper_triangle)
    cumulative_prob = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)
    axes[1,1].plot(sorted_similarities, cumulative_prob, linewidth=2)
    axes[1,1].set_xlabel('Jaccard Similarity Coefficient')
    axes[1,1].set_ylabel('Cumulative Probability')
    axes[1,1].set_title('Similarity Coefficient Cumulative Distribution Function')
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pathway_jaccard_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(jaccard_matrix, pathway_genes, output_prefix='pathway_jaccard'):
    print(f"\nSaving results...")
    jaccard_matrix.to_csv(f'{output_prefix}_matrix.csv')
    print(f"Jaccard matrix saved to {output_prefix}_matrix.csv")
    pathway_info = []
    for pathway, genes in pathway_genes.items():
        pathway_info.append({
            'pathway_name': pathway,
            'gene_count': len(genes),
            'genes': '|'.join(sorted(genes))
        })
    pathway_info_df = pd.DataFrame(pathway_info)
    pathway_info_df.to_csv(f'{output_prefix}_pathway_info.csv', index=False)
    print(f"Pathway information saved to {output_prefix}_pathway_info.csv")
    
    similar_pairs = []
    pathway_names = jaccard_matrix.index.tolist()
    
    for i in range(len(pathway_names)):
        for j in range(i + 1, len(pathway_names)):
            similarity = jaccard_matrix.iloc[i, j]
            if similarity > 0:
                similar_pairs.append({
                    'pathway1': pathway_names[i],
                    'pathway2': pathway_names[j],
                    'jaccard_similarity': similarity,
                    'shared_genes': len(pathway_genes[pathway_names[i]] & pathway_genes[pathway_names[j]]),
                    'union_genes': len(pathway_genes[pathway_names[i]] | pathway_genes[pathway_names[j]])
                })
    
    similar_pairs_df = pd.DataFrame(similar_pairs)
    similar_pairs_df = similar_pairs_df.sort_values('jaccard_similarity', ascending=False)
    similar_pairs_df.to_csv(f'{output_prefix}_similar_pairs.csv', index=False)
    print(f"Similar pathway pairs saved to {output_prefix}_similar_pairs.csv")

def main():
    file_path = '../PCGraph-DRP/data/raw/filtered_pathways_714.csv'
    
    try:
        pathway_genes, original_df = load_pathway_data(file_path)
        jaccard_matrix = calculate_jaccard_matrix(pathway_genes)
        analyze_jaccard_statistics(jaccard_matrix)
        top_similar = find_most_similar_pathways(jaccard_matrix, top_n=20)
        visualize_jaccard_matrix(jaccard_matrix, sample_size=100)
        save_results(jaccard_matrix, pathway_genes)
        print("\n=== Analysis Completed! ===")
        print("Generated files:")
        print("- pathway_jaccard_matrix.csv: Jaccard similarity coefficient matrix")
        print("- pathway_jaccard_pathway_info.csv: Detailed pathway information")
        print("- pathway_jaccard_similar_pairs.csv: List of similar pathway pairs")
        print("- pathway_jaccard_analysis.png: Visualization plots")
        return jaccard_matrix, pathway_genes
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        print("Please ensure the file path is correct")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    jaccard_matrix, pathway_genes = main()