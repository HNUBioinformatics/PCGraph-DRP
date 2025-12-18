<<<<<<< HEAD
# PCGraph-DRP

PCGraph-DRP：A Pathway-Cell Line Heterogeneous Graph Model Driven by Multi-Omics Data for Cancer Drug Response Prediction

## Requirements

- Please install the environment using anaconda3;
  conda create -n PCGraph-DRP python=3.11.11

- Install the necessary packages.

  torch == 2.6.0+cu118

  torch_cluster == 1.6.3+pt26cu118

  torch-geometric == 2.6.1

  torch_scatter == 2.1.2+pt26cu118

  torch_sparse == 0.6.18+pt26cu118

  torch_spline_conv == 1.2.2+pt26cu118

  torchaudio == 2.6.0+cu118

  torchvision == 0.21.0+cu118

  tornado == 6.5.1

  numpy  == 1.26.4

  pandas == 1.5.3

  scipy == 1.13.1

  rdkit == 2025.3.2

## **Overview**

1. data/ : Contains the required dataset files.

   processed/ : Mapping table between drug names and drug IDs

   ​					  Mapping table between cell line names and cell line IDs

   ​					  Cell line–drug–LN(IC50) triplets

   raw/ ：Omics data

   ​			 Drug SMILES strings

   ​			 GDSC LN(IC50)

   ​			Pathway data

   ​			Pathway protein sequence data

2. run data_preparation.py to obtain the training set, test set, and validation set.

2. run espf_morgan_maccs.py for drug feature processing.

3. data_obtain/ : 

   run cell_pathway_gsva.r to obtain pathway activity scores for each cell line.

   run download_protein_sequence_pathway_gene.py to obtain protein sequence data for proteins in each pathway.

   run esm2_protein_sequence_feature_extract.py to extract protein features using ESM2.

   run esm2_z_score.py to standardize the pathway protein features.

   run pathway_node_feature_extract.py to aggregate protein features into pathway-level features.

   run pathway_jaccard.py to obtain the Jaccard similarity matrix of pathways.

model/ : HierarchicalGNNFeatureExtractor.py PCGraph-DRP model file

train-> run train_omics_gatefusion.py
=======
# PCGraph-DRP
