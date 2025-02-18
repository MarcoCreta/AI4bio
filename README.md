# Protein-to-Pathway Classification

## Overview
This repository contains code and models for classifying protein sequences into biological pathways in *Homo sapiens*. The project leverages transformer-based protein language models to extract meaningful sequence representations and perform multi-class, multi-label classification.

## Project Goal
The primary objective is to determine whether a given protein sequence belongs to specific biological pathways. This is achieved by:
- Extracting context-aware protein sequence representations.
- Utilizing transformer-based models for classification.
- Handling unbalanced datasets through weighting and oversampling techniques.

## Models Used
The following pretrained models are employed for feature extraction:
- **ProteinBERT**: A BERT-based model trained on UniRef90.
- **ProteinGPT2**: A GPT2-based generative model trained on UniRef50.
- **ProtT5-XL-BFD**: A T5-based encoder-decoder model trained on 2.1 billion protein sequences.

## Datasets
- **UP000005640 (Homo sapiens Proteome)**: A comprehensive dataset from UniProt containing 20,644 protein entries.
- **Reactome Pathways (R-HSA-*)**: Manually curated datasets containing proteins associated with specific biological pathways.

## Methods
### Classification Task
- Multi-class, multi-label classification where each protein can belong to multiple pathways.
- Pathways selected: *Metabolism* (2,022 proteins) and *Apoptosis (R-HSA-109581)* (168 proteins), with 42 proteins overlapping.
- Custom weighting and batch balancing to address data imbalance.

### Data Processing
- Tokenized protein sequences using transformer model-specific tokenization.
- Sliding window chunking with overlap to maximize sequence utilization.
- Embedding extraction and feature transformation for classification.

### Model Architecture
- **FeedForward Network**: Multi-layer perceptron for classification.
- **Attentional Pooling (optional)**: Self-attention mechanism to aggregate feature representations.

## Training
- Weighted **Binary Cross-Entropy Loss** for multi-label classification.
- Custom **Focal Loss** for handling imbalanced data.
- Oversampling to ensure balanced representation in training batches.
- Controlled training with fixed seeds and early stopping.

## Results
- **ProteinGPT2** and **ProtT5** exhibited superior performance over ProteinBERT.
- The final model effectively classifies protein sequences into pathways with promising precision and recall scores.

## Installation & Usage
1. Clone the repository
2. Put UP000005640.fasta file in the data folder
3. Precompute the dataset
4.
```sh
   python precompute_data.py
```
5. Manually map and add the desired pathway mappings R-HSA-* to the data folder
6. Change parameters from config file as desired
7. Run training on the given pathways:
```sh
python main.py
```
