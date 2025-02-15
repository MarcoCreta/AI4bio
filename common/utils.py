from Bio import SeqIO
from typing import List, Optional
import json
from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from config import Config
import torch
import random
from transformers import pipeline, AutoTokenizer, AutoModel, T5Tokenizer
import re

def read_fasta(file_path:str):
    sequences = []
    identifiers = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))  # Protein sequence
            identifier = record.id.split("|")[-1]  # Extract label if encoded in the FASTA header
            identifiers.append(identifier)
    except Exception as e:
        print(f"Error while reading file {file_path} : {e}")

    return sequences, identifiers

import os
import json
from typing import List, Optional

def pathways_to_class_mapping(
    path: str,
    input_names: List[str],
    extension: str = "fasta",
    output_name: Optional[str] = None
) -> dict:
    """
    Generate a mapping of pathways to class identifiers from multiple FASTA files.

    Args:
        path (str): Directory containing the input files.
        input_names (List[str]): List of input file names without extensions.
        extension (str): File extension for the input files (default: "fasta").
        output_name (Optional[str]): Name of the output JSON file (default: None).

    Returns:
        dict: Mapping of pathways to their corresponding protein identifiers.
    """
    class_mapping = {}

    for file_name in input_names:
        try:
            file_path = os.path.join(path, f"{file_name}.{extension}")
            _, identifiers = read_fasta(file_path)
            class_mapping[file_name] = identifiers

            print(f"class mapping: {file_path} - {'|'.join(str(key) + ':' + str(len(v)) for key, v in class_mapping.items())}")

        except IOError as e:
            print(f"Error reading {file_path}: {e}")

    if output_name:
        output_path = os.path.join(path, f"{output_name}.json")
        try:
            with open(output_path, "w") as outfile:
                json.dump(class_mapping, outfile, indent=4)
        except IOError as e:
            print(f"Error writing to {output_path}: {e}")

    return class_mapping

def count_tokens(dataset, model):
    """
    Computes the ordered count of tokenized sequences.

    Args:
        dataset: Iterable of tuples containing sequence data (e.g., (id, sequence, ...)).
        tokenizer: Tokenizer with an `encode` method that returns tokenized sequences.
        output_file (str): Path to the file where the result will be saved as JSON.

    Returns:
        None
    """



    sequences = dataset.sequences
    if model == "Rostlab/prot_bert":
        tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)
        sequences = [" ".join(list(sequence)) for sequence in sequences]
    if model == "Rostlab/prot_t5_xl_bfd":
        tokenizer = T5Tokenizer.from_pretrained(model, do_lower_case=False)
        sequences = [" ".join(list(sequence)) for sequence in sequences]
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    if model == "nferruz/ProtGPT2":
        tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False)
        sequences = [list(sequence) for sequence in sequences]


    # Tokenize sequences and compute their lengths
    tokenized_lengths = np.array([len(tokenizer("".join(sequence), add_special_tokens=False).input_ids) for sequence in sequences])
    tokenized_lengths = np.sort(tokenized_lengths)
    #remove top 0.03% outliers
    tokenized_lengths = tokenized_lengths[0:int(len(tokenized_lengths)*0.997)]

    # Count the occurrences of each token length
    unique_lengths, unique_counts = np.unique(tokenized_lengths, return_counts=True)

    counts = np.zeros(max(unique_lengths) + 1, dtype=int)

    for length, count in zip(unique_lengths, unique_counts):
        counts[length]=count

    return counts


def print_counts(counts, model, n_chunks):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Get model basename
    model_name = os.path.basename(model)

    # Indices represent token counts, counts[i] = count of sequences with i tokens
    indices = np.arange(len(counts))

    # Plotting Barplot
    plt.figure(figsize=(12, 6))
    plt.bar(indices, counts, width=0.8, color='cornflowerblue', label='Bar Values', capsize=5)

    # Labels and Title for Barplot with model included
    plt.title(f"tokens count for {model_name}", fontsize=14)
    plt.xlabel("Sequences Length", fontsize=12)
    plt.ylabel("Sequences count", fontsize=12)
    plt.legend()

    # Save the barplot
    barplot_path = os.path.join(Config.OUTPUT_PATH, f"plots/barplot_{model_name}.jpg")
    plt.tight_layout()
    plt.savefig(barplot_path, dpi=300)
    plt.show()
    print(f"Barplot saved as {barplot_path}")

    # Adding Boxplot with Binning
    bin_width = 512
    residue = len(counts) % bin_width
    max_bin = len(counts) + (bin_width - residue) if residue != 0 else len(counts)
    bin_edges = np.arange(0, max_bin + 1, bin_width)
    binned_data = [counts[start:end] for start, end in zip(bin_edges[:-1], bin_edges[1:])]

    # Sum counts in each bin, ignoring empty bins
    binned_values = [np.sum(bin_group) for bin_group in binned_data if len(bin_group) > 0]

    data_perc = sum(binned_values[:n_chunks])/sum(binned_values)
    print(f"percentage of data considered: {data_perc:.4f}")

    # Plotting Binned Data as Barplot
    plt.figure(figsize=(12, 6))
    plt.xticks(ticks=np.arange(len(binned_values)), labels=bin_edges[1:], rotation=45)
    plt.bar(np.arange(len(binned_values)), binned_values, width=0.8, color='cornflowerblue',
            label='Bar Values', capsize=5)

    # Labels and Title for Boxplot with bin width and model included
    plt.title(f"tokens Binned Data (Bin Width = {bin_width}) for {model_name}", fontsize=14)
    plt.xlabel(f"Bins (Width={bin_width})", fontsize=12)
    plt.ylabel("Counts in Bins", fontsize=12)

    # Save the boxplot
    boxplot_path = os.path.join(Config.OUTPUT_PATH, f"plots/hist_{model_name}.jpg")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300)
    plt.show()
    print(f"Boxplot saved as {boxplot_path}")


def compute_multilabel_class_weights(onehot_labels):
    """
    Compute class weights for multi-label classification based on class frequency.

    Args:
        onehot_labels (torch.Tensor): A binary tensor of shape (N, num_classes) where each row is a multi-hot vector.

    Returns:
        torch.Tensor: Weights for each class, shape (num_classes,)
    """
    num_classes = onehot_labels.shape[1]
    class_counts = onehot_labels.sum(dim=0).cpu().numpy()  # Count occurrences of each class

    # Compute inverse sqrt class weights (to downweight frequent classes)
    class_weights = 1.0 / (np.sqrt(class_counts +1) - 1)

    # Normalize to keep values reasonable
    class_weights = class_weights / class_weights.sum()

    return class_weights


def merge_embeddings(embeddings):
    """
    Merges overlapping embeddings using a weighted mean for overlapped regions.
    Assumes embeddings are in the shape (num_windows, embedding_dim).
    """
    raise NotImplementedError


def seed_worker(worker_id):
    worker_seed = Config.RANDOMNESS["PYTORCH_SEED"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_random_seeds():
    torch.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    np.random.seed(Config.RANDOMNESS["NUMPY_SEED"])
    torch.cuda.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    random.seed(Config.RANDOMNESS["PYTHON_SEED"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    return g