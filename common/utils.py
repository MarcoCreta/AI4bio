from Bio import SeqIO
from typing import List, Optional
import json
from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

def count_tokens(dataset, tokenizer):
    """
    Computes the ordered count of tokenized sequences.

    Args:
        dataset: Iterable of tuples containing sequence data (e.g., (id, sequence, ...)).
        tokenizer: Tokenizer with an `encode` method that returns tokenized sequences.
        output_file (str): Path to the file where the result will be saved as JSON.

    Returns:
        None
    """
    # Tokenize sequences and compute their lengths
    tokenized_lengths = np.array([len(tokenizer.encode(sequence)) for _, sequence, *_ in dataset])
    tokenized_lengths = np.sort(tokenized_lengths)
    tokenized_lengths = tokenized_lengths[0:int(len(tokenized_lengths)*0.997)]

    # Count the occurrences of each token length
    unique_lengths, unique_counts = np.unique(tokenized_lengths, return_counts=True)

    counts = np.zeros(max(unique_lengths) + 1, dtype=int)

    for length, count in zip(unique_lengths, unique_counts):
        counts[length]=count

    return counts


def print_counts(counts, output_path):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Indices represent token counts, counts[i] = count of sequences with i tokens
    indices = np.arange(len(counts))

    # Plotting Barplot
    plt.figure(figsize=(12, 6))
    plt.bar(indices, counts, width=0.8, color='cornflowerblue', label='Bar Values', capsize=5)

    # Labels and Title for Barplot
    plt.title("Accuracy of Model vs. Dataset Size", fontsize=14)
    plt.xlabel("Dataset Size", fontsize=12)
    plt.ylabel("Mean Absolute Percentage Error", fontsize=12)

    # Legend
    plt.legend()

    # Save the barplot
    barplot_path = os.path.join(output_path, "barplot.jpg")
    plt.tight_layout()
    plt.savefig(barplot_path, dpi=300)
    plt.show()
    print(f"Barplot saved as {barplot_path}")

    # Adding Boxplot with Binning
    plt.figure(figsize=(12, 6))

    # Create bins with a width of 512
    residue = len(counts)%512
    max_bin = len(counts)+ (512 - residue)
    bin_edges = np.arange(0, max_bin, 512)
    binned_data = [counts[start:end] for start, end in zip(bin_edges[:-1], bin_edges[1:])]

    # Flatten data and filter out empty bins for boxplot
    binned_values = [np.sum(bin_group) for bin_group in binned_data if len(bin_group) > 0]

    # Plotting Barplot
    plt.figure(figsize=(12, 6))
    # Adjust x-axis to show all labels clearly
    plt.xticks(ticks=np.arange(len(binned_values)), labels=[str(x) for x in binned_values], rotation=45)

    plt.bar(np.arange(len(binned_values)), binned_values, width=0.8, color='cornflowerblue', label='Bar Values', capsize=5)

    # Labels and Title for Boxplot
    plt.title("Boxplot of Binned Data (512 Width)", fontsize=14)
    plt.xlabel("Bins (Width=512)", fontsize=12)
    plt.ylabel("Counts in Bins", fontsize=12)

    # Save the boxplot
    boxplot_path = os.path.join(output_path, "hist.jpg")
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=300)
    plt.show()
    print(f"Boxplot saved as {boxplot_path}")


