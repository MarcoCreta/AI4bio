from Bio import SeqIO
from typing import List, Optional
import json
from collections import OrderedDict
import os

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




