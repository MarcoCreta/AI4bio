from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, BatchSampler, DataLoader, WeightedRandomSampler
from typing import Dict, Optional
from collections import OrderedDict, Counter
from common.utils import read_fasta
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

from torch.utils.data import Dataset
import torch
import os
from Bio import SeqIO
from collections import OrderedDict
from typing import Optional, Dict, List

class FastaDataset(Dataset):
    def __init__(self, file_path: str, data_path: Optional[str] = None, class_mapping: Optional[Dict[str, List[str]]] = None):
        self.sequences = []
        self.identifiers = []
        self.embeddings = []
        self.labels = []         # Store multi-class labels
        self.onehot_labels = []  # Store multi-hot encoded labels

        # Map pathways to numeric class indices (OrderedDict ensures consistency)
        self.pathway_to_class = OrderedDict()
        if class_mapping:
            tmp = [('default',0)] #set default class to 0
            tmp.extend([(pathway, idx+1) for idx, pathway in enumerate(class_mapping.keys())])
            self.pathway_to_class = OrderedDict(tmp)

            # Reverse mapping: protein ID -> list of class indices
            reverse_mapping = {}
            for pathway, proteins in class_mapping.items():
                for protein in proteins:
                    if protein not in reverse_mapping:
                        reverse_mapping[protein] = []
                    reverse_mapping[protein].append(self.pathway_to_class.get(pathway, 0))
            reverse_mapping = {key:np.unique(reverse_mapping[key]) for key in reverse_mapping}

        try:
            for record in SeqIO.parse(file_path, "fasta"):
                self.sequences.append(str(record.seq))  # Protein sequence
                identifier = record.id.split("|")[-1]  # Extract identifier from FASTA header
                self.identifiers.append(identifier)

                if class_mapping:
                    # Assign list of class indices (multi-label support)
                    label_indices = reverse_mapping.get(identifier, [0])
                    self.labels.append(label_indices)

        except Exception as e:
            print(f"Error during initialization: {e}")

        # Create one-hot encoded labels
        num_classes = len(self.pathway_to_class)
        for label_indices in self.labels:
            onehot = torch.zeros(num_classes, dtype=torch.float)
            onehot[label_indices] = 1  # Set indices for active classes
            self.onehot_labels.append(onehot)

        try:
            # Load precomputed embeddings if available
            if data_path and os.path.exists(data_path):
                self.embeddings = torch.load(data_path)
            else:
                self.embeddings = [None] * len(self.sequences)  # Placeholder for missing embeddings
        except Exception as e:
            print(f"Error loading precomputed data: {e}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        # Check if embeddings exist and provide default values if missing
        if self.embeddings is not None and idx < len(self.embeddings):
            embedding = self.embeddings[idx] if self.embeddings[idx] is not None else torch.zeros(1)
        else:
            embedding = torch.zeros(1)  # Default embedding

        # Get one-hot encoded label
        onehot_label = self.onehot_labels[idx] if self.onehot_labels and idx < len(self.onehot_labels) else torch.zeros(len(self.pathway_to_class))

        return idx, self.sequences[idx], self.identifiers[idx], onehot_label, embedding

    def get_labels(self):
        """Return all one-hot labels as a tensor."""
        return torch.stack(self.onehot_labels) if self.onehot_labels else None







#https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/6
class BalancedBatchSampler(WeightedRandomSampler):
    def __init__(self, dataset: FastaDataset):
        labels = dataset.get_labels()
        if labels is None:
            raise ValueError("Dataset must have labels for balanced sampling.")

        sample_weights = np.sqrt(compute_sample_weight(class_weight="balanced", y=labels))

        super().__init__(sample_weights, len(sample_weights))

from torch.utils.data import Sampler
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from common.utils import compute_multilabel_class_weights
from torch.utils.data import Sampler, Subset


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset: FastaDataset):
        """
        Balanced batch sampler for multi-label datasets.
        Args:
            dataset (FastaDataset or Subset): Dataset with one-hot encoded labels.
        """
        # If a Subset is passed, get the underlying dataset and select only the subset labels.
        if isinstance(dataset, Subset):
            # dataset.dataset is the original FastaDataset and dataset.indices are the selected indices.
            full_labels = dataset.dataset.get_labels()  # Tensor of shape (num_samples, num_classes)
            # Select only the labels for the subset.
            onehot_labels = full_labels[dataset.indices]
            self.indices = dataset.indices
        else:
            onehot_labels = dataset.get_labels()
            self.indices = list(range(len(dataset)))

        # Compute per-class weights.
        class_weights = compute_multilabel_class_weights(onehot_labels)

        # Compute per-sample weights based on their labels.
        sample_weights = (onehot_labels.cpu().numpy() * class_weights).sum(axis=1)
        sample_weights = sample_weights / sample_weights.sum()  # Normalize

        # Store sample weights as a tensor.
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float)

        # Use WeightedRandomSampler for balanced sampling.
        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return len(self.sample_weights)


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os

class FastaClassificationDataset(Dataset):
    def __init__(self, fasta_path, class_map, precomputed_path=None):
        """
        Args:
            fasta_path (str): Path to the FASTA file.
            class_map (dict): Mapping of sequence identifiers to class labels.
            precomputed_path (str): Path to the precomputed PyTorch data file (optional).
        """
        self.fasta_path = fasta_path
        self.precomputed_path = precomputed_path
        self.raw_data = []
        self.precomputed_data = None
        self.class_map = class_map

        # Initialize label encoder and fit it to the unique classes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(class_map.values()))

        # Load identifiers and raw sequences from the FASTA file
        self._load_fasta()

        # If a precomputed path is provided, load precomputed tensors
        if self.precomputed_path and os.path.exists(self.precomputed_path):
            self.precomputed_data = torch.load(self.precomputed_path)

    def _load_fasta(self):
        """Load raw data from the FASTA file."""
        with open(self.fasta_path, 'r') as f:
            identifier = None
            sequence = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save the previous sequence if it exists
                    if identifier:
                        self.raw_data.append((identifier, ''.join(sequence)))
                    identifier = line[1:]  # Remove the '>'
                    sequence = []
                else:
                    sequence.append(line)
            # Add the last sequence
            if identifier:
                self.raw_data.append((identifier, ''.join(sequence)))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        """
        Retrieve an item. Handles raw data and precomputed data differently.
        """
        identifier, sequence = self.raw_data[idx]
        label = self.class_map.get(identifier, None)
        if label is None:
            raise ValueError(f"Identifier '{identifier}' not found in class_map.")

        # Encode the label using the label encoder
        encoded_label = self.label_encoder.transform([label])[0]

        if self.precomputed_data is not None:
            # Use precomputed data if available
            data = self.precomputed_data[idx]
        else:
            # Otherwise, process raw data
            data = self._process_sequence(sequence)

        return {"identifier": identifier, "data": data, "label": encoded_label}

    def _process_sequence(self, sequence):
        """
        Converts a raw sequence into a tensor or desired representation.
        Example: Map 'A', 'C', 'G', 'T' to integers.
        """
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return torch.tensor([mapping[char] for char in sequence if char in mapping], dtype=torch.long)

    def get_class_labels(self):
        """Return the mapping of numeric labels to class names."""
        return dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
