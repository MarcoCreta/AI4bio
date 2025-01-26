from data.dataloader import FastaClassificationDataset, FastaDataset, BalancedBatchSampler
from torch.utils.data import Dataset, BatchSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from common.utils import pathways_to_class_mapping
import numpy as np
import re

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def precompute_prot_bert(device):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, padding=True, truncation=True, model_max_length=512, return_tensors="pt")
    extractor = pipeline("feature-extraction", model="Rostlab/prot_bert", tokenizer=tokenizer, device=device)
    dataset = FastaDataset(file_path="/homes/mcreta/AI4bio/UP000005640_9606.fasta")

    dataloader = DataLoader(dataset, batch_size=32, drop_last=False, pin_memory=str(device) == "cuda:0", pin_memory_device=str(device))# , sampler=sampler)

    embeddings = []
    for batch in tqdm(dataloader, desc="batch"):
        # Unpack the batch tuple returned by DataLoader

        result = extractor(batch[1])

        embeddings.extend([e[0] for e in result])

    torch_embeddings = torch.tensor(embeddings, dtype=torch.float32)  # Convert to tensor
    torch.save(torch_embeddings, "prot_bert.pt")  # Save the tensor


from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def precompute_prot_gpt2(device):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "nferruz/ProtGPT2",
        do_lower_case=False,
        padding=True,
        truncation=True,
        model_max_length=512,
        return_tensors="pt"
    )
    model = AutoModel.from_pretrained("nferruz/ProtGPT2").to(device)
    model.eval()  # Set model to evaluation mode

    # Load dataset and dataloader
    dataset = FastaDataset(file_path="/homes/mcreta/AI4bio/UP000005640_9606.fasta")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        pin_memory=(str(device).startswith("cuda")),
        pin_memory_device=str(device)
    )

    embeddings = []
    for i, instance in tqdm(enumerate(dataloader), desc="batch"):

        inputs = tokenizer(instance[1], truncation=True, max_length=512)
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items() if k in tokenizer.model_input_names}

        # Perform forward pass to get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract embeddings from the last hidden state
        last_hidden_state = outputs.last_hidden_state.cpu()  # Move to CPU for storage
        embeddings.extend([embedding[0].numpy() for embedding in last_hidden_state])  # Use the first token's embedding

    # Convert embeddings to tensor and save
    torch_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    torch.save(torch_embeddings, "/homes/mcreta/AI4bio/prot_gpt2.pt")


def precompute_prot_t5(device):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_bfd",
        do_lower_case=False,
        padding=True,
        truncation=True,
        model_max_length=512,
        return_tensors="pt"
    )
    model = AutoModel.from_pretrained("Rostlab/prot_t5_xl_bfd").to(device)
    model.eval()  # Set model to evaluation mode

    # Load dataset and dataloader
    dataset = FastaDataset(file_path="/homes/mcreta/AI4bio/UP000005640_9606.fasta")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        pin_memory=(str(device).startswith("cuda")),
        pin_memory_device=str(device)
    )

    embeddings = []
    for i, instance in tqdm(enumerate(dataloader), desc="batch"):

        sequence = re.sub(r"[UZOB]", "X", instance[1])
        inputs = tokenizer(sequence, truncation=True, max_length=512, add_special_tokens=True)
        inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items() if k in tokenizer.model_input_names}

        # Perform forward pass to get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract embeddings from the last hidden state
        last_hidden_state = outputs.last_hidden_state.cpu()  # Move to CPU for storage
        #using embeddings coming from encoder only as stated in specifications for feature extraction
        embeddings.extend([embedding[2].numpy() for embedding in last_hidden_state])  # Use the first token's embedding

    # Convert embeddings to tensor and save
    torch_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    torch.save(torch_embeddings, "prot_t5.pt")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    precompute_prot_gpt2(device)

    print("dataset precomputed successfully")