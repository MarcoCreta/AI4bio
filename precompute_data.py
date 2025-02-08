from data.dataloader import FastaClassificationDataset, FastaDataset, BalancedBatchSampler
from torch.utils.data import Dataset, BatchSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from common.utils import pathways_to_class_mapping
import numpy as np
import re
from config import Config

import os

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

def precompute_prot_bert(device):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, padding=True, truncation=True, model_max_length=512, return_tensors="pt")
    extractor = pipeline("feature-extraction", model="Rostlab/prot_bert", tokenizer=tokenizer, device=device)
    dataset = FastaDataset(file_path=Config.DATASET)

    dataloader = DataLoader(dataset, batch_size=32, drop_last=False, pin_memory=str(device) == "cuda:0", pin_memory_device=str(device))# , sampler=sampler)

    embeddings = []
    for batch in tqdm(dataloader, desc="batch"):
        # Unpack the batch tuple returned by DataLoader

        result = extractor(batch[1])

        embeddings.extend([e[0] for e in result])

    torch_embeddings = torch.tensor(embeddings, dtype=torch.float32)  # Convert to tensor
    torch.save(torch_embeddings, os.path.join(Config.OUTPUT_PATH,"data/prot_bert.pt"))  # Save the tensor


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
        truncation=False,
        model_max_length=1536,  # Adjusted max length
        return_tensors="pt"
    )
    model = AutoModel.from_pretrained("nferruz/ProtGPT2").to(device)
    model.eval()  # Set model to evaluation mode

    # Load dataset and dataloader
    dataset = FastaDataset(file_path=Config.DATASET)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        pin_memory=(str(device).startswith("cuda")),
        pin_memory_device=str(device)
    )

    window_size = 512
    shift_size = 256
    max_windows = 5

    embeddings = []
    for i, instance in tqdm(enumerate(dataloader), desc="batch"):
        sequence = instance[1][0]  # Extract sequence
        tokenized_inputs = tokenizer(sequence, truncation=True, max_length=1536, add_special_tokens=True)
        input_ids = tokenized_inputs["input_ids"]
        seq_length = len(input_ids)

        sequence_embeddings = []
        for shift in range(max_windows):
            start_idx = shift * shift_size
            end_idx = start_idx + window_size
            if start_idx >= seq_length:
                break  # Stop if the window exceeds sequence length

            sub_input_ids = input_ids[start_idx:end_idx]
            inputs = {
                "input_ids": torch.tensor([sub_input_ids]).to(device),
                "attention_mask": torch.tensor(tokenized_inputs['attention_mask'][start_idx:end_idx]).to(device),
            }

            # Perform forward pass to get embeddings
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except:
                print(f"error, \n index:{i} \nseq_length{len(input_ids)} \n indices:[{start_idx}:{end_idx}] \n frag_len:{len(sub_input_ids)}\n name:{instance[2][0]} \n sequence:{instance[1][0]}")

            # Extract embeddings from the last hidden state
            last_hidden_state = outputs.last_hidden_state.cpu()
            sequence_embeddings.append(last_hidden_state[0][0].numpy())  # Use the first token's embedding

        # Zero-pad if fewer than 5 embeddings are obtained
        while len(sequence_embeddings) < max_windows:
            sequence_embeddings.append(np.zeros_like(sequence_embeddings[0]))

        embeddings.append(sequence_embeddings)

    # Convert embeddings to tensor and save
    torch_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    torch.save(torch_embeddings, os.path.join(Config.OUTPUT_PATH, "data/prot_gpt2_multi.pt"))


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
    dataset = FastaDataset(file_path=Config.DATASET)
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
    torch.save(torch_embeddings, os.path.join(Config.OUTPUT_PATH, "data/prot_t5.pt"))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    precompute_prot_gpt2(device)

    print("dataset precomputed successfully")