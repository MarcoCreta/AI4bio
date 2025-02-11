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

def precompute_prot_bert(device, checkpoint_interval=100):
    print("precomputing protein bert")

    # Define file paths for checkpoint and final output.
    checkpoint_path = os.path.join(Config.OUTPUT_PATH, "data", "prot_bert_multi_checkpoint.pt")
    final_output_path = os.path.join(Config.OUTPUT_PATH, "data", "prot_bert_multi.pt")

    # Initialize tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(
        "Rostlab/prot_bert",
        do_lower_case=False,
        padding=True,
        return_tensors="pt",
        model_max_length=512,
    )
    model = AutoModel.from_pretrained("Rostlab/prot_bert").to(device)
    model.eval()

    # Load dataset and create DataLoader (FastaDataset and Config must be defined)
    dataset = FastaDataset(file_path=Config.DATASET)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        pin_memory=str(device).startswith("cuda"),
        pin_memory_device=str(device)
    )

    # Define window parameters.
    max_seq_length = 512
    usable_length = max_seq_length - 2  # Reserve two tokens for [CLS] and [SEP]
    shift_size = (256 - 2)
    max_windows = 5

    # If a checkpoint exists, load it and set the starting index accordingly.
    embeddings = []
    start_index = 0
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        checkpoint = torch.load(checkpoint_path)
        # Assume checkpoint was saved as a tensor; convert it to a list.
        if isinstance(checkpoint, torch.Tensor):
            embeddings = checkpoint.numpy().tolist()
        else:
            embeddings = checkpoint
        start_index = len(embeddings)
        print(f"Resuming from checkpoint. Starting at iteration {start_index}.")

    # Iterate over the dataset, skipping examples already processed.
    for i, instance in tqdm(enumerate(dataloader), desc="batch"):
        if i < start_index:
            continue

        sequence = instance[1][0]
        sequence = list(sequence)
        # Limit sequence to a maximum number of characters.
        if len(sequence) > 1536:
            sequence = sequence[:1536]
        sequence_embeddings = []
        seq_length = len(sequence)

        for shift in range(max_windows):
            start_idx = shift * shift_size
            end_idx = start_idx + usable_length
            if start_idx >= seq_length:
                break

            # Extract a window of characters.
            sub_input_ids = sequence[start_idx:end_idx]
            # Preprocess the window: join characters with spaces.
            spaced_sequence = " ".join(sub_input_ids)

            # Tokenize with special tokens added automatically.
            tokenized_inputs = tokenizer(
                spaced_sequence,
                truncation=True,
                add_special_tokens=True,
                padding=True,
                max_length=512
            )
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]

            inputs = {
                "input_ids": torch.tensor([input_ids]).to(device),
                "attention_mask": torch.tensor([attention_mask]).to(device),
            }

            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except Exception as e:
                print(f"error, \n index:{i} \n seq_length:{seq_length} \n indices:[{start_idx}:{end_idx}]"
                      f"\n frag_len:{len(sub_input_ids)} \n name:{instance[2][0]} \n sequence:{sequence}")
                continue

            # Extract the embedding for the first token (typically [CLS]).
            last_hidden_state = outputs.last_hidden_state.cpu()
            sequence_embeddings.append(last_hidden_state[0][0].numpy())

        # If no window produced an embedding, use a zero vector.
        if len(sequence_embeddings) == 0:
            hidden_size = model.config.hidden_size
            sequence_embeddings.append(np.zeros(hidden_size))

        # Zero-pad to ensure we have max_windows embeddings per sequence.
        while len(sequence_embeddings) < max_windows:
            sequence_embeddings.append(np.zeros_like(sequence_embeddings[0]))

        embeddings.append(sequence_embeddings)

        # Save checkpoint every checkpoint_interval iterations.
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)
            torch.save(checkpoint_tensor, checkpoint_path)
            print(f"Checkpoint saved at iteration {i + 1}.")

    # Save the final tensor.
    torch_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    torch.save(torch_embeddings, final_output_path)
    print(f"Dataset precomputed successfully and saved in: {final_output_path}")


def precompute_prot_gpt2(device):
    print("precomputing protein_gpt2")
    # Load tokenizer and model for ProtGPT2.
    # ProtGPT2 uses "<|endoftext|>" for bos, eos, and unk.
    tokenizer = AutoTokenizer.from_pretrained(
        "nferruz/ProtGPT2",
        do_lower_case=False,
        padding=True,
        truncation=False,
        model_max_length=1536,
        return_tensors="pt"
    )
    model = AutoModel.from_pretrained("nferruz/ProtGPT2").to(device)
    model.eval()  # Set model to evaluation mode

    # For ProtGPT2, the only special token is "<|endoftext|>".
    # If no pad token is set, assign it to eos_token (which is "<|endoftext|>").
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id   # Typically 0
    bos_token_id = tokenizer.bos_token_id     # Also 0

    # Load dataset and create DataLoader (assumes FastaDataset and Config are defined)
    dataset = FastaDataset(file_path=Config.DATASET)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        pin_memory=str(device).startswith("cuda"),
        pin_memory_device=str(device)
    )

    # Define window parameters in token space.
    # Maximum input length is 512 tokens.
    # Since we manually prepend the bos token, we can use 511 tokens from the tokenized sequence.
    max_seq_length = 512
    usable_length = max_seq_length - 1  # Reserve one slot for the bos token.
    shift_size = usable_length // 2     # 50% overlap (for example, 255 tokens)
    max_windows = 5

    embeddings = []
    for i, instance in tqdm(enumerate(dataloader), desc="batch"):
        sequence = instance[1][0]  # Extract the raw protein sequence

        # Tokenize the full sequence without adding any special tokens.
        tokenized_inputs = tokenizer(sequence, truncation=True, max_length=1536, add_special_tokens=False)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        seq_length = len(input_ids)

        sequence_embeddings = []
        for window in range(max_windows):
            start_idx = window * shift_size
            end_idx = start_idx + usable_length
            if start_idx >= seq_length:
                break

            sub_input_ids = input_ids[start_idx:end_idx]
            sub_attention_mask = attention_mask[start_idx:end_idx]

            # Prepend the bos token manually.
            window_ids = [bos_token_id] + sub_input_ids
            window_attention_mask = [1] + sub_attention_mask

            inputs = {
                "input_ids": torch.tensor([window_ids]).to(device),
                "attention_mask": torch.tensor([window_attention_mask]).to(device),
            }

            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except Exception as e:
                print(f"Error at index {i}: seq_length {seq_length}, indices [{start_idx}:{end_idx}], "
                      f"fragment length {len(sub_input_ids)}; sequence: {sequence}")
                continue

            # For classification, extract the embedding corresponding to the first token (bos token).
            #last_hidden_state = outputs.last_hidden_state.cpu()

            #sequence_embeddings.append(last_hidden_state[0][0].numpy())

            sequence_embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0])


        # If no window produced an embedding, create a zero vector with the model's hidden size.
        if len(sequence_embeddings) == 0:
            hidden_size = model.config.hidden_size
            sequence_embeddings.append(np.zeros(hidden_size))
        # Zero-pad if fewer than max_windows embeddings were produced.
        while len(sequence_embeddings) < max_windows:
            sequence_embeddings.append(np.zeros_like(sequence_embeddings[0]))

        embeddings.append(sequence_embeddings)

    # Convert the collected embeddings to a tensor and save.
    torch_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    output_path = os.path.join(Config.OUTPUT_PATH, "data/prot_gpt2_multi.pt")
    torch.save(torch_embeddings, output_path)
    print(f"Dataset precomputed successfully and saved in: {output_path}")


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