from data.dataloader import FastaClassificationDataset, FastaDataset, BalancedBatchSampler
from torch.utils.data import Dataset, BatchSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import torch
from transformers import pipeline, AutoTokenizer
from common.utils import pathways_to_class_mapping
import numpy as np
from models.train import train_cls
from models.classifier.heads import ClassificationHead
import torch.optim as optim


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mapping = pathways_to_class_mapping(path="/homes/mcreta/AI4bio", input_names=["R-HSA-109581"])

    dataset = FastaDataset(file_path="/homes/mcreta/AI4bio/UP000005640_9606.fasta", class_mapping=mapping, data_path="/homes/mcreta/AI4bio/prot_gpt2.pt")
    # Create the balanced batch sampler
    sampler = BalancedBatchSampler(dataset)
    # Create DataLoader with sampler
    dataloader = DataLoader(dataset, batch_size=64, drop_last=False, pin_memory=str(device) == "cuda:0", pin_memory_device=str(device), sampler=sampler)

    cls_head = ClassificationHead(1280, 256, 1)
    optimizer = optim.Adam(cls_head.parameters(), lr=5e-4)
    train_cls(cls_head, dataloader, None, optimizer, 100, "test", str(device))