from data.dataloader import FastaClassificationDataset, FastaDataset, BalancedBatchSampler
from torch.utils.data import Dataset, BatchSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import torch
from transformers import pipeline, AutoTokenizer
from common.utils import pathways_to_class_mapping
import numpy as np
from models.train import train_cls
from models.classifier.heads import ClassificationHead, ClsAttn
import torch.optim as optim
from common.utils import count_tokens, print_counts
from config import Config
import os

from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mapping = pathways_to_class_mapping(path=os.path.join(Config.OUTPUT_PATH,"data"), input_names=["R-HSA-metabolism"])

    dataset = FastaDataset(file_path=Config.DATASET, class_mapping=mapping, data_path=os.path.join(Config.OUTPUT_PATH, "data/prot_gpt2_multi.pt"))
    # Create the balanced batch sampler
    sampler = BalancedBatchSampler(dataset) #removing sampler kills perfomace completely, even if weights are preserved
    # Create DataLoader with sampler
    dataloader = DataLoader(dataset, batch_size=64, drop_last=False, pin_memory=str(device) == "cuda:0", pin_memory_device=str(device), sampler=sampler)

    labels = dataset.labels
    weights = torch.tensor(np.sqrt(compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)), dtype=torch.float).to(device)

    cls_head = ClsAttn(1280, 256, 2, 0.5)
    optimizer = optim.Adam(cls_head.parameters(), lr=1e-4)#, weight_decay=1e-4)
    cls = train_cls(cls_head, dataloader, None, optimizer, weights, 50, "test", str(device))

    with torch.no_grad():
        embeddings = dataset.embeddings.to(device)

        outputs = cls(embeddings)  # Forward pass
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax
        predicted_classes = torch.argmax(probabilities, dim=1)  # Get predicted class indices

        predicted_classes = predicted_classes.cpu().numpy()

        print(precision_recall_fscore_support(labels, predicted_classes, average='macro'))
        print("\n\n\n")
        print(confusion_matrix(labels, predicted_classes))



