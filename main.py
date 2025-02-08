from data.dataloader import FastaClassificationDataset, FastaDataset, BalancedBatchSampler
from torch.utils.data import Dataset, BatchSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
import torch
from transformers import pipeline, AutoTokenizer
from common.utils import pathways_to_class_mapping
import numpy as np
from models.train import train_cls, fusion_functions
from models.classifier.heads import ClassificationHead, ClsAttn
import torch.optim as optim
from common.utils import count_tokens, print_counts
from config import Config
import os
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    np.random.seed(Config.RANDOMNESS["NUMPY_SEED"])
    torch.cuda.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    random.seed(Config.RANDOMNESS["PYTHON_SEED"])

    # Ensure deterministic behavior in CUDA (may slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = Config.RANDOMNESS["PYTORCH_SEED"] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])




    mapping = pathways_to_class_mapping(path=os.path.join(Config.OUTPUT_PATH,"data"), input_names=Config.CLASSES)

    dataset = FastaDataset(file_path=Config.DATASET, class_mapping=mapping, data_path=os.path.join(Config.OUTPUT_PATH, Config.DATA_NAME))
    # Create the balanced batch sampler
    sampler = BalancedBatchSampler(dataset) #removing sampler kills perfomace completely, even if weights are preserved
    # Create DataLoader with sampler
    dataloader = DataLoader(
        dataset,
        batch_size=Config.TRAIN_ARGS['BATCH_SIZE'],
        drop_last=False,
        pin_memory=str(device) == "cuda:0",
        pin_memory_device=str(device),
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g)

    labels = dataset.labels
    weights = torch.tensor(np.sqrt(compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)), dtype=torch.float).to(device)

    cls_head = ClsAttn(Config.FF_ARGS['INPUT_SIZE'], Config.FF_ARGS['HIDDEN_SIZE'], Config.FF_ARGS['OUTPUT_SIZE'], Config.FF_ARGS['DROPOUT'])

    optimizer = optim.Adam(
        cls_head.parameters(),
        lr=Config.TRAIN_ARGS['LEARNING_RATE'],
        weight_decay=Config.TRAIN_ARGS['WEIGHT_DECAY']
    )

    cls = train_cls(
        cls_head,
        dataloader,
        None,
        optimizer,
        weights,
        Config.TRAIN_ARGS['N_EPOCHS'],
        Config.TRAIN_ARGS['EMB_CHUNKS'],
        fusion_functions[Config.TRAIN_ARGS['EMB_FUSION_FN']] if Config.TRAIN_ARGS['EMB_FUSION_FN'] else None,
        "test",
        str(device))

    predicted_classes = cls.inference(dataset.embeddings.to(device))

    print(precision_recall_fscore_support(labels, predicted_classes, average='macro'))
    print("\n\n\n")
    print(confusion_matrix(labels, predicted_classes))



