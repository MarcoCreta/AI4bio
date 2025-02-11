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
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from common.utils import compute_multilabel_class_weights
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # Assume the following are defined:
    # - Config (with attributes: DATASET, OUTPUT_PATH, DATA_NAME, CLASSES, TRAIN_ARGS, etc.)
    # - FastaDataset
    # - pathways_to_class_mapping
    # - BalancedBatchSampler
    # - seed_worker, g (random seed generator)

    # Build the class mapping and create the full dataset.
    mapping = pathways_to_class_mapping(
        path=os.path.join(Config.OUTPUT_PATH, "data"),
        input_names=Config.CLASSES
    )
    dataset = FastaDataset(
        file_path=Config.DATASET,
        class_mapping=mapping,
        data_path=os.path.join(Config.OUTPUT_PATH, Config.DATA_NAME)
    )

    # Retrieve the multilabel (multi-hot) labels.
    onehot_labels = dataset.get_labels()  # shape: (num_samples, num_classes)
    # Convert the one-hot labels to NumPy and to integers (0/1)
    onehot_labels_np = onehot_labels.cpu().numpy().astype(int)


    # Convert each multi-hot vector to a unique identifier.
    # For example, treat each vector as a binary number.
    def multihot_to_int(label_vector):
        # label_vector is something like [1, 0, 1] which becomes '101'
        # Then convert to integer using base 2.
        binary_str = ''.join(str(x) for x in label_vector)
        return int(binary_str, 2)


    combination_labels = np.array([multihot_to_int(label) for label in onehot_labels_np])

    # Now perform stratified splitting using these "fake" classes.
    indices = np.arange(len(dataset))

    # Split into train_val and test (for example, 20% test)
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=combination_labels,
        random_state=42
    )

    # Now, split train_val into train and validation (for example, 25% of train_val for validation)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.25,  # 25% of train_val (i.e. 20% overall)
        stratify=combination_labels[train_val_indices],
        random_state=42
    )

    # Create subset datasets.
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders.
    # For training, we use a BalancedBatchSampler (if desired) on the training subset.
    train_sampler = BalancedBatchSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_ARGS['BATCH_SIZE'],
        drop_last=False,
        pin_memory=(str(device) == "cuda:0"),
        pin_memory_device=str(device),
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.TRAIN_ARGS['BATCH_SIZE'],
        shuffle=False,
        drop_last=False,
        pin_memory=(str(device) == "cuda:0"),
        pin_memory_device=str(device)
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=Config.TRAIN_ARGS['BATCH_SIZE'],
        shuffle=False,
        drop_last=False,
        pin_memory=(str(device) == "cuda:0"),
        pin_memory_device=str(device)
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    ###############################################
    # The rest of your pipeline follows below:
    ###############################################

    # Build the classifier head.
    cls_head = ClsAttn(
        Config.FF_ARGS['INPUT_SIZE'],
        Config.FF_ARGS['HIDDEN_SIZE'],
        Config.FF_ARGS['OUTPUT_SIZE'],
        Config.FF_ARGS['DROPOUT']
    )

    # Compute class weights based on the entire dataset’s one-hot labels.
    weights = torch.tensor(
        compute_multilabel_class_weights(onehot_labels), dtype=torch.float
    ).to(device)

    optimizer = torch.optim.Adam(
        cls_head.parameters(),
        lr=Config.TRAIN_ARGS['LEARNING_RATE'],
        weight_decay=Config.TRAIN_ARGS['WEIGHT_DECAY']
    )

    # Train the classifier using only the training dataloader.
    cls = train_cls(
        cls_head,
        train_dataloader,
        val_dataloader,  # Optionally, pass the validation dataloader to monitor validation performance.
        optimizer,
        weights,
        Config.TRAIN_ARGS['N_EPOCHS'],
        Config.TRAIN_ARGS['EMB_CHUNKS'],
        None,
        "test",
        str(device)
    )

    # For inference, assume the embeddings have been computed and stored in dataset.embeddings.
    # (You might need to adapt this part so that you compute embeddings for the test set only.)
    embeddings = test_dataset.dataset.embeddings[test_dataset.indices]
    onehot_labels =test_dataset.dataset.get_labels()[test_dataset.indices]
    if not Config.TRAIN_ARGS['ATTN_POOLING']:
        embeddings = embeddings[:, 0, :]

    # Get predictions (here, using the classifier on all embeddings).
    predicted_classes = cls.inference(embeddings.to(device))

    # Convert labels to NumPy (ensure labels are NumPy arrays).
    labels_np = onehot_labels.clone().cpu().numpy()

    # Compute per-class precision, recall, and F1-score.
    from sklearn.metrics import precision_recall_fscore_support, classification_report

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, predicted_classes, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels_np, predicted_classes, average='macro', zero_division=0
    )

    class_names = ["default"]
    class_names.extend(Config.CLASSES)
    report = classification_report(labels_np, predicted_classes, target_names=class_names)

    print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
    print("\nPer-Class Scores:\n", report)