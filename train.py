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
from sklearn.model_selection import train_test_split, StratifiedKFold
from common.utils import set_random_seeds, seed_worker


def prepare_dataset():
    """Build the class mapping and create the full dataset along with label arrays."""
    mapping = pathways_to_class_mapping(
        path=os.path.join(Config.OUTPUT_PATH, "data"),
        input_names=Config.CLASSES
    )
    dataset = FastaDataset(
        file_path=Config.DATASET,
        class_mapping=mapping,
        data_path=os.path.join(Config.OUTPUT_PATH, Config.DATA_NAME)
    )
    onehot_labels = dataset.get_labels()  # Tensor of shape (num_samples, num_classes)
    onehot_labels_np = onehot_labels.cpu().numpy().astype(int)

    # Convert multi-hot vectors to unique integer identifiers by treating them as binary numbers.
    def multihot_to_int(label_vector):
        binary_str = ''.join(str(x) for x in label_vector)
        return int(binary_str, 2)

    combination_labels = np.array([multihot_to_int(label) for label in onehot_labels_np])
    indices = np.arange(len(dataset))
    return dataset, onehot_labels, combination_labels, indices


def get_data_loaders_for_fold(train_indices, val_indices, dataset, config, device, seed_worker, g):
    """Create DataLoaders for training and validation for a given fold."""
    train_dataset_fold = Subset(dataset, train_indices)
    val_dataset_fold = Subset(dataset, val_indices)

    train_sampler = BalancedBatchSampler(train_dataset_fold, Config.USE_DEFAULT)
    train_dataloader = DataLoader(
        train_dataset_fold,
        batch_size=config.TRAIN_ARGS['BATCH_SIZE'],
        drop_last=False,
        pin_memory=(str(device) == "cuda:0"),
        pin_memory_device=str(device),
        sampler=train_sampler,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_dataloader = DataLoader(
        val_dataset_fold,
        batch_size=config.TRAIN_ARGS['BATCH_SIZE'],
        shuffle=False,
        drop_last=False,
        pin_memory=(str(device) == "cuda:0"),
        pin_memory_device=str(device)
    )
    return train_dataloader, val_dataloader, train_dataset_fold, val_dataset_fold


def split_dataset(indices, combination_labels, test_size=0.2, random_state=42):
    """Split indices into a hold-out test set and train+val set, using stratification."""
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=combination_labels,
        random_state=random_state
    )
    return train_val_indices, test_indices


def train_and_evaluate_fold(fold, train_indices, val_indices, dataset, config, device, seed_worker, g):
    """Train the classifier on one fold and evaluate its performance on the validation subset."""
    print(f"\n========== Fold {fold + 1} ==========")
    train_dataloader, val_dataloader, train_dataset_fold, val_dataset_fold = get_data_loaders_for_fold(
        train_indices, val_indices, dataset, config, device, seed_worker, g
    )
    print(f"Fold {fold + 1} - Train samples: {len(train_dataset_fold)}, Val samples: {len(val_dataset_fold)}")

    # Build the classifier head.
    cls_head = ClsAttn(
        config.FF_ARGS['INPUT_SIZE'],
        config.FF_ARGS['HIDDEN_SIZE'],
        config.FF_ARGS['OUTPUT_SIZE'],
        config.FF_ARGS['DROPOUT'],
        Config.TRAIN_ARGS['EMB_CHUNKS'],
    )

    # Compute class weights (using the full dataset labels in this example).

    weights = torch.tensor(
        compute_multilabel_class_weights(dataset.get_labels(), np.sqrt, norm=True),
        dtype=torch.float
    ).to(device)

    optimizer = torch.optim.Adam(
        cls_head.parameters(),
        lr=config.TRAIN_ARGS['LEARNING_RATE'],
        weight_decay=config.TRAIN_ARGS['WEIGHT_DECAY']
    )

    # Train the classifier for this fold.
    cls = train_cls(
        cls_head,
        train_dataloader,
        val_dataloader,
        optimizer,
        weights,
        config.TRAIN_ARGS['N_EPOCHS'],
        config.TRAIN_ARGS['EMB_CHUNKS'],
        None,
        "fold_validation",
        str(device)
    )

    # Evaluate on the validation fold.
    # Here, we assume dataset.embeddings has been precomputed and corresponds to each sample.
    # Select embeddings for the current validation fold.
    actual_val_indices = val_indices
    embeddings_val = dataset.embeddings[actual_val_indices]
    if not config.TRAIN_ARGS['ATTN_POOLING']:
        embeddings_val = torch.flatten(embeddings_val[:,:Config.TRAIN_ARGS['EMB_CHUNKS'],:], start_dim=1)

    predicted_classes = cls.inference(embeddings_val.to(device), Config.FF_ARGS['THRESHOLD'])
    labels_val = dataset.get_labels()[actual_val_indices]
    labels_np = labels_val.clone().cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, predicted_classes, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels_np, predicted_classes, average='macro', zero_division=0
    )

    print(
        f"Fold {fold + 1} - Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
    report = classification_report(labels_np, predicted_classes, target_names=config.CLASSES)
    print(f"Fold {fold + 1} Classification Report:\n{report}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


def perform_kfold_cv(dataset, train_val_indices, combination_labels, config, device, seed_worker, g):
    """Perform stratified k-fold cross validation on the train_val set."""
    train_val_combination_labels = combination_labels[train_val_indices]
    n_folds = config.TRAIN_ARGS.get("N_FOLDS", 5)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_combination_labels)):
        # Map fold indices back to actual dataset indices.
        actual_train_indices = train_val_indices[train_idx]
        actual_val_indices = train_val_indices[val_idx]
        metrics = train_and_evaluate_fold(fold, actual_train_indices, actual_val_indices, dataset, config, device,
                                          seed_worker, g)
        fold_metrics.append(metrics)

    macro_precisions = [m["macro_precision"] for m in fold_metrics]
    macro_recalls = [m["macro_recall"] for m in fold_metrics]
    macro_f1s = [m["macro_f1"] for m in fold_metrics]

    avg_macro_precision = np.mean(macro_precisions)
    avg_macro_recall = np.mean(macro_recalls)
    avg_macro_f1 = np.mean(macro_f1s)

    print("\n========== Cross Validation Results ==========")
    print(f"Average Macro Precision: {avg_macro_precision:.4f}")
    print(f"Average Macro Recall:    {avg_macro_recall:.4f}")
    print(f"Average Macro F1:        {avg_macro_f1:.4f}")

    return fold_metrics

def evaluate_holdout(dataset, test_indices, config, device):
    """(Optional) Evaluate on the hold-out test set."""
    test_dataset = Subset(dataset, test_indices)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN_ARGS['BATCH_SIZE'],
        shuffle=False,
        drop_last=False,
        pin_memory=(str(device) == "cuda:0"),
        pin_memory_device=str(device)
    )
    print(f"\nHold-out test set samples: {len(test_dataset)}")
    # Here you might retrain a final model on all train_val data and then compute predictions on the test set.
    # For demonstration, we assume that test evaluation is handled separately.
    print("Test evaluation is not implemented in this snippet.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = set_random_seeds()

    # Prepare the full dataset and associated label arrays.
    dataset, onehot_labels, combination_labels, indices = prepare_dataset()


    # Split into a hold-out test set and train+val set.
    train_val_indices, test_indices = split_dataset(indices, combination_labels, test_size=0.2, random_state=42)

    print(f"Total samples: {len(dataset)}")
    print(f"Train+Val: {len(train_val_indices)}, Test: {len(test_indices)}")

    # Perform k-fold cross validation on the train+val set.
    fold_metrics = perform_kfold_cv(dataset, train_val_indices, combination_labels, Config, device, seed_worker, g)

    # Optionally, evaluate on the hold-out test set.
    #evaluate_holdout(dataset, test_indices, Config, device)

if __name__ == "__main__":
    main()

