from data.dataloader import FastaDataset, BalancedBatchSampler
import torch
from common.utils import pathways_to_class_mapping
import numpy as np
from models.train import train_cls
from models.classifier.heads import ClassificationHead, ClsAttn
from config import Config
import os
import random
from common.utils import compute_multilabel_class_weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pprint
import pandas as pd
from common.utils import multihot_to_int, initialize_weights


def compute_per_class_scores(cf, class_names, labels_info):

    for label_int, label_bin, class_list in labels_info:
        print(f"numeric_label : {label_int}, binary_label:{label_bin}, classes:{class_list}")

    # Initialize per-class (atomic) counts.
    per_class = {i: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for i in range(len(class_names))}

    # Loop over each cell in the confusion matrix.
    # i indexes the true combination; j indexes the predicted combination.
    for i in range(cf.shape[0]):
        true_vector = labels_info[i][1]  # the multi-hot binary vector for the true label combination
        for j in range(cf.shape[1]):
            pred_vector = labels_info[j][1]  # the multi-hot binary vector for the predicted label combination
            count = cf[i, j]
            # For each atomic class, update counts.
            for k in range(len(class_names)):
                if true_vector[k] == 1 and pred_vector[k] == 1:
                    per_class[k]["TP"] += count
                elif true_vector[k] == 0 and pred_vector[k] == 1:
                    per_class[k]["FP"] += count
                elif true_vector[k] == 1 and pred_vector[k] == 0:
                    per_class[k]["FN"] += count
                else:  # true_vector[k] == 0 and pred_vector[k] == 0
                    per_class[k]["TN"] += count

    # Now compute precision, recall, and F1-score for each atomic class.
    results = {}
    for k in range(len(class_names)):
        counts = per_class[k]
        TP = counts["TP"]
        FP = counts["FP"]
        FN = counts["FN"]
        TN = counts["TN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results[class_names[k]] = {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "Precision": precision, "Recall": recall, "F1": f1
        }

    # Create a DataFrame from the results dictionary.
    df = pd.DataFrame(results).T
    df = df[['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1']]

    df[['Precision', 'Recall', 'F1']] = df[['Precision', 'Recall', 'F1']].round(3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)
    print(df)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    np.random.seed(Config.RANDOMNESS["NUMPY_SEED"])
    torch.cuda.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])
    random.seed(Config.RANDOMNESS["PYTHON_SEED"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = Config.RANDOMNESS["PYTORCH_SEED"] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(Config.RANDOMNESS["PYTORCH_SEED"])

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
    #unique labelling for each class combination
    combination_labels = np.array([multihot_to_int(label) for label in onehot_labels_np])
    # Select IDs that appear more than 10 times
    unique_ids, counts = np.unique(combination_labels, return_counts=True)
    combination_labels = combination_labels * np.isin(combination_labels, unique_ids[counts > 3])

    #perform stratified splitting using these "fake" classes.
    indices = np.arange(len(dataset))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=combination_labels,
        random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.25,
        stratify=combination_labels[train_val_indices],
        random_state=42
    )

    # Create subset datasets.
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders.
    # For training, we use a BalancedBatchSampler on the training subset.
    train_sampler = BalancedBatchSampler(train_dataset, Config.USE_DEFAULT)
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

    # Build the classifier head.
    cls_head = ClsAttn(
        Config.FF_ARGS['INPUT_SIZE'],
        Config.FF_ARGS['HIDDEN_SIZE'],
        Config.FF_ARGS['OUTPUT_SIZE'],
        Config.FF_ARGS['DROPOUT'],
        Config.TRAIN_ARGS['EMB_CHUNKS'],
    )

    # Compute class weights based on the entire dataset’s one-hot labels.
    weights = torch.tensor(
        compute_multilabel_class_weights(onehot_labels, np.sqrt, norm=True), dtype=torch.float
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
        val_dataloader,
        optimizer,
        weights,
        Config.TRAIN_ARGS['GAMMA'],
        Config.TRAIN_ARGS['N_EPOCHS'],
        Config.TRAIN_ARGS['EMB_CHUNKS'],
        None,
        "test",
        str(device)
    )

    #inference on validation dataset
    embeddings = test_dataset.dataset.embeddings[test_dataset.indices]
    onehot_labels =test_dataset.dataset.get_labels()[test_dataset.indices]

    embeddings = embeddings[:,:Config.TRAIN_ARGS['EMB_CHUNKS'],:]
    if not Config.TRAIN_ARGS['ATTN_POOLING']:
        embeddings = torch.flatten(embeddings, start_dim=1)

    predicted_classes = cls.inference(embeddings.to(device), Config.FF_ARGS['THRESHOLD'])

    #compute scikit statistics
    labels_np = onehot_labels.clone().cpu().numpy()
    class_names = Config.CLASSES
    report = classification_report(labels_np, predicted_classes, target_names=class_names)
    print("\nPer-Class Scores:\n", report)

    #compute custom fine-grained counts
    combination_true = np.array([multihot_to_int(label) for label in labels_np.astype(int)])
    combination_pred = np.array([multihot_to_int(label) for label in predicted_classes.astype(int)])
    cf_labels_int, cf_labels_idx = np.unique(combination_labels, return_index=True)
    cf_labels_bin = onehot_labels_np[cf_labels_idx]
    labels_info = [(cf_labels_int[i], cf_labels_bin[i], np.array(class_names)[cf_labels_bin[i] == 1]) for i in range(len(cf_labels_int))]

    cf = confusion_matrix(combination_true, combination_pred, labels=cf_labels_int)
    df_cf = pd.DataFrame(cf, index=cf_labels_int, columns=cf_labels_int)
    print(df_cf)

    compute_per_class_scores(cf, class_names, labels_info)

