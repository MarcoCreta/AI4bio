import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from common.utils import merge_embeddings
from models.classifier.losses import WeightedFocalLoss
from torch import autograd

def train_one_epoch(cls_head, train_loader, optimizer, criterion, emb_chunks, fusion, device, epoch):
    """Train the classification head for one epoch."""
    cls_head.train()
    total_train_loss = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for _, _, _, targets, embeddings in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()

            embeddings, targets = embeddings[:,:emb_chunks,:].to(device), targets.to(device)

            if not Config.TRAIN_ARGS['ATTN_POOLING']:
                embeddings = torch.flatten(embeddings, start_dim=1)

            with autograd.detect_anomaly():
                logits = cls_head(embeddings).to(device)  # Forward pass
                loss = criterion(logits, targets)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(cls_head.parameters(), max_norm=1.0)

                optimizer.step()

            total_train_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    return total_train_loss / len(train_loader)

def validate_one_epoch(cls_head, validation_loader, criterion, emb_chunks, fusion, device):
    """Validate the classification head for one epoch."""
    cls_head.eval()
    total_val_loss = 0

    with torch.no_grad():
        for _, _, _, targets, embeddings in validation_loader:
            embeddings, targets = embeddings[:, :emb_chunks, :].to(device), targets.to(device)
            if not Config.TRAIN_ARGS['ATTN_POOLING']:
                embeddings = torch.flatten(embeddings, start_dim=1)
            logits = cls_head(embeddings).to(device)  # Forward pass
            loss = criterion(logits, targets)
            total_val_loss += loss.item()

    return total_val_loss / len(validation_loader)

def plot_loss_curves(train_loss_history, val_loss_history, suffix, epoch, save_path):
    """Plot and save training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(val_loss_history, label='Validation Loss', color='red')
    plt.grid(True)
    plt.title(f'Training and Validation Loss Curve {suffix} at epoch: {epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(top=max(0.5, max(max(train_loss_history), max(val_loss_history))))
    plt.savefig(save_path)
    plt.close()


def train_cls(
        cls_head, train_loader, validation_loader, optimizer, weights=None, gamma=0, num_epochs=100, emb_chunks=None, fusion=None, suffix='test', device='cuda:0'
):
    """Train a classification head for a multi-label classification task."""

    cls_head.to(device)

    # Use Binary Cross Entropy with Logits for multi-label classification

    if Config.TRAIN_ARGS['LOSS'] == 'bce':
        if not Config.TRAIN_ARGS['WEIGHTS']:
            weights = torch.full_like(weights, 1).to(device).float()
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    elif Config.TRAIN_ARGS['LOSS'] == 'focal':
        weights = torch.full_like(weights, Config.TRAIN_ARGS['ALPHA']).to(device).float()
        criterion = WeightedFocalLoss(alpha=weights, gamma=gamma, reduction="mean")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(cls_head, train_loader, optimizer, criterion, emb_chunks, fusion, device, epoch)
        val_loss = validate_one_epoch(cls_head, validation_loader, criterion, emb_chunks, fusion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Save loss curves periodically
        if epoch % 20 == 0 and epoch != 0:
            plot_loss_curves(
                train_loss_history, val_loss_history, suffix, num_epochs,
                os.path.join(Config.OUTPUT_PATH, f'loss_curves/{suffix}_partial.png')
            )

    #Save final model and loss curves
    #torch.save(cls_head.state_dict(), os.path.join(Config.OUTPUT_PATH, f'weights/{suffix}.pth'))
    plot_loss_curves(
        train_loss_history, val_loss_history, suffix, num_epochs,
        os.path.join(Config.OUTPUT_PATH, f'loss_curves/{suffix}_final.png')
    )

    return cls_head