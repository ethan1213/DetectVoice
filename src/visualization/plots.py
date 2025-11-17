"""
Comprehensive Visualization Suite
Training curves, ROC, PR, confusion matrix, DET curves, embeddings
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.manifold import TSNE
import umap
from pathlib import Path

sns.set_style('whitegrid')

def plot_training_curves(history: dict, save_path: str = None):
    """Plot loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_scores, save_path: str = None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Real', 'Fake'], save_path: str = None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_det_curve(y_true, y_scores, save_path: str = None):
    """
    Plot Detection Error Tradeoff (DET) curve
    Standard in speaker verification and anti-spoofing
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    plt.figure(figsize=(8, 6))
    plt.plot(fpr * 100, fnr * 100, lw=2)
    plt.xlabel('False Positive Rate (%)')
    plt.ylabel('False Negative Rate (%)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_embeddings(embeddings, labels, method='umap', save_path: str = None):
    """
    Plot embedding projections using t-SNE or UMAP
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)

    embedded = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(f'Embedding Projection ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectrogram_comparison(real_spec, fake_spec, save_path: str = None):
    """Compare real vs fake spectrograms"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    im1 = ax1.imshow(real_spec, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title('Real Audio Spectrogram')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(fake_spec, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('Fake Audio Spectrogram')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
