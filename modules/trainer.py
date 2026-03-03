import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)


def train_epoch(model, data_loader, optimizer, criterion, args):
    """Execute one training epoch for AnomExpert"""

    model.train()
    epoch_loss = 0.0
    predictions, ground_truths, prob_list = [], [], []

    for batch_data in tqdm(data_loader, desc="Training Epoch"):

        # Unpack batch data
        images = batch_data['images'].cuda()
        case_lengths = batch_data['lengths']
        labels = batch_data['labels'].cuda()

        # Forward pass
        optimizer.zero_grad()
        logits, loss_proto, aux_outputs = model(images, case_lengths)
        pred_probs = torch.softmax(logits, dim=1)

        # Total loss (cross-entropy + prototype loss)
        loss_ce = criterion(logits, labels)
        total_loss = loss_ce + loss_proto * args.lambda_proto

        # Record predictions and ground truths
        predictions.append(torch.argmax(pred_probs, dim=-1))
        ground_truths.append(labels)

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Accumulate loss and probabilities
        epoch_loss += total_loss.item()
        prob_list.append(pred_probs)  
        
    # Aggregate results across all batches
    predictions = torch.cat(predictions, dim=0).tolist()
    ground_truths = torch.cat(ground_truths, dim=0).tolist()
    y_prob = torch.cat(prob_list, dim=0).detach().cpu().numpy()

    # Calculate evaluation metrics (matching paper Sec.3.1)
    metrics = calculate_evaluation_metrics(ground_truths, predictions, y_prob, args)
    
    # Average loss over all batches
    avg_loss = epoch_loss / len(data_loader)

    return avg_loss, metrics


def validate_model(model, data_loader, criterion, stage, args):
    """Evaluate AnomExpert on validation/test set"""

    model.eval()
    epoch_loss = 0.0
    predictions, ground_truths, prob_list = [], [], []

    with torch.no_grad():  # Disable gradient computation for inference
        for batch_data in tqdm(data_loader, desc=stage):

            # Unpack batch data
            images = batch_data['images'].cuda()
            case_lengths = batch_data['lengths']
            labels = batch_data['labels'].cuda()

            # Forward pass (no gradient computation)
            logits, loss_proto, aux_outputs = model(images, case_lengths)
            pred_probs = torch.softmax(logits, dim=1)

            # Total loss (for monitoring only)
            loss_ce = criterion(logits, labels)
            total_loss = loss_ce + loss_proto * args.lambda_proto

            # Record predictions and ground truths
            predictions.append(torch.argmax(pred_probs, dim=-1))
            ground_truths.append(labels)
            
            # Accumulate loss and probabilities
            epoch_loss += total_loss.item()
            prob_list.append(pred_probs)  
        
    # Aggregate results across all batches
    predictions = torch.cat(predictions, dim=0).tolist()
    ground_truths = torch.cat(ground_truths, dim=0).tolist()
    y_prob = torch.cat(prob_list, dim=0).detach().cpu().numpy()  # (N, C)

    # Calculate evaluation metrics
    metrics = calculate_evaluation_metrics(ground_truths, predictions, y_prob, args)
    
    # Average loss over all batches
    avg_loss = epoch_loss / len(data_loader)

    return avg_loss, metrics




def calculate_evaluation_metrics(ground_truths: list, predictions: list, y_prob: np.ndarray, args):
    """Calculate standard evaluation metrics"""

    # Basic classification metrics (macro-averaged, matching paper)
    accuracy = accuracy_score(ground_truths, predictions)
    precision = precision_score(ground_truths, predictions, average='macro', zero_division=0)
    recall = recall_score(ground_truths, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truths, predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(ground_truths, predictions).tolist()

    # AUC calculation (handle binary/multi-class cases)
    auc = float("nan")
    try:
        num_classes = y_prob.shape[1]
        if num_classes == 2:
            # Binary classification (positive class probability)
            auc = roc_auc_score(ground_truths, y_prob[:, 1])
        else:
            # Multi-class classification (One-vs-Rest, macro-averaged)
            auc = roc_auc_score(
                ground_truths, y_prob,
                multi_class=getattr(args, "auc_multi_class", "ovr"),
                average=getattr(args, "auc_average", "macro"),
            )
    except (ValueError, AttributeError):
        # Handle cases where AUC cannot be computed (e.g., single class in batch)
        auc = float("nan")

    # Organize metrics in dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }

    return metrics