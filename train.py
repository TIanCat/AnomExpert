import os
import os
import sys
import argparse
import logging
import time
import torch
import json
import torch.nn as nn

from modules.trainer import train_epoch , validate_model
from utils import (set_random_seed, create_exp_dir, count_parameters_in_MB, 
                   plot_training_trend, get_trainable_parameters, plot_confusion_matrix)

from dataloader.loader import get_dataloader
from models.model import AnomExpert  

def get_arguments():
    """Parse command-line arguments for training pipeline"""
    parser = argparse.ArgumentParser('AnomExpert Training')
    
    # Dataset configuration
    parser.add_argument('-data_dir', type=str, default="/path/to/anonymous/dataset", 
                        help='Path to anonymized dataset directory (replaced for submission)')

    # Training hyperparameters
    parser.add_argument('-epochs', type=int, default=60, 
                        help='Total training epochs (default:60)')
    parser.add_argument('-batch_size', type=int, default=8, 
                        help='Case-level training batch size (default:8)')
    parser.add_argument('-lr', type=float, default=1e-4, 
                        help='Initial learning rate (default:1e-4)')
    parser.add_argument('-weight_decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer (default:1e-5)')
    parser.add_argument('-min_lr', type=float, default=1e-6, 
                        help='Minimum learning rate for cosine annealing (default:1e-6)')
    parser.add_argument('-patience', type=int, default=20, 
                        help='Patience for early stopping (default:20)')

    # AnomExpert model hyperparameters
    parser.add_argument('-lambda_proto', type=float, default=0.1, 
                        help='Weight for prototype assignment loss (λ, default:0.1)')
    parser.add_argument('-num_plane_protos', type=int, default=30, 
                        help='Number of plane prototypes (K, default:30)')
    parser.add_argument('-topk_planes', type=int, default=4, 
                        help='Top-k planes for disease-aware selection (default:4)')
    parser.add_argument('-max_images_per_case', type=int, default=25, 
                        help='Maximum number of images per case (default:25)')
    parser.add_argument('-num_disease_categories', type=int, default=9, 
                        help='Number of disease categories (C, default:9)')
    parser.add_argument('-encoder_dim', type=int, default=384, 
                        help='ViT encoder output dimension (default:384 for ViT-small)')
    parser.add_argument('-hidden_dim', type=int, default=256, 
                        help='Feature embedding dimension (D, default:256)')
    
    # Runtime settings
    parser.add_argument('-gpu_id', type=int, default=0, 
                        help='GPU device ID for training (anonymized, default:0)')
    parser.add_argument('-seed', type=int, default=42, 
                        help='Random seed for reproducibility (default:45)')
    parser.add_argument('-num_workers', type=int, default=4, 
                        help='Number of data loading workers (default:4)')

    # Experiment saving directory
    parser.add_argument('-save_dir', type=str, default='runs', 
                        help='Root directory for saving experiment results')

    return parser.parse_args()



def main():
    
    """Main training pipeline for AnomExpert."""
    # Parse command-line arguments
    args = get_arguments()
    
    # Setup experiment environment (seed, directories, logging)
    exp_dir, ckp_dir = setup_experiment(args)

    # Initialize core components (model, data loaders, optimizer, loss)
    model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion = \
        initialize_model_and_data(args)


    # Training loop tracking variables
    no_improvement_epochs = 0  # Epochs without validation improvement
    best_val_f1 = 0.0          # Best validation F1-score (primary metric)

    # Metric tracking lists (for plotting)
    train_metrics = {
        'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []
    }
    val_metrics = {
        'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []
    }
    
    # Main training loop
    for epoch in range(1, args.epochs + 1):
        # Log epoch status
        logging.info("-" * 100)
        logging.info(f"Epoch: {epoch}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Training phase (single epoch)
        train_loss, train_epoch_metrics = train_epoch(
            model, train_loader, optimizer, criterion, args
        )
        scheduler.step()

        # Validation phase
        val_loss, val_epoch_metrics = validate_model(
            model, val_loader, criterion, stage='Validation', args=args
        )
        
        # Testing phase (evaluate on test set for monitoring)
        test_loss, test_epoch_metrics = validate_model(
            model, test_loader, criterion, stage='Testing', args=args
        )

        # Log epoch results
        logging.info(f"Train | Loss: {train_loss:.4f} | Metrics: {train_epoch_metrics}")
        logging.info(f"Val   | Loss: {val_loss:.4f} | Metrics: {val_epoch_metrics}")
        logging.info(f"Test  | Loss: {test_loss:.4f} | Metrics: {test_epoch_metrics}")

        # Update metric tracking lists
        train_metrics['loss'].append(train_loss)
        val_metrics['loss'].append(val_loss)
    
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
            train_metrics[metric].append(train_epoch_metrics.get(metric, float('nan')))
            val_metrics[metric].append(val_epoch_metrics.get(metric, float('nan')))


        # Plot training trends (update after each epoch)
        for metric_name in ['Loss', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC']:
            plot_training_trend(
                train_metrics[metric_name.lower()],
                val_metrics[metric_name.lower()],
                metric_name,
                exp_dir
            )
        

        # Get current validation F1 (primary metric for early stopping)
        current_val_f1 = val_epoch_metrics.get('f1', 0.0)

        # Early stopping check
        if current_val_f1 <= best_val_f1:
            no_improvement_epochs += 1
            logging.info(f"No improvement | Current Val F1: {current_val_f1:.4f} | Best: {best_val_f1:.4f} | Patience: {no_improvement_epochs}/{args.patience}")
            
            if no_improvement_epochs >= args.patience:
                logging.info("Early stopping triggered - training completed.")
                break
            continue  


        # Update best model and save results
        no_improvement_epochs = 0
        best_val_f1 = current_val_f1
        
        # Save best test metrics (with epoch info)
        best_test_metrics = test_epoch_metrics.copy()
        best_test_metrics['epoch'] = epoch
        metrics_path = os.path.join(exp_dir, 'best_test_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(best_test_metrics, f, ensure_ascii=False, indent=4)

        # Save best model checkpoint
        model_checkpoint_path = os.path.join(ckp_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_checkpoint_path)
        logging.info(f"Saved best model to {model_checkpoint_path} (Epoch {epoch})")

        # Plot and save confusion matrix (if available)
        confusion_matrix = test_epoch_metrics.get("confusion_matrix", None)
        if confusion_matrix is not None:
            # Save non-normalized confusion matrix
            cm_path = os.path.join(exp_dir, "best_confusion_matrix.png")
            plot_confusion_matrix(
                confusion_matrix,
                save_path=cm_path,
                class_names=args.class_names,
                normalize=False,
                title=f"Best Model Confusion Matrix (Epoch {epoch}) | Val F1: {best_val_f1:.4f}"
            )
            
            # Save normalized confusion matrix (for better visualization)
            cm_norm_path = os.path.join(exp_dir, "best_confusion_matrix_normalized.png")
            plot_confusion_matrix(
                confusion_matrix,
                save_path=cm_norm_path,
                class_names=args.class_names,
                normalize=True,
                title=f"Best Model Confusion Matrix (Normalized) | Epoch {epoch}"
            )
        
        logging.info(f"Best model updated | Val F1: {best_val_f1:.4f} | Test F1: {test_epoch_metrics.get('f1', 0.0):.4f}")






def setup_experiment(args):
    """Configure experiment environment: seed, GPU, directories, logging"""
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Configure GPU device
    torch.cuda.set_device(args.gpu_id)
    
    # Create experiment directories
    exp_dir = f'EXP-{time.strftime("%Y%m%d-%H%M%S")}'
    exp_dir = os.path.join(args.save_dir, exp_dir)
    ckp_dir = os.path.join(exp_dir, 'checkpoint')
    create_exp_dir(ckp_dir)
    
    # Configure logging (console + file)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    ]
    log_format = '%(asctime)s %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Log experiment configuration
    logging.info("Experiment arguments: %s", args)
    
    return exp_dir, ckp_dir



def initialize_model_and_data(args):
    """Initialize data loaders, model, optimizer, scheduler and loss function"""

    train_loader, val_loader, test_loader  = get_dataloader(args)

    args.class_names = getattr(train_loader.dataset, "class_names", None)

    if args.class_names is None:
        args.class_names = [str(i) for i in range(args.num_classes)]
    logging.info(f"class_names: {args.class_names}")

    # Initialize AnomExpert model (aligned with normalized model code)
    model = AnomExpert(
        num_disease_categories=args.num_disease_categories,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        num_plane_protos=args.num_plane_protos,
        topk_planes=args.topk_planes
    ).cuda()

    # Log model parameter size (matching paper Table 1)
    model_param_size = count_parameters_in_MB(model)
    logging.info(f"AnomExpert model parameter size: {model_param_size:.2f} MB")


    # Initialize optimizer (only trainable parameters)
    params = get_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.min_lr
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    return model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion



if __name__ == "__main__":
    
    main()