import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import wandb

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_dataset, NeuroForcastDataset
from utils.trainer import Trainer, AdvancedTrainer
from utils.graph_utils import get_pearson_correlation
from utils.loss import WeightedMSELoss
from model import NFBaseModel, DLinear, AMAGModel, DLinearGNNModel

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beignet', choices=['beignet', 'affi'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--model', type=str, default='amag', choices=['amag', 'dlinear_gnn'], help='Model architecture')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for weighted loss')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    parser.add_argument('--full_data', action='store_true', help='Use all available data (Public+Private) for training')
    parser.add_argument('--name', type=str, default=None, help='WandB run name')
    args = parser.parse_args()

    # --- Configuration ---
    dataset_name = args.dataset
    model_type = args.model
    
    if dataset_name == 'beignet':
        num_channels = 89
    else:
        num_channels = 239

    batch_size = 32
    num_epochs = args.epochs
    input_size = num_channels
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Hyperparameters ---
    # Model-specific Learning Rates
    learning_rate = 1e-3 # DLinear/AMAG work better with 1e-3

    # --- WandB Init ---
    run = wandb.init(
        # Set the wandb project where this run will be logged.
        project="NSF neural forecasting",
        name=args.name,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": model_type,
            "dataset": dataset_name,
            "epochs": num_epochs,
            "loss_type": "weighted_mse"
        },
    )
    print(f"WandB initialized: {wandb.run.name}")
    
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model Type: {model_type}")
    print(f"Learning Rate: {learning_rate}")

    # --- Data Loading ---
    data_filename = f'train_data_{dataset_name}.npz'
    
    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(os.path.join('./data', data_filename)):
        print(f"Error: Data file {data_filename} not found in ./data/")
        print("Please download the data first.")
        return

    print("Loading data...")
    try:
        extra_files = []
        if args.full_data:
            print("--- FULL DATA MODE ENABLED ---")
            if dataset_name == 'beignet':
                extra_files = [
                    'train_data_beignet_2022-06-01_private.npz',
                    'train_data_beignet_2022-06-02_private.npz'
                ]
            elif dataset_name == 'affi':
                extra_files = [
                    'train_data_affi_2024-03-20_private.npz'
                ]
        
        train_data, test_data, val_data = load_dataset(
            data_filename, 
            input_dir='./data', 
            extra_files=extra_files, 
            full_train=args.full_data
        )
        
        # Create Datasets
        # We need to compute stats from train_data first
        train_dataset = NeuroForcastDataset(train_data, use_graph=False)
        
        # Save average and std for model.py to use during inference
        # Note: If we switch models, stats are likely the same, but weights differ.
        if model_type == 'amag':
             stats_path = os.path.join('weights', f'stats_{dataset_name}.npz')
        else:
             stats_path = os.path.join('weights', f'stats_{dataset_name}_{model_type}.npz')
             
        np.savez(stats_path, average=train_dataset.average, std=train_dataset.std)
        print(f"Saved training stats to {stats_path}")
        
        test_dataset = NeuroForcastDataset(test_data, use_graph=False, average=train_dataset.average, std=train_dataset.std)
        val_dataset = NeuroForcastDataset(val_data, use_graph=False, average=train_dataset.average, std=train_dataset.std)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # --- Graph Initialization ---
    # Only AMAG and DLinearGNN use Pearson graph? STNDT uses learnable or nothing?
    # STNDT uses separate spatial embeddings and attention, does not strictly require Pearson Adjacency Matrix
    # But maybe we can inject it? Current implementation of STNDT does NOT use adj_init.
    
    adj_init = None
    if model_type in ['amag', 'dlinear_gnn']:
        print("Computing Pearson Correlation Matrix...")
        adj_init = get_pearson_correlation(train_dataset.data)
        print(f"Computed Pearson Matrix with shape: {adj_init.shape}")

    # --- Model Setup ---
    print(f"Initializing model ({model_type})...")
    
    # We use the Model wrapper indirectly? Or call AMAGModel/DLinearGNN directly?
    # Actually train.py was calling AMAGModel directly in previous edit (Step 32).
    # But for better design, let's use the Model wrapper? 
    # The Model wrapper in model.py is designed for inference (load weights, normalize).
    # For training, earlier we instantiated AMAGModel directly.
    # Let's instantiate based on model_type argument logic consistent with model.py classes.
    
    if model_type == 'dlinear_gnn':
        model = DLinearGNNModel(num_nodes=input_size, adj_init=adj_init, dropout=args.dropout, hidden_dim=args.hidden_dim)
    else:
        model = AMAGModel(num_nodes=input_size, adj_init=adj_init, hidden_dim=args.hidden_dim, dropout=args.dropout)
        
    model = model.to(device)

    print(f"Using WeightedMSELoss (alpha={args.alpha})")
    loss_fn = WeightedMSELoss(alpha=args.alpha, reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    print("Using Cosine Annealing Scheduler")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, cooldown=5, min_lr=1e-6)
    
    # --- Training ---
    print("Starting training...")
    if model_type == 'amag':
        save_path = os.path.join('weights', f'model_{dataset_name}.pth')
    else:
        save_path = os.path.join('weights', f'model_{dataset_name}_{model_type}.pth')
    
    # Select Trainer
    trainer = Trainer(
        model=model,
        train_data_loader=train_loader,
        test_data_loader=test_loader,
        val_data_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        forecasting_mode='multi_step',
        init_steps=10,
        save_path=save_path, # No checkpoint loading for now
        run=run
    )
    
    # Apply Curriculum


    trainer.train(num_epochs)
    print("Training complete.")

if __name__ == '__main__':
    main()
