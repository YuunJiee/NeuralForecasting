import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_dataset, NeuroForcastDataset
from utils.trainer import Trainer, AdvancedTrainer
from utils.graph_utils import get_pearson_correlation
from model import NFBaseModel, DLinear, AMAGModel, DLinearGNNModel, STNDTModel, DLinearSTNDTModel

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beignet', choices=['beignet', 'affi'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--model', type=str, default='amag', choices=['amag', 'dlinear_gnn', 'stndt', 'dlinear_stndt'], help='Model architecture')
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
    
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model Type: {model_type}")

    # --- Hyperparameters ---
    # Model-specific Learning Rates
    if model_type in ['stndt', 'dlinear_stndt']:
        learning_rate = 1e-4 # Transformers need lower LR
    else:
        learning_rate = 1e-3 # DLinear/AMAG work better with 1e-3
        
    print(f"Learning Rate: {learning_rate}")

    # --- Data Loading ---
    data_filename = f'train_data_{dataset_name}.npz'
    
    # Check if data exists
    if not os.path.exists(os.path.join('./data', data_filename)):
        print(f"Error: Data file {data_filename} not found in ./data/")
        print("Please download the data first.")
        return

    print("Loading data...")
    try:
        train_data, test_data, val_data = load_dataset(data_filename, input_dir='./data')
        
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
        model = DLinearGNNModel(num_nodes=input_size, adj_init=adj_init)
    elif model_type == 'stndt':
        model = STNDTModel(num_nodes=input_size)
    elif model_type == 'dlinear_stndt':
        model = DLinearSTNDTModel(num_nodes=input_size)
    else:
        model = AMAGModel(num_nodes=input_size, adj_init=adj_init)
        
    model = model.to(device)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # --- Training ---
    print("Starting training...")
    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)
    if model_type == 'amag':
        save_path = os.path.join('weights', f'model_{dataset_name}.pth')
    else:
        save_path = os.path.join('weights', f'model_{dataset_name}_{model_type}.pth')
    
    # Select Trainer
    if model_type in ['stndt', 'dlinear_stndt']:
        print("Using AdvancedTrainer (Masking + Contrastive Loss)")
        trainer = AdvancedTrainer(
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
            mask_ratio=0.15,
            lambda_recon=0.5,
            lambda_contrast=0.1
        )
    else:
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
            save_path=save_path # No checkpoint loading for now
        )
    
    trainer.train(num_epochs)
    print("Training complete.")

if __name__ == '__main__':
    main()
