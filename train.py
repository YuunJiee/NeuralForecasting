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
from utils.trainer import Trainer
from model import NFBaseModel, DLinear

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beignet', choices=['beignet', 'affi'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    args = parser.parse_args()

    # --- Configuration ---
    dataset_name = args.dataset
    
    if dataset_name == 'beignet':
        num_channels = 89
    else:
        num_channels = 239

    batch_size = 32
    num_epochs = args.epochs
    learning_rate = 1e-4 # Demo used 1e-4 variables, but optimizer used 0.001 hardcoded. Let's use 0.001
    hidden_size = 1024
    input_size = num_channels
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")

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
        stats_path = os.path.join('weights', f'stats_{dataset_name}.npz')
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

    # --- Model Setup ---
    print("Initializing model (DLinear)...")
    # model = NFBaseModel(input_size=input_size, hidden_size=hidden_size)
    model = DLinear(input_size=input_size)
    model = model.to(device)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scheduler from demo
    lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 50)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # --- Training ---
    print("Starting training...")
    # Ensure weights directory exists
    os.makedirs('weights', exist_ok=True)
    save_path = os.path.join('weights', f'model_{dataset_name}.pth')
    
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
        save_path=save_path
    )
    
    trainer.train(num_epochs)
    print("Training complete.")

if __name__ == '__main__':
    main()
