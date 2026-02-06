"""
Land Price Prediction using Graph Attention Networks
=====================================================
Main pipeline that orchestrates all components:
1. Data preprocessing
2. Graph construction
3. GAT model training
4. Evaluation and visualization

Author: AI Assistant
Date: February 2026
"""

import os
import sys
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import load_data, explore_data, preprocess_data, create_masks
from src.graph_construction import create_graph, visualize_graph_stats
from src.model import LandPriceGAT, LandPriceGATv2, count_parameters
from src.train import train
from src.evaluate import evaluate_model, generate_all_plots

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


def main():
    """
    Main function to run the complete land price prediction pipeline.
    """
    print("\n" + "="*70)
    print("  GRAPH ATTENTION NETWORK FOR LAND PRICE PREDICTION")
    print("="*70 + "\n")
    
    # =========================================================================
    # STEP 1: LOAD AND EXPLORE DATA
    # =========================================================================
    print("STEP 1: Loading and exploring data...")
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'land_data.csv')
    df = load_data(data_path)
    stats = explore_data(df)
    
    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================
    print("\nSTEP 2: Preprocessing data...")
    
    features, target, feature_names, feature_scaler, target_scaler = preprocess_data(df)
    train_mask, val_mask, test_mask = create_masks(len(df))
    
    # =========================================================================
    # STEP 3: CONSTRUCT GRAPH
    # =========================================================================
    print("\nSTEP 3: Constructing graph from tabular data...")
    
    # Convert tabular data to graph using KNN for edges
    # k=5 means each land parcel is connected to 5 most similar parcels
    data = create_graph(
        features=features,
        target=target,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        k=5
    )
    
    visualize_graph_stats(data)
    
    # =========================================================================
    # STEP 4: BUILD AND TRAIN GAT MODEL
    # =========================================================================
    print("\nSTEP 4: Building and training GAT model...")
    
    # Initialize GAT model
    model = LandPriceGAT(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=1,
        heads=4,
        dropout=0.3
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Train the model
    history = train(
        model=model,
        data=data,
        epochs=500,
        lr=0.01,
        weight_decay=5e-4,
        patience=50,
        save_dir='checkpoints',
        verbose=True
    )
    
    # =========================================================================
    # STEP 5: EVALUATE MODEL
    # =========================================================================
    print("\nSTEP 5: Evaluating model performance...")
    
    train_metrics, val_metrics, test_metrics = evaluate_model(
        model=model,
        data=data,
        target_scaler=target_scaler,
        save_dir='results'
    )
    
    # =========================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # =========================================================================
    print("\nSTEP 6: Generating visualizations...")
    
    generate_all_plots(
        model=model,
        data=data,
        history=history,
        target_scaler=target_scaler,
        save_dir='results'
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"\n  Model: Graph Attention Network (GAT)")
    print(f"  Dataset: {len(df)} land parcels")
    print(f"  Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"\n  Test Set Performance:")
    print(f"    R² Score:        {test_metrics['R2']:.4f}")
    print(f"    MAE:             {test_metrics['MAE']:.2f}")
    print(f"    RMSE:            {test_metrics['RMSE']:.2f}")
    print(f"    MAPE:            {test_metrics['MAPE']:.2f}%")
    print(f"    Accuracy (±10%): {test_metrics['Accuracy_10pct']:.2f}%")
    print(f"\n  Outputs saved to:")
    print(f"    - Model checkpoint: checkpoints/best_gat_model.pt")
    print(f"    - Visualizations:   results/")
    print("="*70 + "\n")
    
    return model, data, history, test_metrics


if __name__ == "__main__":
    model, data, history, test_metrics = main()

