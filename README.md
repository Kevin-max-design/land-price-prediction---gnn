# Graph Neural Network Based Land Price Prediction

A Graph Attention Network (GAT) model for predicting land prices by converting tabular geospatial data into a graph structure.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```

## Project Structure

```
gnn-land-price-prediction/
├── data/land_data.csv          # Land dataset
├── src/
│   ├── data_preprocessing.py   # Load, clean, normalize data
│   ├── graph_construction.py   # Convert tabular → graph (KNN edges)
│   ├── model.py                # GAT architecture
│   ├── train.py                # Training loop with early stopping
│   └── evaluate.py             # Metrics and visualization
├── main.py                     # End-to-end pipeline
├── checkpoints/                # Saved models
└── results/                    # Evaluation plots
```

## How It Works

1. **Graph Construction**: Each land parcel becomes a node. Edges connect similar parcels using K-Nearest Neighbors (k=5) in feature space.

2. **GAT Architecture**: Multi-head attention (4 heads) learns which neighbor features are most relevant for price prediction.

3. **Training**: MSE loss with early stopping, learning rate scheduling, and gradient clipping.

## Metrics

| Metric | Description |
|--------|-------------|
| R² Score | Coefficient of determination |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| Accuracy ±10% | % predictions within 10% of actual |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- pandas, numpy, scikit-learn, matplotlib, seaborn
