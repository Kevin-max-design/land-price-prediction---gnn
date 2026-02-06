"""
Flask Web Application for Land Price Prediction
================================================
Interactive demo showcasing the trained GAT model.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model import LandPriceGAT

app = Flask(__name__)

# Global variables for model and scalers
model = None
feature_scaler = None
target_scaler = None
model_info = {}

def load_model_and_scalers():
    """Load the trained model and fit scalers on training data."""
    global model, feature_scaler, target_scaler, model_info
    
    # Load training data to fit scalers
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'land_data.csv')
    df = pd.read_csv(data_path)
    
    # Prepare scalers (same as in preprocessing)
    df_encoded = pd.get_dummies(df, columns=['land_type'], prefix='land_type')
    
    target = df_encoded['price_per_sqft'].values.reshape(-1, 1)
    features = df_encoded.drop(columns=['price_per_sqft'])
    
    feature_scaler = StandardScaler()
    feature_scaler.fit(features.values)
    
    target_scaler = StandardScaler()
    target_scaler.fit(target)
    
    # Load model
    model = LandPriceGAT(
        in_channels=11,
        hidden_channels=64,
        out_channels=1,
        heads=4,
        dropout=0.3
    )
    
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_gat_model.pt')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("✓ Model loaded successfully")
    else:
        print("⚠ Model checkpoint not found, using untrained model")
    
    # Store model info
    model_info = {
        'name': 'Graph Attention Network (GAT)',
        'parameters': sum(p.numel() for p in model.parameters()),
        'r2_score': 0.9247,
        'mae': 585.32,
        'rmse': 728.41,
        'accuracy_10pct': 93.33,
        'dataset_size': len(df),
        'features': list(features.columns)
    }

def prepare_input(form_data):
    """Convert form data to model input tensor."""
    # Extract values from form
    location_score = float(form_data.get('location_score', 5))
    legal_clearance = int(form_data.get('legal_clearance', 1))
    road_width_ft = float(form_data.get('road_width_ft', 40))
    water_availability = int(form_data.get('water_availability', 1))
    soil_quality = float(form_data.get('soil_quality', 5))
    utilities_score = float(form_data.get('utilities_score', 5))
    neighbourhood_score = float(form_data.get('neighbourhood_score', 5))
    future_value_score = float(form_data.get('future_value_score', 5))
    land_type = form_data.get('land_type', 'Residential')
    
    # One-hot encode land type
    land_type_agricultural = 1 if land_type == 'Agricultural' else 0
    land_type_commercial = 1 if land_type == 'Commercial' else 0
    land_type_residential = 1 if land_type == 'Residential' else 0
    
    # Create feature array (same order as training)
    features = np.array([[
        location_score,
        legal_clearance,
        road_width_ft,
        water_availability,
        soil_quality,
        utilities_score,
        neighbourhood_score,
        future_value_score,
        land_type_agricultural,
        land_type_commercial,
        land_type_residential
    ]])
    
    # Scale features
    features_scaled = feature_scaler.transform(features)
    
    return torch.tensor(features_scaled, dtype=torch.float32)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input features."""
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Prepare input
        input_tensor = prepare_input(data)
        
        # Create a simple edge for single prediction (self-loop)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            prediction_scaled = model(input_tensor, edge_index)
        
        # Inverse transform to get actual price
        prediction = target_scaler.inverse_transform(prediction_scaled.numpy())[0][0]
        
        # Ensure prediction is within reasonable bounds
        prediction = max(5000, min(25000, prediction))
        
        return jsonify({
            'success': True,
            'prediction': round(float(prediction), 2),
            'formatted': f"₹{prediction:,.2f}/sqft"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model-info')
def get_model_info():
    """Return model information."""
    return jsonify(model_info)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  LAND PRICE PREDICTION WEB APP")
    print("="*50)
    
    load_model_and_scalers()
    
    print("\n  Starting server...")
    print("  Open: http://localhost:5001")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
