# app.py
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import math
from time import time
from itertools import cycle
from copy import deepcopy

# Import functions from your neural network module
import neural_network
# We assume neural_network.py defines: makemodel, generate_data,
# train_and_test_model_time_based, and (optionally) train_classification_model_time_based.

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def convert_config(config_list):
    """
    Convert the configuration from the GUI into a list of lists that your neural network code expects.
    The GUI sends each component as a dict with "type" and "config" (a list of string values).
    This helper maps names (e.g., "Convolutional" → "Conv", "ReLU Activation" → "ReLU") and converts parameters to integers.
    """
    converted = []
    # Mapping from GUI names to the names used in your neural network code.
    mapping = {
        "Linear": "Linear",
        "Convolutional": "Conv",
        "Dropout": "Dropout",
        "ReLU": "ReLU",
        "ReLU Activation": "ReLU",
        "Leaky": "LeakyReLU",           # In case only "Leaky" is sent
        "Leaky ReLU": "LeakyReLU",
        "ELU": "ELU",
        "Sigmoid": "Sigmoid",
        "Sigmoid Activation": "Sigmoid",
        "Batch": "BatchNorm",           # In case only "Batch" is sent
        "Batch Normalization": "BatchNorm",
        "Pool": "Pool",
        "Flatten": "Flatten"
    }
    
    for comp in config_list:
        # Get the layer name from the component. (The GUI sends only the first word, so you might need to adjust.)
        comp_type = comp.get("type", "").strip()
        # Map the name to the expected key
        comp_mapped = mapping.get(comp_type, comp_type)
        # Convert any configuration parameters to numbers (if provided)
        raw_params = comp.get("config", [])
        params = []
        for p in raw_params:
            try:
                # If the input is empty, skip it.
                if p == "":
                    continue
                params.append(int(p))
            except ValueError:
                try:
                    params.append(float(p))
                except ValueError:
                    continue

        # Build the list expected by makemodel:
        if comp_mapped == "Conv":
            # Expected: ["Conv", in_channels, out_channels, kernel_size]
            # Here, we assume a default in_channels=1 if not specified by the user.
            if len(params) >= 2:
                converted.append(["Conv", 1, params[0], params[1]])
            else:
                # If parameters are missing, skip this layer.
                continue
        elif comp_mapped == "Linear":
            # Expected: ["Linear", out_features]
            if len(params) >= 1:
                converted.append(["Linear", params[0]])
            else:
                continue
        elif comp_mapped == "Pool":
            # Expected: ["Pool", kernel_size]
            if len(params) >= 1:
                converted.append(["Pool", params[0]])
            else:
                continue
        else:
            # For layers that do not require parameters (Dropout, ReLU, LeakyReLU, ELU, Sigmoid, BatchNorm, Flatten)
            converted.append([comp_mapped])
    return converted

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    if not data or 'components' not in data:
        return jsonify({'error': 'No network configuration provided.'}), 400
    
    # Convert the configuration from the GUI into the format your engine expects.
    gui_config = data['components']
    converted_config = convert_config(gui_config)
    if not converted_config:
        return jsonify({'error': 'Invalid network configuration.'}), 400

    try:
        # Build the model using your neural network engine function.
        model = neural_network.makemodel(converted_config)
    except Exception as e:
        return jsonify({'error': f'Error building model: {str(e)}'}), 500

    try:
        # Generate regression data using the sine function.
        # generate_data expects a function that accepts a numpy array.
        X_train, X_test, y_train, y_test = neural_network.generate_data(np.sin, test_size=0.2)
    except Exception as e:
        return jsonify({'error': f'Error generating data: {str(e)}'}), 500

    try:
        # Train the model using your time-based training function.
        # Here we use a 10-second training period.
        total_inacc, std_inacc = neural_network.train_and_test_model_time_based(
            model, X_train, y_train, X_test, y_test,
            time_limit=10, batch_size=32, learning_rate=0.001
        )
    except Exception as e:
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

    try:
        # For plotting: create 100 evenly spaced x-values from the min to max of X_test.
        x_test_np = X_test.cpu().numpy().flatten() if torch.is_tensor(X_test) else np.array(X_test).flatten()
        x_min, x_max = float(np.min(x_test_np)), float(np.max(x_test_np))
        x_plot = np.linspace(x_min, x_max, 100)
        y_true = np.sin(x_plot)
        
        # Get predictions from the trained model.
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1)
            # Ensure x_tensor is on the same device as the model.
            device = next(model.parameters()).device
            x_tensor = x_tensor.to(device)
            y_pred_tensor = model(x_tensor)
            y_pred = y_pred_tensor.cpu().numpy().flatten()
    except Exception as e:
        return jsonify({'error': f'Error during evaluation: {str(e)}'}), 500

    # Return the plotting data and metrics.
    return jsonify({
        "x": x_plot.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "total_inaccuracy": total_inacc,
        "std_inaccuracy": std_inacc
    })

if __name__ == '__main__':
    app.run(debug=True)
