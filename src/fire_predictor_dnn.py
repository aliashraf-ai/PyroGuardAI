"""
DNN Fire Spread Predictor
Predicts fire spread risk from weather and terrain data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


class FireSpreadPredictor:
    """Deep Neural Network for fire spread risk prediction"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'wind_speed_kmh',
            'humidity_percent',
            'temperature_celsius',
            'vegetation_density',
            'slope_degrees',
            'fuel_moisture_percent',
            'elevation_meters'
        ]

    def generate_training_data(self, n_samples=10000):
        """Generate synthetic weather and terrain data for training"""

        print(f"🔥 Generating {n_samples} training samples...")

        # Generate realistic weather/terrain features
        np.random.seed(42)

        data = {
            # Wind speed: 0-80 km/h (higher = more dangerous)
            'wind_speed_kmh': np.random.gamma(3, 5, n_samples),

            # Humidity: 10-90% (lower = more dangerous)
            'humidity_percent': np.random.beta(2, 2, n_samples) * 80 + 10,

            # Temperature: 10-45°C (higher = more dangerous)
            'temperature_celsius': np.random.normal(25, 8, n_samples),

            # Vegetation density: 0-1 (higher = more fuel)
            'vegetation_density': np.random.beta(2, 2, n_samples),

            # Slope: 0-60 degrees (steeper = faster spread)
            'slope_degrees': np.abs(np.random.normal(15, 10, n_samples)),

            # Fuel moisture: 5-40% (lower = more flammable)
            'fuel_moisture_percent': np.random.gamma(3, 3, n_samples) + 5,

            # Elevation: 0-3000m
            'elevation_meters': np.random.exponential(500, n_samples)
        }

        df = pd.DataFrame(data)

        # Clip values to realistic ranges
        df['wind_speed_kmh'] = df['wind_speed_kmh'].clip(0, 80)
        df['humidity_percent'] = df['humidity_percent'].clip(10, 90)
        df['temperature_celsius'] = df['temperature_celsius'].clip(10, 45)
        df['vegetation_density'] = df['vegetation_density'].clip(0, 1)
        df['slope_degrees'] = df['slope_degrees'].clip(0, 60)
        df['fuel_moisture_percent'] = df['fuel_moisture_percent'].clip(5, 40)
        df['elevation_meters'] = df['elevation_meters'].clip(0, 3000)

        # Calculate fire risk based on physical principles
        df['risk_score'] = (
                (df['wind_speed_kmh'] / 80) * 0.25 +  # Wind contribution
                ((90 - df['humidity_percent']) / 80) * 0.20 +  # Humidity (inverted)
                ((df['temperature_celsius'] - 10) / 35) * 0.15 +  # Temperature
                (df['vegetation_density']) * 0.15 +  # Fuel load
                (df['slope_degrees'] / 60) * 0.15 +  # Slope
                ((40 - df['fuel_moisture_percent']) / 35) * 0.10  # Fuel moisture (inverted)
        )

        # Classify into 4 risk levels
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=[0, 1, 2, 3],  # Low, Medium, High, Critical
            include_lowest=True
        ).astype(int)

        # Save data
        os.makedirs('data/weather_data', exist_ok=True)
        df.to_csv('data/weather_data/fire_risk_training_data.csv', index=False)

        print(f"✅ Data generated and saved!")
        print(f"\nRisk Distribution:")
        print(df['risk_level'].value_counts().sort_index())
        print(f"\n0 = Low Risk, 1 = Medium, 2 = High, 3 = Critical")

        return df


class FireRiskDNN(nn.Module):
    """Neural network architecture for fire risk classification"""

    def __init__(self, input_size=7, hidden_sizes=[64, 32, 16], num_classes=4):
        super(FireRiskDNN, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers with BatchNorm and Dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(data_path='data/weather_data/fire_risk_training_data.csv', epochs=100):
    """Train the DNN model"""

    print("\n🔥 TRAINING FIRE SPREAD PREDICTOR DNN...")

    # Load data
    if not os.path.exists(data_path):
        print("❌ Data file not found! Generate data first.")
        return None

    df = pd.read_csv(data_path)

    # Prepare features and labels
    feature_cols = [
        'wind_speed_kmh', 'humidity_percent', 'temperature_celsius',
        'vegetation_density', 'slope_degrees', 'fuel_moisture_percent',
        'elevation_meters'
    ]

    X = df[feature_cols].values
    y = df['risk_level'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    # Initialize model
    model = FireRiskDNN(input_size=7, hidden_sizes=[64, 32, 16], num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\n📊 Training Progress:")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Training accuracy
                train_pred = torch.argmax(outputs, dim=1)
                train_acc = (train_pred == y_train_tensor).float().mean()

                # Test accuracy
                test_outputs = model(X_test_tensor)
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = (test_pred == y_test_tensor).float().mean()

                print(f"Epoch [{epoch + 1:3d}/{epochs}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Test Acc: {test_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_pred = torch.argmax(test_outputs, dim=1)
        test_acc = (test_pred == y_test_tensor).float().mean()

        print("\n" + "=" * 60)
        print(f"✅ FINAL TEST ACCURACY: {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print("=" * 60)

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/fire_risk_dnn.pth')
    with open('models/fire_risk_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n💾 Model saved to models/fire_risk_dnn.pth")
    print("💾 Scaler saved to models/fire_risk_scaler.pkl")

    return model, scaler


def predict_fire_risk(weather_data, model=None, scaler=None):
    """
    Predict fire risk for new weather/terrain data

    Args:
        weather_data: dict or list with keys/values:
            ['wind_speed_kmh', 'humidity_percent', 'temperature_celsius',
             'vegetation_density', 'slope_degrees', 'fuel_moisture_percent',
             'elevation_meters']

    Returns:
        risk_level: 0=Low, 1=Medium, 2=High, 3=Critical
        probabilities: confidence for each class
    """

    # Load model if not provided
    if model is None:
        model = FireRiskDNN()
        model.load_state_dict(torch.load('models/fire_risk_dnn.pth'))
        model.eval()

    if scaler is None:
        with open('models/fire_risk_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

    # Convert input to array
    if isinstance(weather_data, dict):
        features = [
            weather_data['wind_speed_kmh'],
            weather_data['humidity_percent'],
            weather_data['temperature_celsius'],
            weather_data['vegetation_density'],
            weather_data['slope_degrees'],
            weather_data['fuel_moisture_percent'],
            weather_data['elevation_meters']
        ]
    else:
        features = weather_data

    # Preprocess
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    features_tensor = torch.FloatTensor(features_scaled)

    # Predict
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        risk_level = np.argmax(probabilities)

    return risk_level, probabilities


def test_predictor():
    """Test the trained model with sample scenarios"""

    print("\n🧪 TESTING FIRE RISK PREDICTOR...\n")

    test_scenarios = [
        {
            'name': 'Low Risk - Cool & Humid',
            'wind_speed_kmh': 10,
            'humidity_percent': 70,
            'temperature_celsius': 18,
            'vegetation_density': 0.3,
            'slope_degrees': 5,
            'fuel_moisture_percent': 30,
            'elevation_meters': 500
        },
        {
            'name': 'Medium Risk - Moderate Conditions',
            'wind_speed_kmh': 25,
            'humidity_percent': 45,
            'temperature_celsius': 28,
            'vegetation_density': 0.6,
            'slope_degrees': 15,
            'fuel_moisture_percent': 18,
            'elevation_meters': 800
        },
        {
            'name': 'High Risk - Hot & Windy',
            'wind_speed_kmh': 45,
            'humidity_percent': 25,
            'temperature_celsius': 38,
            'vegetation_density': 0.8,
            'slope_degrees': 30,
            'fuel_moisture_percent': 10,
            'elevation_meters': 1200
        },
        {
            'name': 'CRITICAL - Extreme Conditions',
            'wind_speed_kmh': 70,
            'humidity_percent': 12,
            'temperature_celsius': 42,
            'vegetation_density': 0.9,
            'slope_degrees': 45,
            'fuel_moisture_percent': 6,
            'elevation_meters': 1500
        }
    ]

    risk_labels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    for scenario in test_scenarios:
        name = scenario.pop('name')
        risk, probs = predict_fire_risk(scenario)

        print(f"📍 {name}")
        print(f"   Prediction: {risk_labels[risk]} RISK")
        print(f"   Confidence: {probs[risk] * 100:.1f}%")
        print(f"   Probabilities: Low={probs[0]:.2f}, Med={probs[1]:.2f}, "
              f"High={probs[2]:.2f}, Critical={probs[3]:.2f}\n")


if __name__ == '__main__':
    # Step 1: Generate training data
    predictor = FireSpreadPredictor()
    data = predictor.generate_training_data(n_samples=10000)

    # Step 2: Train model
    model, scaler = train_model(epochs=100)

    # Step 3: Test predictions
    test_predictor()

    print("\n✅ DNN COMPONENT COMPLETE!")
    print("\n🎯 Next: Run satellite_analyzer_cnn.py for CNN component")