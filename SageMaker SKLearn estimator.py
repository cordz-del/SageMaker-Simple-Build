import argparse
import os
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

def main(args):
    # Generate synthetic training data
    np.random.seed(42)
    X = np.random.rand(100, 3)  # 100 samples, 3 features
    # A simple linear relationship: y = 1.5*x1 - 2.0*x2 + 3.0*x3 + noise
    y = np.dot(X, np.array([1.5, -2.0, 3.0])) + 0.5 + np.random.randn(100) * 0.1

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to the model directory provided by SageMaker
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # SageMaker passes the model directory via the SM_MODEL_DIR environment variable
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()
    main(args)
