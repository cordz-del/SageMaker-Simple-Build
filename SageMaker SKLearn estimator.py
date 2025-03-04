import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role

# Get the SageMaker execution role
role = get_execution_role()
print("Using role:", role)

# Define the SKLearn estimator with our custom training script
sklearn_estimator = SKLearn(
    entry_point="train_script.py",         # Your training code
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    hyper
