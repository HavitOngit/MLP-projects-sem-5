import json
import os

# Example JSON data (replace this with your actual JSON data)
projects = [
    {"project_number": 1, "short_file_name": "numpy_arrays_operations"},
    {"project_number": 2, "short_file_name": "pandas_data_manipulation"},
    {"project_number": 3, "short_file_name": "iris_classification_sklearn"},
    {"project_number": 4, "short_file_name": "dice_simulation"},
    {"project_number": 5, "short_file_name": "probability_distribution"},
    {"project_number": 6, "short_file_name": "spam_classification_logreg"},
    {"project_number": 7, "short_file_name": "customer_segmentation_kmeans"},
    {"project_number": 8, "short_file_name": "pca_dimensionality_reduction"},
    {"project_number": 9, "short_file_name": "kfold_regression_validation"},
    {"project_number": 10, "short_file_name": "text_data_preprocessing"},
    {"project_number": 11, "short_file_name": "linear_logistic_regression"},
    {"project_number": 12, "short_file_name": "cca_implementation"},
    {"project_number": 13, "short_file_name": "knn_classification"},
    {"project_number": 14, "short_file_name": "svm_classification"},
    {"project_number": 15, "short_file_name": "random_forest_implementation"},
    {"project_number": 16, "short_file_name": "decision_tree_classification"},
    {"project_number": 17, "short_file_name": "mnist_digit_classification"}
]

def rename_files(projects):
    # Get the directory of the current script
    directory = os.path.dirname(os.path.abspath(__file__))
    
    for project in projects:
        old_name = f"{project['short_file_name']}.py"  # Use zero-padding for project numbers
        new_name = f"{project['project_number']}_{project['short_file_name']}.py"
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        else:
            print(f"File not found: {old_name}")

# Run the rename function
rename_files(projects)