""" Downloading Kaggle dataset using Kaggle API """

import os
import zipfile
import kaggle

def download_and_extract_data(kaggle_dataset, download_path, extract_path):
    # Check if the dataset is already downloaded
    if not os.path.exists(download_path):
        print("Downloading dataset...")
        kaggle.api.dataset_download_files(kaggle_dataset, path=download_path, unzip=False)
    else:
        print("Dataset already downloaded.")

    # Extract the dataset if not already extracted
    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(os.path.join(download_path, f"{kaggle_dataset.split('/')[-1]}.zip"), 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset extracted to {extract_path}")
    else:
        print("Dataset already extracted.")

    return extract_path

if __name__ == "__main__":
    # Define the Kaggle dataset and paths
    kaggle_dataset = "abdallahalidev/plantvillage-dataset"
    download_path = "./data/raw"
    extract_path = "./data/raw/plantvillage"

    # Download and extract the dataset
    dataset_path = download_and_extract_data(kaggle_dataset, download_path, extract_path)
    print(f"Dataset downloaded and extracted to: {dataset_path}")