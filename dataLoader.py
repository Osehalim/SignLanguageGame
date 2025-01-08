# data_loader.py
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.labels = data.iloc[:, 0].values  # First column is labels
        self.images = data.iloc[:, 1:].values  # Remaining columns are pixel values
        self.images = self.images.reshape(-1, 28, 28).astype(np.float32)  # Reshape to 28x28
        
        # Sanity check: Ensure labels are between 0 and 23
        self.labels = np.clip(self.labels, 0, 23)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.expand_dims(image, axis=0)  # Add channel dimension (1x28x28)
        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)


def get_data_loaders(batch_size=32, test_size=0.2):
    # Load CSV data from the Kaggle dataset
    dataset = SignLanguageDataset('./data/sign_mnist_train.csv')
    
    # Split into train and test sets
    train_data, test_data = train_test_split(dataset, test_size=test_size)
    
    # Create PyTorch DataLoader for both sets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
