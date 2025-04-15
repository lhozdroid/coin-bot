import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class CandleDataset(Dataset):
    def __init__(self, features_path: str, labels_path: str):
        """
        Initializes the dataset by loading the features and labels

        Args:
            features_path:
            labels_path:
        """
        # Loads the preprocessed numpy arrays
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        """
        Returns the number of samples in the dataset

        Returns:
            int
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset

        Args:
            idx:

        Returns:
            Tuple[Tensor, Tensor]
        """
        x = torch.tensor(self.features[idx], dtype=torch.float32)  # [seq_len, input_size]
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # []
        return x, y


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Splits the dataset into train, validation, and test sets

    Args:
        dataset:
        train_ratio:
        val_ratio:

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])
