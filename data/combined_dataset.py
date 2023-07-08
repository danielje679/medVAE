from torch.utils.data import Dataset
import torch
import numpy as np

from data.single_dataset import get_single_dataset


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)
        self.cumulative_lengths = torch.tensor(self.lengths).cumsum(0)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_index = torch.searchsorted(self.cumulative_lengths, idx, right=True)
        if dataset_index > 0:
            sample_index = idx - self.cumulative_lengths[dataset_index - 1]
        else:
            sample_index = idx
        sample = self.datasets[dataset_index][sample_index][0]
        label = np.array([dataset_index])  # Assign the dataset index as the label
        return sample, label


def get_combined_dataset(names, split='train', batch_size=128, download=True):
    datasets = [get_single_dataset(name=name, split=split, batch_size=batch_size, download=download)[0] for name in names]
    combined_dataset = CombinedDataset(datasets)
    return combined_dataset

