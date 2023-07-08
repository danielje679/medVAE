import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


def get_single_dataset(name='chestmnist', split='train', batch_size=128, download=True):
    info = INFO[name]
    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    shuffle = True if split == 'train' else False

    dataset = DataClass(split=split, transform=data_transform, download=download)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, data_loader
