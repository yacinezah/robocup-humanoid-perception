import torch
from torch.utils.data import DataLoader, Subset

from soccer_segmentation.data.dataloader.dataset import DatasetSegmentation


def get_loader(folder_path, shuffle, small_mask=False, batch_size=32, num_workers=8,
               pin_memory=True, random_crop=True):
    dataset = DatasetSegmentation(
        folder_path=folder_path, small_mask=small_mask, random_crop=random_crop,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def get_train_val_loaders(folder_path, val_size, seed, small_mask=False, batch_size=32, num_workers=8, pin_memory=True):
    """Load the train folder and split into train/val subsets."""
    train_dataset = DatasetSegmentation(folder_path=folder_path, small_mask=small_mask, random_crop=True)
    val_dataset = DatasetSegmentation(folder_path=folder_path, small_mask=small_mask, random_crop=False)
    train_size = len(train_dataset) - val_size
    if train_size <= 0:
        raise ValueError(f"val_size={val_size} is too large for a dataset of {len(train_dataset)} images")
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    train_subset = Subset(train_dataset, indices[:train_size])
    val_subset = Subset(val_dataset, indices[train_size:])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
