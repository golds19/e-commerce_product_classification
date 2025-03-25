from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

def get_data_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)), # matchine the size for the pretrained model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_data(df, subcategories):
    return df.loc[lambda df: df['subCategory'].isin(subcategories)]

def create_data_loaders(train_dataset,
                        val_dataset,
                        batch_size=32):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader