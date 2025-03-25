import torch
import torch.nn as nn
from src.data.dataset import CustomImageDataset
from src.data.data_loader import get_data_transforms, create_data_loaders
from src.preprocess import preprocess_data
from src.models.model import setup_model
from src.training.trainer import Trainer
from src.config import Config
import pandas as pd
import os

def main():
    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and preprocess data
    df_filtered = preprocess_data()

    print(df_filtered)

    # setup transforms
    transforms = get_data_transforms()

    # create datasets
    train_dataset = CustomImageDataset(
        dataframe=df_filtered,
        directory=os.path.join(Config.DATASET_PATH, "images"),
        transform=transforms,
        subset="training"
    )

    val_dataset = CustomImageDataset(
        dataframe=df_filtered,
        directory=os.path.join(Config.DATASET_PATH, "images"),
        transform=transforms,
        subset="validation"
    )

    # Create dataloaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset,
        batch_size=Config.BATCH_SIZE
    )

    # Setup model
    model = setup_model(num_classes=Config.NUM_CLASSES, device=device)
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        **Config.OPTIMIZER["params"]
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    print(f"Training for {Config.EPOCHS} epochs...")
    results = trainer.train(epochs=Config.EPOCHS)
    print("Training Completed")

if __name__ == "__main__":
    main()
