import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self,
                 dataframe,
                 directory,
                 transform=None,
                 subset="training"
                 ):
        self.df = dataframe
        self.dir = directory
        self.transform = transform

        # Create label-to-index mapping inside dataset class
        self.class_to_idx = {class_name:idx for idx, class_name in enumerate(sorted(self.df["subCategory"].unique()))}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()} # reverse mapping

        print(f"Class Mapping: {self.class_to_idx}") # checking the mapping

        # Split the data into training and validation
        self.validation_split = 0.2
        self.total_size = len(self.df)
        self.val_size = int(self.validation_split * self.total_size)

        if subset == "training":
            self.df = self.df.iloc[self.val_size:]
        else: # validation
            self.df = self.df.iloc[:self.val_size]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.dir, self.df.iloc[idx]['image']) # get the image name
        image = Image.open(img_name).convert('RGB') # convert the image to RGB format
        label = self.df.iloc[idx]['subCategory'] # get the label of the image
        label = self.class_to_idx[label] # converts to integer

        if self.transform:
            image = self.transform(image)

        return image, label