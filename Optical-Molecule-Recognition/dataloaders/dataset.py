import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from utils import collate_fn

class MoleculeImageDataset(Dataset):
    
    SMILES_CHARS = ' ()[].,=#-1234567890BCFHNOPSclnors+'  # Common characters in SMILES
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(SMILES_CHARS)}
    IDX_TO_CHAR = {idx: char for idx, char in enumerate(SMILES_CHARS)}
    MAX_SMILES_LENGTH = 100
    
    def __init__(self, image_dir, csv_file, transform=None):
        """
        Args:
            image_dir (string): Directory with all the molecule images
            csv_file (string): Path to the csv file with image names and SMILES
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        smiles = self.annotations.iloc[idx, 1]
        # Convert SMILES to sequence of indices
        smiles_encoded = [self.CHAR_TO_IDX.get(c, 0) for c in smiles]
        # Pad if necessary
        smiles_encoded = smiles_encoded + [0] * (self.MAX_SMILES_LENGTH - len(smiles_encoded))
        smiles_encoded = smiles_encoded[:self.MAX_SMILES_LENGTH]  # Truncate if too long
        
        return {
            'image': image,
            'smiles': torch.tensor(smiles_encoded, dtype=torch.long),
            'length': min(len(smiles), self.MAX_SMILES_LENGTH)
        }
        
        
        
if __name__ == "__main__":
    # Define the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    

    dataset = MoleculeImageDataset(image_dir='Optical-Molecule-Recognition/images', csv_file='Optical-Molecule-Recognition/molecules.csv', transform=transform)

    # Create a data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Iterate over the data loader
    for batch in dataloader:
        import pdb;pdb.set_trace()
        
        images = batch['image']
        smiles = batch['smiles']
        lengths = batch['length']

        # Perform some operations on the batch
        # For example, print the shapes of the images and smiles tensors
        print("Image shapes:", images.shape)
        print("Smiles shapes:", smiles.shape)
        print("Lengths:", lengths)