import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class MoleculeImageDataset(Dataset):
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
        smiles_encoded = [CHAR_TO_IDX.get(c, 0) for c in smiles]
        # Pad if necessary
        smiles_encoded = smiles_encoded + [0] * (MAX_SMILES_LENGTH - len(smiles_encoded))
        smiles_encoded = smiles_encoded[:MAX_SMILES_LENGTH]  # Truncate if too long
        
        return {
            'image': image,
            'smiles': torch.tensor(smiles_encoded, dtype=torch.long),
            'length': min(len(smiles), MAX_SMILES_LENGTH)
        }