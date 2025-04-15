# Example for creating a synthetic dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os


example_smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC1=C(C(=O)C2=C(C1=O)C=C(C)C=C2OC)C",  # Ubiquinone
    "C1=CC=C(C=C1)C=O",  # Benzaldehyde
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "C1CCCCC1",  # Cyclohexane
    "C1=CC=CC=C1",  # Benzene
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "CCCC",  # Butane
    "C(C(=O)O)N",  # Glycine
    "CC(C)O",  # Isopropanol
    "CCN(CC)CC",  # Triethylamine
    "O=C=O",  # Carbon dioxide
    "N#N",  # Nitrogen
    "CC(C)C(=O)O",  # Isobutyric acid
    "COC",  # Dimethyl ether
    "CCOC(=O)C",  # Ethyl acetate
    "CC(C)C",  # Isobutane
    "CC(C)CO",  # Butanol
    "CC1=CC=CC=C1",  # Toluene
    "CC(C)(C)O",  # tert-Butanol
    "C1=CC=C(C=C1)O",  # Phenol
    "C1=CC(=CC=C1O)O",  # Hydroquinone
    "CCOC(=O)C1=CC=CC=C1",  # Ethyl benzoate
    "CCN(CC)C(=O)C1=CC=CC=C1",  # Lidocaine (simplified)
    "C1=CC(=C(C=C1Cl)Cl)O",  # 2,4-Dichlorophenol
    "CC(C)NC(=O)C1=CC=CC=C1",  # Acetanilide
    "CCOC(=O)C(C)N",  # Methyl 2-aminobutanoate
    "CC(=O)NC1=CC=CC=C1",  # Acetanilide
    "COC1=CC=CC=C1OC",  # Anisole
    "CC(C)OC(=O)C1=CC=CC=C1",  # Ibuprofen methyl ester
    "C1=CC(=CC=C1N)Cl",  # 4-Chloroaniline
    "COC(=O)C",  # Methyl acetate
    "CCN(CC)C",  # Diethylamine
    "CC(C)C1=CC=CC=C1",  # Cumene
    "CCC(=O)O",  # Propionic acid
    "CC(C)(C)C(=O)O",  # Pivalic acid
    "CC(C)OC",  # Isopropyl methyl ether
    "CCOC(=O)C2=CC=CC=C2",  # Ethyl benzoate
    "COC(=O)C1=CC=CC=C1",  # Methyl benzoate
    "CN(C)C(=O)NC1=CC=CC=C1",  # Paracetamol
    "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
    "COC1=CC=CC=C1OC",  # Dimethoxybenzene
    "C1=CC=C(C=C1)N",  # Aniline
    "CN(C)C",  # Trimethylamine
    "COC(=O)N",  # Methyl carbamate
    "CC(C)C(C(=O)O)N",  # Leucine
    "CCC(=O)OC",  # Propyl acetate
    "C(CO)O",  # Ethylene glycol
    "O=C(NC1=CC=CC=C1)C2=CC=CC=C2",  # Benzanilide
    "CC(C)NC(=O)C(C)N",  # Valine
    "CC(=O)NC(C)C",  # Alanine amide
    "CC1=CC=C(C=C1)N",  # p-Toluidine
    "COC(=O)NC1=CC=CC=C1",  # Methyl aniline carbamate
    "CCOC(=O)NC",  # Ethyl carbamate
    "CCOC(=O)N",  # Ethyl carbamate
    "CCC(C)CO",  # 2-Methylbutanol
    "CC(C)COC(=O)C",  # Isobutyl acetate
    "CN1C=CN=C1",  # Methylimidazole
    "C1=CC=C(C=C1)CN",  # Benzylamine
    "CC(C)(C)CN",  # tert-Butylamine
    "CCCN",  # Propylamine
    "CC(C)(O)C(=O)O",  # tert-Butyl glycolic acid
    "C1=CC(=O)OC=C1",  # 1,2-Benzoquinone
    "CC(C(=O)O)N",  # Alanine
    "C(C(C(=O)O)N)O",  # Serine
    "C1CCOC1",  # Tetrahydrofuran (THF)
    "C1=NC=CN=C1",  # Pyrimidine
    "C1=CC=CN=C1",  # Pyridine
    "C1=CC=C2C=CC=CC2=C1",  # Naphthalene
    "C1=CC=C3C(=C1)C=CC2=CC=CC=C23",  # Anthracene
    "CC1=CN=CN1",  # Methylpyrazole
    "CCN(C)C",  # N,N-Dimethylethylamine
    "CCOC",  # Ethyl methyl ether
    "CNC(=O)C",  # Acetamide
    "CN(C)C=O",  # Dimethylformamide
    "COC1=CC=CC=C1",  # Anisole
    "COC(=O)OC",  # Dimethyl carbonate
    "C1=CC=C(C=C1)CO",  # Benzyl alcohol
    "C1=CC=C(C=C1)CBr",  # Benzyl bromide
    "C1=CC=C(C=C1)Cl",  # Chlorobenzene
    "C1=CC=C(C=C1)F",  # Fluorobenzene
    "C1=CC=C(C=C1)I",  # Iodobenzene
    "C1=CC=C(C=C1)CN(C)C",  # N,N-Dimethylbenzylamine
    "C1=CC=C(C=C1)C(C)N",  # Phenylethanolamine
    "C1=CC=C(C=C1)C(=O)N",  # Benzamide
    "CN(C)C(=O)C1=CC=CC=C1",  # Lidocaine core
    "COC(=O)C(C)O",  # Methyl lactate
    "CC(C)C(=O)OC",  # Methyl isobutyrate
    "C1CNCCN1",  # Piperazine
    "C1CCNCC1",  # Piperidine
    "C1CCOC1",  # Tetrahydrofuran
    "C1COCCO1",  # 1,3-Dioxolane
    "CCOCC",  # Diethyl ether
    "C1=CC=C(C=C1)N(=O)=O",  # Nitrobenzene
    "C1=CC=C(C=C1)C(C)O",  # Phenylethanol
    "CN(C)CCOC(=O)C",  # Procaine
    "C1=CC=C(C=C1)CC(=O)O",  # Phenylacetic acid
    "CC(C)C1=CC=CC=C1OC",  # Isobutylbenzene
    "CN1C=CN=C1C",  # 1-Methylimidazole
    "C1CCC(CC1)N",  # Cyclohexylamine
    "CCC(C)OC",  # Propyl isopropyl ether
    "C1=CC2=CC=CC=C2C=C1",  # Biphenyl
    "CC1=CC=CC=C1C",  # o-Xylene
    "CC1=CC=CC=C1C(C)O",  # o-Methylphenylethanol
    "CN(C)C(=O)CO",  # Dimethylacetamide
    "CCN(CC)C(=O)OC",  # Lidocaine simplified ester
]


def create_synthetic_dataset(output_dir='molecule_dataset', num_samples=1000):
    """
    Create a synthetic dataset of molecule images and corresponding SMILES
    This is just a placeholder - in reality, you'd need real molecular data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'images')):
        os.makedirs(os.path.join(output_dir, 'images'))
    
    
    smiles_list = []
    for i in range(num_samples):
        # Use one of the example molecules or generate random ones
        if i < len(example_smiles):
            smiles = example_smiles[i]
        else:
            # In reality, you'd generate valid molecules
            # This is just a placeholder
            idx = i % len(example_smiles)
            smiles = example_smiles[idx]
        
        # Generate molecule image
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img_path = os.path.join(output_dir, 'images', f'molecule_{i}.png')
            img = Draw.MolToImage(mol, size=(300, 300))
            img.save(img_path)
            smiles_list.append([f'molecule_{i}.png', smiles])
    
    # Save CSV file
    df = pd.DataFrame(smiles_list, columns=['image', 'smiles'])
    df.to_csv(os.path.join(output_dir, 'molecules.csv'), index=False)
    
    print(f"Created {len(smiles_list)} synthetic molecule samples")
    return os.path.join(output_dir, 'images'), os.path.join(output_dir, 'molecules_synthetic.csv')





# Custom collate function
def collate_fn(batch):
    # Sort the batch by sequence length (descending) for packed sequence
    batch.sort(key=lambda x: x['length'], reverse=True)
    
    # Get individual components
    images = torch.stack([item['image'] for item in batch])
    smiles = torch.stack([item['smiles'] for item in batch])
    lengths = [item['length'] for item in batch]
    
    return {
        'image': images,
        'smiles': smiles,
        'length': lengths
    }
    
    
    
    
# Update DataLoader to use the custom collate_fn
def prepare_data(image_dir, csv_file, collator_train_kwargs:dict, collator_test_kwargs: dict)
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Create the dataset and data loader
    dataset = MoleculeImageDataset(image_dir=image_dir, csv_file=csv_file, transform=transform)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use the custom collate_fn
    train_loader = DataLoader(
        train_dataset, 
        **collator_train_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        **collator_test_kwargs
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    
    create_synthetic_dataset(output_dir=".", num_samples=500)