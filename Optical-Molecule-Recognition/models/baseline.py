import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models



class MoleculeEncoder(nn.Module):
    def __init__(self, embed_size=256):
        super(MoleculeEncoder, self).__init__()
        # Use a pre-trained ResNet but modify it for our grayscale input
        resnet = models.resnet34(pretrained=True)
        # Modify the first layer to accept grayscale (1 channel) input
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final FC layer
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Add a custom pooling and FC layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        
    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features
    
    
    
# Define the Decoder (LSTM)
class SMILESDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(SMILESDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        # Concatenate image features with embedded captions
        embeddings = torch.cat((embeddings, features.unsqueeze(1)), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
    
    
# Full model combining encoder and decoder
class MoleculeToSMILES(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(MoleculeToSMILES, self).__init__()
        self.encoder = MoleculeEncoder(embed_size)
        self.decoder = SMILESDecoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    
    
if __name__ == "__main__":
        # Example usage
    model = MoleculeToSMILES(embed_size=768, hidden_size=768, vocab_size=10, num_layers=2)
    print(model)

    # Create a random input tensor with shape (batch_size, seq_len, channels, height, width)
    input_tensor = torch.randn(32, 3, 224, 224)  # Example shape
    tokens = torch.randint(size=(32, 5), high=10)
    output = model(input_tensor, tokens, lengths=torch.tensor([5 for i in range(32)]))
    print(output.shape)  # Should be (batch_size, seq_len, num_char)
    