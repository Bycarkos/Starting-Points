import numpy as np
import random
from torchvision.models import resnet18, ResNet18_Weights


from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import pandas as pd


class Baseline(nn.Module):
    
    def __init__(self, num_char:int, hidden_dimension: int=768, char_dimension: int=768, text_max_len: int=30):
        
        super(Baseline, self).__init__()
        
        self.text_max_len = text_max_len
        self.hidden_dimension = hidden_dimension
        self.char_dimension = char_dimension
        self.num_char = num_char
        self.num_layers = 2
        
        
        resnet =  resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        decoder_input_dimension = resnet.fc.in_features

        modules=list(resnet.children())[:-1]


        self.encoder = nn.Sequential(*modules)

        ## decoder Part
        self.rnn = nn.GRU(char_dimension, decoder_input_dimension, num_layers=self.num_layers, batch_first=True)
        self.proj = nn.Linear(decoder_input_dimension, num_char)
        
        self.lup = nn.Embedding(num_char, char_dimension)
        
    ## Forward function for Image Captioning Autorregressive model
    def forward(self, x: torch.Tensor, token_labels: torch.Tensor, teacher_forcing: float = 0.5):
        batch_size, channels, height, width = x.shape

        # Process each image in the sequence
        outputs = []
        hidden_state = self.encoder(x).flatten(1, -1)  # Shape: (batch_size, feature_dim)
        hidden_state = hidden_state.repeat(self.num_layers, 1, 1) # Shape: (D*num_layers, batch_size, feature_dim)
        
        maximun_iterations = min(self.text_max_len, token_labels.shape[1])
        
        for t in range(maximun_iterations):
            # Encode the image

            # Predict next character
            if (random.random() < teacher_forcing) and (t != 0):
                tokens = torch.argmax(char_logits, dim=1)
            else:
                tokens = token_labels[:, t]  # Shape: (batch_size,)
                

            decoder_input = self.lup(tokens).unsqueeze(1)  # Shape: (batch_size, 1, char_dimension)

            # Pass through GRU
            rnn_output, hidden_state = self.rnn(decoder_input, hidden_state)  # Shape: (num_layers, seq_len, hidden_dim)

            # Project to character space
            char_logits = self.proj(rnn_output[:, -1, :].relu())  # Shape: (batch_size, num_char)

            # Store output
            outputs.append(char_logits)

        # Stack outputs along the sequence dimension
        return torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, num_char)


        
        
if __name__ == "__main__":
    # Example usage
    model = Baseline(num_char=10)
    print(model)

    # Create a random input tensor with shape (batch_size, seq_len, channels, height, width)
    input_tensor = torch.randn(32, 3, 224, 224)  # Example shape
    tokens = torch.randint(size=(32, 5), high=10)
    output = model(input_tensor, tokens)
    print(output.shape)  # Should be (batch_size, seq_len, num_char)
    
    