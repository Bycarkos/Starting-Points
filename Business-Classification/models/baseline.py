
import time,os,json
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset

from einops import rearrange

import fasttext
import fasttext.util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)

class ConTextTransformer(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        modules=list(resnet50.children())[:-2]
        self.resnet50=nn.Sequential(*modules)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.num_cnn_features = 64  # 8x8
        self.dim_cnn_features = 2048
        self.dim_fasttext_features = 300

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_cnn_features + 1, dim))
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim)
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        encoder_norm = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, txt, mask=None):
        x = self.resnet50(img)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.cnn_feature_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x2 = self.fasttext_feature_to_embedding(txt.float())
        x = torch.cat((x,x2), dim=1)

        #tmp_mask = torch.zeros((img.shape[0], 1+self.num_cnn_features), dtype=torch.bool)
        #mask = torch.cat((tmp_mask.to(device), mask), dim=1)
        #x = self.transformer(x, src_key_padding_mask=mask)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)