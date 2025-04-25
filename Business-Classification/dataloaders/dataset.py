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



class ConTextDataset(Dataset):
    def __init__(self, json_file, root_dir, root_dir_txt, train=True, transform=None):
        raise NotImplementedError("This dataset is not implemented yet.")


    def __len__(self):
        raise NotImplementedError("This dataset is not implemented yet.")
    def __getitem__(self, idx):
        raise NotImplementedError("This dataset is not implemented yet.")

        #return image, text, text_mask, target
