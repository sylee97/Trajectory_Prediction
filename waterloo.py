

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
#___________________________________________________________________________________________________________________________

### Dataset class for Waterloo dataset

class waterlooDataset(Dataset):
    
    def __init__(self, data_dir):
        self.dataroot=data_dir 
    
    def __getitem__(self, idx):
        pass