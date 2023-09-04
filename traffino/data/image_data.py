# Imports
import torch
import logging
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from torch.autograd import Variable



logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        list_img_seq, 
        list_ten_img_seq, 
    ) = zip(*data)

    _len = [len(seq) for seq in list_img_seq]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    list_img = torch.cat(list_img_seq, dim=0)
    list_ten_img = torch.cat(list_ten_img_seq, dim=0)
    
    out = [
        list_img, 
        list_ten_img, 
    ]

    return tuple(out)


    
class ImageDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args: 
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        
        img_list=[]
        img_tensor_list=[]
        
        all_files = os.listdir(self.main_dir)
        
        for file in all_files:
            path = main_dir+file
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                tensor_image = self.transform(image)
            img_list.append(image)
            img_tensor_list.append(tensor_image)

            
        
        self.list_img = img_list
        self.list_ten_img = img_tensor_list

        

            
        
    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        out =  [
                self.list_img,
                self.list_ten_img
        ]

        return out