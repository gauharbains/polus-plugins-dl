import os, typing
from pathlib import Path
import numpy as np
import filepattern
import torch 
import torchvision
import gpustat
import segmentation_models_pytorch as smp
from bfio import BioReader, BioWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from utils.preprocess import LocalNorm

tile_size = 256

def get_image_labels_mapping(images_fp, labels_fp):
    """creates a filename map between images and labels
    In the case where image filenames have different filename 
    pattern than label filenames, this function creates a map
    between the corresponding images and labels
    
    Args:
        images_fp ([FilePattern object]): filepattern object for images
        labels_fp ([FilePattern object]): filepattern object for labels

    Returns:
        dict: dictionary containing mapping b/w image & label names
    """
    name_map = {}

    for f in images_fp():
        image_name = f[0]['file']
        vars = {k.upper():v for k,v in f[0].items() if k!='file' }
        label_name = labels_fp.get_matching(**vars)[0]['file']
        name_map[image_name] = label_name
    return name_map


def get_tile_mapping(image_names):
    """ creates a tile map for the Dataset class
    This function iterates over all the files in the input 
    collection and creates a dictionary that can be used in 
    __getitem__ function in the Dataset class. 
    
    Args:
        image_names (list): list containing path to images
        
    Returns:
        dict: dictionary containing tile mapping
    """
    tile_map = {}
    tile_num = 0

    # iterate over all files
    for file_name in image_names:
        with BioReader(file_name) as br:
            
            # iterate over tiles
            for x in range(0,br.X,tile_size):
                x_max = min([br.X,x+tile_size])
                for y in range(0,br.Y, tile_size):
                    y_max = min([br.Y,y+tile_size])

                    # add tile to tile_map
                    tile_map[tile_num] = (file_name, (x,x_max), (y,y_max))
                    tile_num+=1
    return tile_map


def get_max_batch_size(model, tile_size, device):
    """get max possible batch_size based on GPU memory
    This function calculates the maximum possible batch
    size based on the available GPU memory. 

    Args:
        model (pytorch model): model to train
        tile_size (int): size of the input image tile
        device (int): device id

    Returns:
        int: max possible batch size
    """

    # get number of trainable parameters
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            total_param += num_param
    
    # get available GPU memory
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    gpu_mem = (item["memory.total"] - item["memory.used"])*1E6*1.04858

    # max batch size
    max_batch_size = int(gpu_mem/(6*(total_param+tile_size*tile_size))) - 1
    if max_batch_size < 1:
        max_batch_size = 1
    return max_batch_size


class Dataset(BaseDataset):
    
    def __init__(self,
        image_label_map,
        tile_map
    ):

        self.image_label_map = image_label_map
        self.tile_map = tile_map
        self.preprocessing = torchvision.transforms.Compose([
                             torchvision.transforms.ToTensor(),
                             LocalNorm()])

    def __getitem__(self, i):
        image_name = self.tile_map[i][0]
        x, x_max = self.tile_map[i][1]
        y, y_max = self.tile_map[i][2]

        # read and preprocess image
        with BioReader(image_name) as br:
            img = br[y:y_max,x:x_max,0:1,0,0][:,:,0,0,0]
        img = img.astype(np.float32)
        img = self.preprocessing(img).numpy()

        # read and preprocess label
        label_name = self.image_label_map[image_name]
        with BioReader(label_name) as br:
            label = br[y:y_max,x:x_max,0:1,0,0][:,:,0,0,0]      
        label[label>=1] = 1
        label[label<1] = 0
        label = label.astype(np.float32)
        label = label.reshape((1,label.shape[0], label.shape[1]))
        return img, label
        
    def __len__(self):
        return len(self.tile_map.keys())
    







        


    