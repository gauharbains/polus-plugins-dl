import argparse, logging, subprocess, time, multiprocessing
import filepattern
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.losses as losses
import utils.dataloader as dl
from pathlib import Path
from torch.utils.data import DataLoader
from utils.params import models_dict,loss_dict, metric_dict
from segmentation_models_pytorch.utils.base import Activation
from bfio.bfio import BioReader, BioWriter
from sklearn.model_selection import train_test_split

tile_size = 256

def train_model(imagesDir,labelsDir,imagesPattern,labelsPattern,trainValSplit,
                modelName,encoderName,encoderWeights,loss,metric,classes,
                activation,pretrainedModel,batchSize,checkpoint,outDir):
    """ train a model using the segmentation models toolkit

    Args:
        imagesDir (path): path to input image collection
        labelsDir (path): path to labels collection
        imagesPattern (str): filepattern for image collection
        labelsPattern (str): filepattern for label collection
        trainValSplit (float): ratio to split data into train/val
        modelName (str): name of the model
        encoderName (str): encoder name
        encoderWeights (str): pretrained weights to use
        loss (str): loss function for training
        metric (str): performance metric to use
        classes (int): output classes to predict
        activation (str): activation function
        pretrainedModel (path): path to the pretrained model, if any
        batchSize (int): batchSize for training
        checkpoint (.pth file): state dict for pretrained model
        outDir (str): output path
    """

    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    # intialize filepattern objects
    images_fp = filepattern.FilePattern(file_path=imagesDir, pattern=imagesPattern)
    labels_fp = filepattern.FilePattern(file_path=labelsDir, pattern=labelsPattern)
    
    # map images to corresponding masks
    name_map = dl.get_image_labels_mapping(images_fp, labels_fp)

    # train validation split
    image_names = list(name_map.keys())
    train_names, val_names = train_test_split(image_names, shuffle=True, train_size=trainValSplit)
    train_map = {k:name_map[k] for k in train_names}
    val_map = {k:name_map[k] for k in val_names}

    # get tile maps
    train_tile_map = dl.get_tile_mapping(train_names)
    val_tile_map = dl.get_tile_mapping(val_names)

    # intiliaze model and training parameters
    logger.info('Initializing model,loss,metric')
    model_class = models_dict[modelName]
    loss_class = loss_dict[loss]()
    metric_class = [metric_dict[metric](threshold=0.5)]

    model = model_class(
        encoder_name=encoderName,        
        encoder_weights=encoderWeights,     
        in_channels=1,                  
        classes=classes,   
        activation=activation                   
    )

    # get max_batch size if needed
    if torch.cuda.is_available() and batchSize == None:
        logger.info('No batchSize was specified, calculating max size possible')
        batchSize = dl.get_max_batch_size(model=model, tile_size=tile_size,
                                            device=0, classes=classes)
        logger.info('max batch size: {}'.format(batchSize))
        
    # initialize datalaoder
    logger.info('Initializing dataloader')
    train_data = dl.Dataset(train_map, train_tile_map, classes)
    val_data = dl.Dataset(val_map, val_tile_map, classes)
    train_loader = DataLoader(train_data, batch_size=batchSize)
    val_loader = DataLoader(val_data, batch_size=batchSize)

    # device
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info('using device: {}'.format(dev))

    # optimizer
    optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
    ])

    # load model state and optimizer state 
    if pretrainedModel:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    # train and val iterator
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss_class, 
        metrics=metric_class, 
        optimizer=optimizer,
        device=dev,
        verbose=False
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss_class, 
        metrics=metric_class, 
        device=dev,
        verbose=False
    )

    last_val_loss = 1E5
    patience = 5
    i = 0
    # train and save model
    while True:
        i+=1
        logger.info('----- Epoch: {} -----'.format(i))
        if pretrainedModel:
            logger.info('Pretrained + current epochs = {}'.format(epoch+i))

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # print logs
        train_str = ', '.join(['{}: {}'.format(k,train_logs[k]) for k in train_logs.keys()])
        val_str = ', '.join(['{}: {}'.format(k,valid_logs[k]) for k in valid_logs.keys()])
        logger.info('Train logs --- ' + train_str)
        logger.info('Val logs --- ' + val_str)

        # early stopping
        val_loss = valid_logs[loss_class.__name__]

        if val_loss < last_val_loss:
            val_count = 0
            last_val_loss = val_loss
        else:
            val_count += 1
        
        if val_count >= patience:
            logger.info('executing early stopping..')
            break
        
    # save model
    logger.info('saving model...')
    torch.save(model, Path(outDir).joinpath('out_model.pth'))

    # save checkpoint
    checkpoint = {
        'model': modelName,
        'encoder': encoderName,
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, Path(outDir).joinpath('checkpoint.pt'))