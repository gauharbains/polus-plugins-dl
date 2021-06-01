import argparse, logging, subprocess, time, multiprocessing, sys, traceback
from os import name
import numpy as np
from segmentation_models_pytorch.utils import train
import torch 
import segmentation_models_pytorch as smp
import filepattern
import utils.dataloader as dl
from sklearn.model_selection import train_test_split
from pathlib import Path
from bfio.bfio import BioReader, BioWriter
from torch.utils.data import DataLoader
from utils.params import models_dict,loss_dict, metric_dict
from segmentation_models_pytorch.utils.base import Activation

tile_size = 256

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')
    
    # Input arguments
    parser.add_argument('--modelName', dest='modelName', type=str,
                        help='model to use', required=False)
    parser.add_argument('--encoderName', dest='encoderName', type=str,
                        help='encoder to use', required=False)
    parser.add_argument('--encoderWeights', dest='encoderWeights', type=str,
                        help='Pretrained weights for the encoder', required=False)
    parser.add_argument('--imagesPattern', dest='imagesPattern', type=str,
                        help='Filename pattern for images', required=True)
    parser.add_argument('--labelsPattern', dest='labelsPattern', type=str,
                        help='Filename pattern for labels', required=True)
    parser.add_argument('--imagesDir', dest='imagesDir', type=str,
                        help='Collection containing images', required=True)
    parser.add_argument('--labelsDir', dest='labelsDir', type=str,
                        help='Collection containing labels', required=True)
    parser.add_argument('--loss', dest='loss', type=str,
                        help='Loss function', required=False)
    parser.add_argument('--metric', dest='metric', type=str,
                        help='Performance metric', required=False)
    parser.add_argument('--batchSize', dest='batchSize', type=str,
                        help='batchSize', required=False)
    parser.add_argument('--trainValSplit', dest='trainValSplit', type=str,
                        help='trainValSplit', required=False)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    modelName = args.modelName if args.modelName!=None else 'unet'
    logger.info('modelName = {}'.format(modelName))
    encoderName = args.encoderName if args.encoderName!=None else 'resnet34'
    logger.info('encoderName = {}'.format(encoderName))
    encoderWeights = 'imagenet' if args.encoderWeights=='imagenet' else None
    logger.info('encoderWeights = {}'.format(encoderWeights))
    imagesPattern = args.imagesPattern
    logger.info('imagesPattern = {}'.format(imagesPattern))
    labelsPattern = args.labelsPattern
    logger.info('labelsPattern = {}'.format(labelsPattern))
    imagesDir = args.imagesDir
    if (Path.is_dir(Path(args.imagesDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.imagesDir).joinpath('images').absolute())
    logger.info('imagesDir = {}'.format(imagesDir))
    labelsDir = args.labelsDir
    if (Path.is_dir(Path(args.labelsDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.labelsDir).joinpath('images').absolute())
    logger.info('labelsDir = {}'.format(labelsDir))
    loss = args.loss if args.loss!=None else 'Dice'
    logger.info('loss = {}'.format(loss))
    metric = args.metric if args.metric!=None else 'IoU'
    logger.info('metric = {}'.format(metric))
    batchSize = int(args.batchSize) if args.batchSize!=None else None
    logger.info('batchSize = {}'.format(batchSize))
    trainValSplit = float(args.trainValSplit) if args.trainValSplit!=None else 0.7
    logger.info('trainValSplit = {}'.format(trainValSplit))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # Surround with try/finally for proper error catching
    try:
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
            classes=1,   
            activation='sigmoid'                   
        )

        # get max_batch size if needed
        if torch.cuda.is_available() and batchSize == None:
            logger.info('No batchSize was specified, calculating max size possible')
            batchSize = dl.get_max_batch_size(model,tile_size,0)
            logger.info('max batch size: {}'.format(batchSize))
            
        # initialize datalaoder
        logger.info('Initializing dataloader')
        train_data = dl.Dataset(train_map, train_tile_map)
        val_data = dl.Dataset(val_map, val_tile_map)
        train_loader = DataLoader(train_data, batch_size=batchSize)
        val_loader = DataLoader(val_data, batch_size=batchSize)

        # device
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info('using device: {}'.format(dev))

        # optimizer
        optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
        ])

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

        last_val_accuracy = -1E5
        patience = 5
        i = 0
        # train and save model
        while True:
            i+=1
            logger.info('----- Epoch: {} -----'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)

            # print logs
            train_str = ', '.join(['{}: {}'.format(k,train_logs[k]) for k in train_logs.keys()])
            val_str = ', '.join(['{}: {}'.format(k,valid_logs[k]) for k in valid_logs.keys()])
            logger.info('Train logs --- ' + train_str)
            logger.info('Val logs --- ' + val_str)

            # early stopping
            val_accuracy = valid_logs[metric_class[0].__name__]
            if val_accuracy > last_val_accuracy:
                val_count = 0
                last_val_accuracy = val_accuracy
            else:
                val_count += 1
            
            if val_count >= patience:
                logger.info('executing early stopping..')
                break

        # save model
        logger.info('saving model...')
        torch.save(model, Path(outDir).joinpath('out_model.pth'))

    except Exception:
        traceback.print_exc()

    finally:
        # Exit the program
        logger.info('Exiting workflow...')
        sys.exit()