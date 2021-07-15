import argparse, logging, time, multiprocessing, sys, traceback
import torch 
from os import name
from pathlib import Path
from train import train_model


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
    parser.add_argument('--segmentationType', dest='segmentationType', type=str,
                        help='segmentation type', required=True)
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str,
                        help='path to pretrained model', required=False)
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
                        help='Collection containing labels', required=False)
    parser.add_argument('--flowfieldDir', dest='flowfieldDir', type=str,
                        help='Collection containing vector fields', required=False)
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
    segmentationType = args.segmentationType 
    logger.info('segmentationType = {}'.format(segmentationType))
    pretrainedModel = args.pretrainedModel 
    logger.info('pretrainedModel = {}'.format(pretrainedModel))
    modelName = args.modelName if args.modelName!=None else 'unet'
    modelName = modelName if pretrainedModel==None else None
    logger.info('modelName = {}'.format(modelName))
    encoderName = args.encoderName if args.encoderName!=None else 'resnet34'
    encoderName = encoderName if pretrainedModel==None else None
    logger.info('encoderName = {}'.format(encoderName))
    encoderWeights = 'imagenet' if args.encoderWeights=='imagenet' else None
    encoderWeights = encoderWeights if pretrainedModel==None else None
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
    if labelsDir != None:
        if (Path.is_dir(Path(args.labelsDir).joinpath('images'))):
            # switch to images folder if present
            fpath = str(Path(args.labelsDir).joinpath('images').absolute())
    logger.info('labelsDir = {}'.format(labelsDir))
    flowfieldDir = args.flowfieldDir
    logger.info('flowfieldDir = {}'.format(flowfieldDir))
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

    # if pretrained model is provided
    checkpoint = None
    if pretrainedModel != None:
        checkpoint = torch.load(Path(pretrainedModel).joinpath('checkpoint.pt'))
        modelName = checkpoint['model']
        encoderName = checkpoint['encoder']
        encoderWeights = None

    # changes based on segmentation type
    labelsDir = labelsDir if segmentationType == 'Binary' else flowfieldDir
    classes = 1 if segmentationType == 'Binary' else 3
    activation = 'sigmoid' if segmentationType == 'Binary' else None
    logger.info('classes: {}'.format(classes))
    logger.info('activation: {}'.format(activation))
        
    # Surround with try/finally for proper error catching
    try:
        train_model(imagesDir,labelsDir,imagesPattern,labelsPattern,trainValSplit,
                    modelName,encoderName,encoderWeights,loss,metric,classes,
                    activation,pretrainedModel,batchSize,checkpoint,outDir)

    except Exception:
        traceback.print_exc()

    finally:
        # Exit the program
        logger.info('Exiting workflow...')
        sys.exit()