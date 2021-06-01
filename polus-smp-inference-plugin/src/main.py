from bfio import BioReader, BioWriter
import argparse, logging, subprocess, time, multiprocessing, sys, os, typing
import traceback
import numpy as np
from pathlib import Path
import torch 
import torchvision
from preprocess import LocalNorm
import filepattern

tile_size = 1024

def pad_image(img, out_shape=(tile_size,tile_size)):
    """Pad input image to make it a certain zie

    Args:
        img (array): input image
        out_shape (tuple, optional): Desired output shape

    Returns:
        array: padded image
        tuple: tuple consisting of pad dimensions
    """

    pad_x = img.shape[0] - out_shape[0]
    pad_y = img.shape[1] - out_shape[1]
    padded_img = np.pad(img, [(0,pad_x),(0,pad_y)], mode='reflect') 
    return padded_img, (pad_x,pad_y)

    
if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Segmentation models inference plugin')
    
    # Input arguments
    parser.add_argument('--pattern', dest='pattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--modelPath', dest='modelPath', type=str,
                        help='pretrained model to use', required=True)

    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    pattern = args.pattern
    logger.info('pattern = {}'.format(pattern))
    modelPath = args.modelPath
    logger.info('modelPath = {}'.format(modelPath))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    # pretrained model
    model_path = os.path.join(modelPath, 'out_model.pth')

    # initialize preprocessing
    preprocess = torchvision.transforms.Compose([
           torchvision.transforms.ToTensor(),
           LocalNorm() 
           ])

    # Surround with try/finally for proper error catching
    try:
        # device
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info('using device: {}'.format(dev))

        # load model
        logger.info('loading pretrained model')
        model = torch.load(model_path).to(dev)
        model.eval()

        fp = filepattern.FilePattern(file_path=inpDir, pattern=pattern)
         
        # Loop through files in inpDir image collection and process
        for f in fp():
            file_name = f[0]['file']
            logger.info('Processing image: {}'.format(file_name.name))

            with BioReader(file_name) as br, \
                 BioWriter(Path(outDir).joinpath(Path(file_name).name), metadata=br.metadata) as bw:
                bw.dtype = np.uint8
                
                # iterate over tiles
                for x in range(0,br.X,tile_size):
                    x_max = min([br.X,x+tile_size])

                    for y in range(0,br.Y,tile_size):
                        y_max = min([br.Y,y+tile_size])
                        img = br[y:y_max,x:x_max,0:1,0,0][:,:,0,0,0]

                        # pad image if required
                        pad_dims = None
                        if not (img.shape[0]//tile_size==1 and img.shape[1]//tile_size==1):
                            img, pad_dims = pad_image(img)
                        
                        # preprocess image
                        img = img.astype(np.float32)
                        img = preprocess(img).unsqueeze(0).to(dev)

                        with torch.no_grad():
                            out = model(img).cpu().numpy()
                        
                        # postpreocessing and write tile
                        out[out>=0.5] = 255
                        out[out<0.5] = 0
                        out = out[0,0,:-pad_dims[0],:-pad_dims[1]] if pad_dims!=None else out[0,0,:,:]
                        bw[y:y_max,x:x_max,0:1,0,0] = out.astype(np.uint8)

    except Exception:
        traceback.print_exc()
  
    finally:
        logger.info('Finished Execution')
        # Exit the program
        sys.exit()