# SMP Training

This WIPP plugin uses the [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) toolkit to train models for image segmentation. The toolkit is a high level API consisting of 9 models architectures for binary and multiclass segmentation. There are more than a 100 available encoders with pre-trained wieghts which can be used as backbones for these arcitectures. 
  
Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) or [Nick Schaub](mailto:nick.schaub@labshare.org) for more information.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Core Functionality
The details and usage of the plugin is provided in the section below. The repository consists of twp plugin manifests, one each for binary and cellpose type segmentation but they both use the same codebase. 
 
- **Binary Segmentation**: The plugin can be use to train models for binary segmentation. To do this, select segmentationType as Binary and provide path to the binary labels in `labelsDir`.
- **Cellpose Segmnetation**: This functionality is still a work in progress. To enable the plugin to train models for cellpose type segmentation, select segmentationType as Cellpose and provide path to ground truth vectors in `flowfieldDir`. 
- **Train pre-trained models**: Models previously trained and saved using this plugin can be fed into this plugin to continue training. This can be done for both binary and cellpose type segmentation. Provide path to the pre-trained model in `preTrainedModel`. 


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   | Required | Default |
|---------------|-------------------------|--------|--------|----------|---------|
| `--segmentationType` | Type of segmentation | Input | string | Yes | - |
| `--imagesDir` | Collection containing input images | Input | collection | Yes | - |
| `--imagesPattern` | filepattern for input images | Input | string | Yes | - |
| `--labelsPattern` | filepattern for ground truth | Input | string | Yes | - |
| `--preTrainedModel` | pretrained model, if any| Input | genericData | no | None |
| `--labelsDir` | path to binary labels | Input | collection | Yes for binary seg. | - |
| `--flowfieldDir` | path to cellpose labels | Input | genericData | Yes for cellpose seg. | - |
| `--modelName` | model to use | Input | string | No | unet|
| `--encoderName` | encoder to use | Input | string | No | resnet34|
| `--encoderWeights` | Pretrained weights for the encoder | Input | string | No | Random|
| `--loss` | Loss function to use | Input | string | No | Dice |
| `--metric` | Accuracy metric to use | Input | string | No | IoU |
| `--batchSize` | batch size for training | Input | int | Yes for CPU | max for GPU|
| `--trainValSplit` | ratio to split data for train/validation | Input | float | No | 0.7 |
| `--outDir` | Output collection | Output | collection | Yes | -|

