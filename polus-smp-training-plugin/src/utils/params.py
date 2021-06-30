import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.losses as losses

models_dict = {'unet': smp.Unet,
          'unetpp': smp.UnetPlusPlus,
          'Linknet': smp.Linknet,
          'FPN': smp.FPN,
          'PSPNet': smp.PSPNet,
          'PAN': smp.PAN,
          'DeepLabV3': smp.DeepLabV3,
          'DeepLabV3Plus': smp.DeepLabV3Plus}


metric_dict = {'IoU': smp.utils.metrics.IoU,
               'Fscore': smp.utils.metrics.Fscore,
               'Precision': smp.utils.metrics.Precision,
               'Accuracy': smp.utils.metrics.Accuracy,
               'Recall': smp.utils.metrics.Recall}


loss_dict = {'Dice': losses.DiceLoss,
        'Jaccard': losses.JaccardLoss,
        'MSE': losses.MSELoss}

# add more dictionaries in the future for different inputs

