import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def UNet(args, DEVICE):
    print("---------- Loading  Model ----------")

    model = smp.Unet(
        encoder_name     = 'resnet101', 
        encoder_weights  = 'imagenet', 
        encoder_depth    = args.encoder_depth,
        classes          = args.output_channel, 
        activation       = 'sigmoid',
        decoder_channels = args.decoder_channel,
    )
    print("---------- Model Loaded ----------")

    # Erase Sigmoid
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])

    # Train model in multiple GPUs
    # model = nn.DataParallel(model)
    
    return model.to(DEVICE)

def get_rcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model