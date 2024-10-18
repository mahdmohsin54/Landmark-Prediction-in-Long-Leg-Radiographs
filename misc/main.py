import argparse
import torch
import os
from utility.model import UNet, get_rcnn_model
from utility.preprocess import preprocess
from train import train
from test import test

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Torch is running on {DEVICE}')

    rcnn_model = get_rcnn_model(num_classes=2)  # Adjust num_classes as per your dataset
    rcnn_model.load_state_dict(torch.load(args.rcnn_model_path))
    rcnn_model.to(DEVICE)
    rcnn_model.eval()

    # if args.model_type == 'single':
        # model = UNet(args.input_channel, args.output_channel, args.encoder_depth, args.decoder_channel).to(DEVICE)
        # if not args.test:
        # train(args, model, DEVICE)
        # if args.test:
        #     if os.path.exists(f'./results/{args.wandb_name}/best.pth'):
        #         model.load_state_dict(torch.load(f'./results/{args.wandb_name}/best.pth')['state_dict'])
        #     test(args, model, DEVICE)
    # elif args.model_type == 'multiple':
    models = [UNet(args, DEVICE) for _ in range(5)]
    if not args.test:
        train(args, models, DEVICE, rcnn_model)
    if args.test:
        for i in range(5):
            if os.path.exists(f'/home/mahd/results/vis_best_model_run/best_rmse_model_{i}.pth'):
                models[i].load_state_dict(torch.load(f'/home/mahd/results/vis_best_model_run/best_rmse_model_{i}.pth')['state_dict'])
        test(args, DEVICE, models)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## boolean arguments
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_reweight', action='store_false')

    parser.add_argument('--image_segment_path', type=str, default="//home/mahd/Label-Augmentation-Folder/image_segments", help='path to image segments')
    parser.add_argument('--train_label_txt', type=str, default="/path/to/pre_train_cropped.txt", help='train label text file path')
    parser.add_argument('--train_csv', type=str, default="/path/to/save/csv/files/train_dataset.csv", help='train csv file path')
    parser.add_argument('--train_csv_postprocessed', type=str, default="/home/mahd/Label-Augmentation-Folder/dataframes/train_dataset_postprocessed.csv", help='train csv file path')
    parser.add_argument('--test_csv_postprocessed', type=str, default="/home/mahd/Label-Augmentation-Folder/dataframes/test_dataset_postprocessed.csv", help='train csv file path')
    parser.add_argument('--val_csv_postprocessed', type=str, default="/home/mahd/Label-Augmentation-Folder/dataframes/validation_dataset_postprocessed.csv", help='validation csv file path')
    parser.add_argument('--result_directory', type=str, default="/home/mahd/Label-Augmentation-Folder/results", help='directory to save results')
    parser.add_argument('--wandb_name', type=str, default="experiment", help='name for Weights and Biases run')
    parser.add_argument('--wandb_project', type=str, default="my_new_project", help='WandB project name')

    ## hyperparameters
    parser.add_argument('--seed', type=int, default=2022, help='seed customization for result reproduction')
    parser.add_argument('--input_channel', type=int, default=3, help='input channel size for UNet')
    parser.add_argument('--output_channel', type=int, default=11, help='output channel size for UNet')
    parser.add_argument('--encoder_depth', type=int, default=5, help='model depth for UNet')
    parser.add_argument('--decoder_channel', type=int, nargs='+', default=[256, 128, 64, 32, 16], help='model decoder channels')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--image_resize', type=int, default=512, help='image resize value')
    parser.add_argument('--dilate', type=int, default=75, help='initial dilate value')
    parser.add_argument('--dilation_decrease', type=int, default=10, help='dilation decrease value')
    parser.add_argument('--dilation_epoch', type=int, default=50, help='epochs to decrease dilation')
    parser.add_argument('--pixel_loss_weight', type=int, default=1, help='pixel loss weight')

    parser.add_argument('--rcnn_model_path', type=str, default='faster_rcnn_model.pth', help='state dictionary for FRCNN model')
    # parser.add_argument('--model_type', type=str, default='multiple', choices=['single', 'multiple'], help='type of model to use')

    args = parser.parse_args()
    main(args)