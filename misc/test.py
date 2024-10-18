import torch 
import math
import time
import os 
import csv
import numpy as np

from dataset import load_data
from tqdm import tqdm
from utility.log import log_terminal
from utility.train_utils import rmse, geom_element, angle_element
from utility.visualization import visualize
from train import get_segment_labels


def test(args, DEVICE, models, epoch=0):
    print("=====Starting Testing Process=====")
    if not os.path.exists(f'{args.result_directory}/{args.wandb_name}/test'):
        os.mkdir(f'{args.result_directory}/{args.wandb_name}/test')
    _, _, test_loader = load_data(args)

    if isinstance(models, list):
        for model in models:
            model.eval()
    else:
        models.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    extracted_pixels_to_df = []
    rmse_list = [[0]*len(test_loader) for _ in range(args.output_channel)]
    label_total = []

    extracted_pixels_list = []
    with torch.no_grad():
        start = time.time()
        for idx, (images, masks, segments_dir, image_name, label_list) in enumerate(tqdm(test_loader)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            image_path = segments_dir[0]
            image_name = image_name[0].split('.')[0]

            if isinstance(models, list):
                predictions = [model(images[:, i]) for i, model in enumerate(models)]
            else:
                predictions = models(images)
            
            for segment_idx in range(5):
                segment_labels, indices = get_segment_labels(label_list, segment_idx)
                num_labels = len(indices)

                rmse_list, index_list = rmse(args, predictions[segment_idx], label_list, num_labels, idx, rmse_list, segment_idx, indices)
                extracted_pixels_list.append(index_list)

                # Make predictions to be 0 or 1
                prediction_binary = (predictions[segment_idx] > 0.5).float()
                masks_segment = masks[:, segment_idx]
                label_total.append(np.ndarray.tolist(np.array(torch.Tensor(index_list), dtype=object).reshape(num_labels*2,1)))

                segment_path = segments_dir[0] + f"/segment_{segment_idx + 1}.png"

                visualize(
                        args, idx, segment_path, image_name, segment_labels, indices, epoch, 
                        index_list, predictions[segment_idx], prediction_binary,
                        None, None, None, 'test', segment_idx
                    )

                extracted_pixels_array = np.array(index_list[0]).reshape(-1)
                tmp_list = [f'{image_name}.png', len(index_list[0])]
                for i in range(len(extracted_pixels_array)):
                    tmp_list.append(extracted_pixels_array[i])
                extracted_pixels_to_df.append(tmp_list)
            
        end = time.time()

    print("=====Testing Process Done=====")
    # Removing RMSE for annotations that do not exist in the label
    rmse_mean_by_label = []
    for i in range(len(rmse_list)):
        tmp_sum, count = 0, 0
        for j in range(len(rmse_list[i])):
            if rmse_list[i][j] != -1:
               tmp_sum += rmse_list[i][j]
               count += 1
        rmse_mean_by_label.append(tmp_sum / count if count > 0 else 0)

    total_rmse_mean = sum(rmse_mean_by_label) / len(rmse_mean_by_label)
    print(f"Average Pixel to Pixel Distance: {total_rmse_mean}\n")

    for i, calc in enumerate(rmse_mean_by_label):
        print(f"RMSE for label {i}: {calc}")

    print(f"{end - start:.5f} seconds for {len(test_loader)} images")

    log_terminal(args, "test_prediction", extracted_pixels_list)
    log_terminal(args, "test_label", label_total)

    row_name = ["image_name", "number_of_labels"]
    for i in range(args.output_channel):
        row_name.append(f'label_{i}_y')
        row_name.append(f'label_{i}_x')
    csv_path = f'results/{args.wandb_name}/{args.wandb_name}_test_prediction.csv'
    with open(csv_path, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(row_name)
        write.writerows(extracted_pixels_to_df)