import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utility.model import UNet
from dataset import CustomDataset, load_data
from utility.train_utils import set_parameters, rmse, geom_element, angle_element, create_directories
from utility.visualization import visualize
from utility.log import log_terminal
from PIL import Image
import wandb
import os

# Ensure segmented images directory exists
segmented_image_dir = "segmented_images"
if not os.path.exists(segmented_image_dir):
    os.makedirs(segmented_image_dir)

def save_segmented_images(images, rois_list, patient_names):
    """ Save cropped images (segmented images) in a folder for each patient based on predicted ROIs. """
    for i, (image, rois) in enumerate(zip(images, rois_list)):
        patient_dir = os.path.join(segmented_image_dir, patient_names[i])
        os.makedirs(patient_dir, exist_ok=True)  # Create a folder for each patient
        
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C) format
        for j, box in enumerate(rois):
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_image = image_np[y_min:y_max, x_min:x_max, :]
            cropped_pil_image = Image.fromarray((cropped_image * 255).astype(np.uint8))
            cropped_pil_image.save(os.path.join(patient_dir, f"segment_{j+1}.png"))  # Save as segment_1, segment_2, ...

def load_segmented_images(patient_names, rois_list, device):
    """ Load the segmented images back from the folder. """
    segmented_images = []
    for i, rois in enumerate(rois_list):
        cropped_images = []
        patient_dir = os.path.join(segmented_image_dir, patient_names[i])
        for j in range(len(rois)):
            image_path = os.path.join(patient_dir, f"segment_{j+1}.png")
            cropped_image = Image.open(image_path)
            cropped_image_tensor = torch.tensor(np.array(cropped_image)).float().permute(2, 0, 1) / 255.0
            cropped_images.append(cropped_image_tensor.to(device))
        segmented_images.append(cropped_images)
    return segmented_images

def reshape_masks(masks):
    batch_size, num_segments, num_channels, height, width = masks.size()
    return masks.view(batch_size, num_segments * num_channels, height, width)

def train_function(args, DEVICE, models, loss_fn_pixel, optimizers, train_loader, rcnn_model):
    total_loss = 0
    if isinstance(models, list):
        for model in models:
            model.train()
    else:
        models.train()

    for images, masks, segments_dir, image_name, label_list in tqdm(train_loader):
        images = images.to(DEVICE)
        masks = masks.float().to(DEVICE)
        images_list = [img.to(DEVICE) for img in images]

        rois_list = []

        for image in images_list:
            my_image = [image]
            print(image.shape)
            rois = rcnn_model(my_image)
            rois_list.append(rois)

        with open('/home/mahd/Label-Augmentation-Folder/rois.txt', 'w') as f:
            for i, rois in enumerate(rois_list):
                f.write(f"Image {i}:\n")
                for box in rois:
                    box_str = ','.join(map(str, box.tolist()))
                    f.write(f"{box_str}\n")

        # Save the segmented images to disk in patient-specific subfolders
        save_segmented_images(images, rois_list, image_name)

        segmented_images = load_segmented_images(image_name, rois_list, DEVICE)

        if isinstance(models, list):
            for i, model in enumerate(models):
                optimizers[i].zero_grad()
                # predictions = model(images[:, i])
                predictions = [model(seg_img) for seg_img in segmented_images[i]]
                loss = loss_fn_pixel(predictions, masks[:, i]) * args.pixel_loss_weight
                loss.backward()
                optimizers[i].step()
                total_loss += loss.item()
        else:
            optimizers.zero_grad()
            predictions = models(images)
            loss = loss_fn_pixel(predictions, masks) * args.pixel_loss_weight
            loss.backward()
            optimizers.step()
            total_loss += loss.item()

    return total_loss

def get_segment_labels(label_list, segment_idx):
    if segment_idx == 0:
        return label_list[:4], [0,1]  # labels 0 and 1
    elif segment_idx == 1:
        return label_list[4:6], [2]  # label 2
    elif segment_idx == 2:
        return label_list[6:18], [3,4,5,6,7,8]  # labels 3 to 8
    elif segment_idx == 3:
        return label_list[18:20], [9]  # label 9
    elif segment_idx == 4:
        return label_list[20:22], [10]  # label 10
    return []

def calculate_rmse(predicted_landmarks, ground_truth_landmarks):
    # distances = []
    # for (pred_y, pred_x), (gt_y, gt_x) in zip(predicted_landmarks, ground_truth_landmarks):
    # print(f'Me: {ground_truth_landmarks}')
    pred_y, pred_x = predicted_landmarks[0]
    gt_y, gt_x = ground_truth_landmarks[0], ground_truth_landmarks[1]

    distance = (pred_y - gt_y) ** 2 + (pred_x - gt_x) ** 2
        # distances.append(distance)
    return np.sqrt(np.mean(distance))

def validate_function(args, DEVICE, models, epoch, val_loader):
    print("===== Starting Validation =====")
    if isinstance(models, list):
        for model in models:
            model.eval()
    else:
        models.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    rmse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]
    # angle_list = [[0]*len(val_loader) for _ in range(len(args.label_for_angle))]

    with torch.no_grad():
        for idx, (images, masks, segments_dir, image_name, label_list) in enumerate(tqdm(val_loader)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            image_path = segments_dir[0]
            image_name = image_name[0].split('.')[0]
            
            if isinstance(models, list):
                predictions = [model(images[:, i]) for i, model in enumerate(models)]
            else:
                predictions = models(images)
            
            # Validate angle difference
            # if args.label_for_angle != []:
            #     predict_angle, label_angle = angle_element(args, predictions, label_list, DEVICE)
            #     for i in range(len(args.label_for_angle)):
            #         angle_list[i][idx] = abs(label_angle[i] - predict_angle[i])

            # Get rmse difference for each prediction
            for segment_idx in range(5):
                segment_labels, indices = get_segment_labels(label_list, segment_idx)
                num_labels = len(indices)

                # if segment_idx == 1:
                #     print(f'Pred: {predictions[segment_idx]}')
                rmse_list, index_list = rmse(args, predictions[segment_idx], label_list, num_labels, idx, rmse_list, segment_idx, indices)
                extracted_pixels_list.append(index_list)

                # Make predictions to be 0 or 1
                prediction_binary = (predictions[segment_idx] > 0.5).float()
                masks_segment = masks[:, segment_idx]
                print(f"masks_segment:{masks_segment.shape}, {prediction_binary.shape}")
                dice_score += (2 * (prediction_binary * masks_segment).sum()) / ((prediction_binary + masks_segment).sum() + 1e-8)

                segment_path = segments_dir[0] + f"/segment_{segment_idx + 1}.png"

                if idx in [0, 5, 20, 30, 50]: 
                    # print(segments_dir[segment_idx], image_name, label_list, index_list, segment_path )
                    visualize(
                        args, idx, segment_path, image_name, segment_labels, indices, epoch, 
                        index_list, predictions[segment_idx], prediction_binary,
                        None, None, None, 'train', segment_idx
                    )

    dice = dice_score / (len(val_loader) * 5)

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
    print(f"Dice score: {dice}")
    print(f"Average Pixel to Pixel Distance: {total_rmse_mean}")

    # if args.label_for_angle != []:
    #     # Add up angle values
    #     angle_value = []
    #     for i in range(len(args.label_for_angle)):
    #         angle_value.append(sum(angle_list[i]))
    #     angle_value.append(sum(map(sum, angle_list)))

    #     return dice, total_rmse_mean, rmse_list, rmse_mean_by_label, angle_value
    # else:
    return dice, total_rmse_mean, rmse_list, rmse_mean_by_label, 0

def train(args, models, DEVICE, rcnn_model):
    best_loss, best_rmse_mean, best_angle_diff = np.inf, np.inf, np.inf
    if isinstance(models, list):
        optimizers = [optim.Adam(model.parameters(), lr=args.lr) for model in models]
    else:
        optimizers = optim.Adam(models.parameters(), lr=args.lr)

    loss_fn_pixel = nn.BCEWithLogitsLoss()

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
    wandb.watch(models, log="all")

    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch + 1}")

        if epoch % args.dilation_epoch == 0:
            args, loss_fn_pixel, train_loader, val_loader = set_parameters(args, models, epoch, DEVICE)

        train_loss = train_function(args, DEVICE, models, loss_fn_pixel, optimizers, train_loader, rcnn_model)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Train Loss: {avg_train_loss}")

        dice_score, rmse, rmse_list, rmse_mean_by_label, angle_value = validate_function(args, DEVICE, models, epoch, val_loader)
        print(f"Dice Score: {dice_score}, RMSE: {rmse}")

        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_dice_score": dice_score,
            "val_rmse_mean": rmse,
            # Add other metrics as needed
        })

        for i, rmse_val in enumerate(rmse_mean_by_label):
            wandb.log({f"val_rmse_label_{i}": rmse_val})

        if avg_train_loss < best_loss:
            print("===== New best model based on loss =====")
            best_loss = avg_train_loss

        if rmse < best_rmse_mean:
            print("===== New best model based on RMSE =====")
            if isinstance(models, list):
                for i, model in enumerate(models):
                    checkpoint = {"state_dict": model.state_dict(), "optimizer":  optimizers[i].state_dict()}
                    torch.save(checkpoint, f'./results/{args.wandb_name}/best_rmse_model_{i}.pth')
            else:
                torch.save(models.state_dict(), f'./results/{args.wandb_name}/best_rmse_model.pth')
            best_rmse_mean = rmse
            best_rmse_list = rmse_list
    
    log_terminal(args, 'best_rmse', best_rmse_list)
            
    print("Training complete")
    wandb.finish()
