import os
import shutil
import cv2
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import json

def relocate(df, image_path_all, image_path, data_type):
    if not os.path.exists(f'{image_path}/{data_type}'):
        os.makedirs(f'{image_path}/{data_type}')

    for index, row in df.iterrows():
        image_name = row['image_name']
        image_path_src = os.path.join(image_path_all, image_name)
        if os.path.exists(image_path_src):
            shutil.copy(image_path_src, f'{image_path}/{data_type}/{image_name}')
        else:
            print(f"Image {image_name} does not exist in {image_path_all}")

def create_csv(df, csv_path):
    df.to_csv(csv_path, index=False)

def save_segments(df, image_path, image_segment_path, bbox_dict, data_type):
    if not os.path.exists(f'{image_segment_path}/{data_type}'): 
        os.makedirs(f'{image_segment_path}/{data_type}')
    
    postprocessed_data = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_name = row['image_name']
        if image_name not in bbox_dict:
            print(f"Warning: Bounding box for {image_name} not found in dictionary and will be skipped.")
            continue

        image_path_src = os.path.join(image_path, data_type, image_name)
        image_array = cv2.imread(image_path_src)

        if image_array is None:
            print(f"Warning: {image_name} could not be read and will be skipped.")
            continue

        img_dir = os.path.join(image_segment_path, data_type, os.path.splitext(image_name)[0])
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        bboxes = bbox_dict[image_name]['bboxes']
        landmarks = bbox_dict[image_name]['landmarks']
        new_row = [image_name, len(landmarks)]
        
        # Update coordinates to be relative to the bounding box
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            if i == 0:
                landmarks_indices = [0, 1]
            elif i == 1:
                landmarks_indices = [2]
            elif i == 2:
                landmarks_indices = [3, 4, 5, 6, 7, 8]
            elif i == 3:
                landmarks_indices = [9]
            elif i == 4:
                landmarks_indices = [10]

            for j in landmarks_indices:
                ly, lx = landmarks[j]
                # if x1 <= lx <= x2 and y1 <= ly <= y2:
                new_row.extend([ly - y1, lx - x1])
                # else:
                #     new_row.extend([0, 0])

            # Save the segment
            segment = image_array[y1:y2, x1:x2]
            segment_path = os.path.join(img_dir, f'segment_{i+1}.png')
            cv2.imwrite(segment_path, segment)

        postprocessed_data.append(new_row)

    columns = ['image_name', 'number_of_labels']
    for i in range(len(landmarks)):
        columns.extend([f'label_{i}_y', f'label_{i}_x'])

    postprocessed_df = pd.DataFrame(postprocessed_data, columns=columns)
    return postprocessed_df

def preprocess(image_path_all, image_path, image_segment_path, test_label_txt, test_csv, test_csv_preprocessed, bbox_dict_path):
    # Load the txt file and create a DataFrame
    data = []
    with open(test_label_txt, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            image_name = parts[0]
            num_labels = int(parts[1])
            labels = [float(coord) for coord in parts[2:]]
            row = [image_name, num_labels] + labels
            data.append(row)

    max_labels = max(row[1] for row in data)
    columns = ['image_name', 'number_of_labels']
    for i in range(max_labels):
        columns.extend([f'label_{i}_y', f'label_{i}_x'])

    df = pd.DataFrame(data, columns=columns)
    df = df.fillna(0)

    # Verify the existence of each image in the directory and remove those that do not exist
    df['exists'] = df['image_name'].apply(lambda img: os.path.exists(os.path.join(image_path_all, img)))
    df = df[df['exists']].drop(columns=['exists'])

    # Relocate images based on DataFrame
    relocate(df, image_path_all, image_path, 'test')

    # Create CSV files
    create_csv(df, test_csv)

    # Load bbox dictionary
    with open(bbox_dict_path, 'r') as f:
        bbox_dict = json.load(f)

    # Save segments from the original images and update coordinates
    test_postprocessed_df = save_segments(df, image_path, image_segment_path, bbox_dict, 'test')

    # Save postprocessed DataFrames to CSV
    test_postprocessed_df.to_csv(test_csv_preprocessed.replace("preprocessed", "postprocessed"), index=False)

# Define your paths and parameters
image_path_all = '/home/mahd/image/all'
image_path = '/home/mahd/Label-Augmentation-Folder/image'
image_padded_path = '/home/mahd/Label-Augmentation-Folder/image_segments'
test_label_txt = '/home/mahd/pre_test_EN_cropped.txt'
test_csv = '/home/mahd/Label-Augmentation-Folder/dataframes/test_dataset.csv'
test_csv_preprocessed = '/home/mahd/Label-Augmentation-Folder/dataframes/test_dataset_preprocessed.csv'

bbox_dict_path = '/home/mahd/Segmented_Images_test/image_bbox_landmarks_dict.json'

# Run the preprocess function
preprocess(image_path_all, image_path, image_padded_path, test_label_txt, test_csv, test_csv_preprocessed, bbox_dict_path)