import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

def create_mask(image_size, labels):
    mask = np.zeros(image_size, dtype=np.uint8)
    for y, x in labels:
        if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
            mask[int(y), int(x)] = 1
    return mask

def dilate_mask(mask, iterations=55):
    return binary_dilation(mask, iterations=iterations).astype(mask.dtype)

def overlay_mask(image, mask, labels):
    image_array = np.array(image.convert('RGB'))
    overlay = np.zeros_like(image_array)
    overlay[:, :, 0] = mask * 255
    overlay[:, :, 1] = mask * 255
    overlay[:, :, 2] = mask * 255
    combined = np.maximum(image_array, overlay)
    
    # Convert combined array back to an image
    combined_image = Image.fromarray(combined)
    
    # # Draw the landmarks on the combined image
    # draw = ImageDraw.Draw(combined_image)
    # for y, x in labels:
    #     if 0 <= x < combined_image.size[0] and 0 <= y < combined_image.size[1]:
    #         radius = 8
    #         draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue')
    
    return combined_image

def process_segment(segment_path, labels, bbox, segment_idx, output_dir, iterations):
    try:
        segment = Image.open(segment_path)
        adjusted_labels = [(y - bbox[1], x - bbox[0]) for y, x in labels]
        mask = create_mask(segment.size[::-1], adjusted_labels)
        dilated_mask = dilate_mask(mask, iterations)
        combined_image = overlay_mask(segment, dilated_mask, adjusted_labels)
        output_path = os.path.join(output_dir, f'segment_{segment_idx}_overlay.png')
        combined_image.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error processing segment {segment_idx}: {e}")

def main(dataset_csv, patient_image_name, segments_dir, output_dir, bbox_landmark_json, iterations=15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = pd.read_csv(dataset_csv)
    
    # Filter the dataset for the specific patient
    patient_sample = dataset[dataset['image_name'] == patient_image_name]
    
    if patient_sample.empty:
        print(f"No data found for patient image name: {patient_image_name}")
        return
    
    # Load the bounding box and landmark dictionary
    with open(bbox_landmark_json, 'r') as f:
        image_bbox_landmark = json.load(f)
    
    if patient_image_name not in image_bbox_landmark:
        print(f"No bounding box and landmark data found for patient image name: {patient_image_name}")
        return
    
    bboxes = image_bbox_landmark[patient_image_name]['bboxes']
    landmarks = image_bbox_landmark[patient_image_name]['landmarks']
    
    for _, row in patient_sample.iterrows():
        image_name = row['image_name']
        labels = [(row[f'label_{i}_y'], row[f'label_{i}_x']) for i in range(11)]
        
        segments_labels = {
            0: labels[0:2],
            1: labels[2:3],
            2: labels[3:9],
            3: labels[9:10],
            4: labels[10:11]
        }
        
        for segment_idx, segment_labels in segments_labels.items():
            segment_path = os.path.join(segments_dir, f'{image_name}', f'Segment {segment_idx}.png')
            if os.path.exists(segment_path):
                print(f"Processing segment: {segment_path}")
                process_segment(segment_path, segment_labels, bboxes[segment_idx], segment_idx, output_dir, iterations)
            else:
                print(f"Segment not found: {segment_path}")

if __name__ == '__main__':
    dataset_csv = '/home/mahd/pre_test_EN_cropped.txt'  # CSV file containing the entire dataset
    patient_image_name = 'Reith_Guenter_Reith_Guenter_Seq1_Ser4_Img1_dcm.png'  # The image name of the specific patient
    segments_dir = '/home/mahd/Segmented_Images_test'  # Directory containing the segments of the images
    output_dir = '/home/mahd/Segmented_Patient'  # Directory where the overlaid images will be saved
    bbox_landmark_json = '/home/mahd/Segmented_Images_test/image_bbox_landmarks_dict.json'  # Path to the bounding box and landmark JSON file
    
    main(dataset_csv, patient_image_name, segments_dir, output_dir, bbox_landmark_json)