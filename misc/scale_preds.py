import pandas as pd
import cv2
import os

def scale_coordinates(predictions_csv, test_images_dir, output_csv):
    # Load the predictions CSV
    df = pd.read_csv(predictions_csv)
    
    # Initialize a list to store the scaled predictions
    scaled_data = []
    
    for index, row in df.iterrows():
        image_name = row['image_name']
        num_labels = row['number_of_labels']
        
        # Read the original image to get its dimensions
        image_path = os.path.join(test_images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_name} does not exist in {test_images_dir}")
            continue
        
        original_image = cv2.imread(image_path)
        original_height, original_width = original_image.shape[:2]
        
        # Calculate padding
        if original_height > original_width:
            padding = (original_height - original_width) // 2
            padded_height = original_height
            padded_width = original_height  # After padding, width is equal to height
        else:
            padding = (original_width - original_height) // 2
            padded_height = original_width  # After padding, height is equal to width
            padded_width = original_width

        # Scale the coordinates
        scaled_row = [image_name, num_labels]
        for i in range(num_labels):
            y = row[f'label_{i}_y']
            x = row[f'label_{i}_x']
            
            # Adjust for padding
            if original_height > original_width:
                scaled_y = y * original_height / 512
                scaled_x = (x * original_height / 512) - padding
            else:
                scaled_y = (y * original_width / 512) - padding
                scaled_x = x * original_width / 512
            
            scaled_row.extend([scaled_y, scaled_x])
        
        # Add zero coordinates for labels that do not exist
        for i in range(num_labels, 11):
            scaled_row.extend([0, 0])
        
        scaled_data.append(scaled_row)
    
    # Define the columns for the new CSV file
    columns = ['image_name', 'number_of_labels']
    for i in range(11):
        columns.extend([f'label_{i}_y', f'label_{i}_x'])
    
    # Create a DataFrame and save it to a CSV file
    scaled_df = pd.DataFrame(scaled_data, columns=columns)
    scaled_df.to_csv(output_csv, index=False)

# Define your paths
predictions_csv = '/home/mahd/Label-Augmentation-Folder/dataframes/Pre_D75_L2_redo_test_prediction.csv'
test_images_dir = '/home/mahd/Label-Augmentation-Folder/image/test'
output_csv = '/home/mahd/Label-Augmentation-Folder/dataframes/new_output_scaled_predictions.csv'

# Run the function
scale_coordinates(predictions_csv, test_images_dir, output_csv)