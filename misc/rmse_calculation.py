import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def calculate_rmse(file1, file2):
    # Define the column names
    column_names = ['image_name', 'number_of_labels'] + [f'label_{i}_{coord}' for i in range(11) for coord in ['y', 'x']]
    
    # Read the CSV files into DataFrames with headers
    df1 = pd.read_csv(file1, header=0, names=column_names)
    df2 = pd.read_csv(file2, header=0, names=column_names)
    
    # Merge the DataFrames on image_name to keep only matching images
    df_merged = pd.merge(df1, df2, on='image_name', suffixes=('_1', '_2'))
    
    # List to store RMSE for each label
    rmses = []

    # Calculate MSE for each label for each row, and then average the MSEs and take the square root to get the RMSE
    for i in range(11):
        label_y1 = f'label_{i}_y_1'
        label_x1 = f'label_{i}_x_1'
        
        label_y2 = f'label_{i}_y_2'
        label_x2 = f'label_{i}_x_2'
        
        # Combine y and x into coordinate pairs for MSE calculation
        coords_1 = df_merged[[label_y1, label_x1]].to_numpy()
        coords_2 = df_merged[[label_y2, label_x2]].to_numpy()
        
        # Calculate MSE for each row
        mse_values = np.mean((coords_1 - coords_2) ** 2, axis=1)
        
        # Compute the RMSE by taking the square root of the average MSE
        rmse = np.sqrt(np.mean(mse_values))
        
        rmses.append((f'label_{i}', rmse))
    
    return rmses

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse

def calculate_std(file1, file2):
    # Define the column names
    column_names = ['image_name', 'number_of_labels'] + [f'label_{i}_{coord}' for i in range(11) for coord in ['y', 'x']]
    
    # Read the CSV files into DataFrames with headers
    df1 = pd.read_csv(file1, header=0, names=column_names)
    df2 = pd.read_csv(file2, header=0, names=column_names)
    
    # Merge the DataFrames on image_name to keep only matching images
    df_merged = pd.merge(df1, df2, on='image_name', suffixes=('_1', '_2'))
    
    # List to store standard deviation for each label
    stds = []

    # Calculate standard deviation of RMSE for each label for each row
    for i in range(11):
        label_y1 = f'label_{i}_y_1'
        label_x1 = f'label_{i}_x_1'
        
        label_y2 = f'label_{i}_y_2'
        label_x2 = f'label_{i}_x_2'
        
        rmse_y_values = []
        rmse_x_values = []
        
        for _, row in df_merged.iterrows():
            label_y_gt = row[label_y2]
            label_x_gt = row[label_x2]
            label_y_pred = row[label_y1]
            label_x_pred = row[label_x1]
            
            # Calculate RMSE for y and x values separately
            rmse_y = np.sqrt(mse([label_y_gt], [label_y_pred]))
            rmse_x = np.sqrt(mse([label_x_gt], [label_x_pred]))
            
            rmse_y_values.append(rmse_y)
            rmse_x_values.append(rmse_x)
        
        # Compute the standard deviation of RMSE for y and x
        std_rmse_y = np.std(rmse_y_values)
        std_rmse_x = np.std(rmse_x_values)
        
        # Compute the average of these standard deviations
        avg_std_rmse = np.mean([std_rmse_y, std_rmse_x])
        
        stds.append((f'label_{i}', avg_std_rmse))
    
    return stds

# Paths to the input files
file1 = '/home/mahd/Label-Augmentation-Folder/dataframes/new_output_scaled_predictions.csv'
file2 = '/home/mahd/pre_test_EN_cropped.txt'

# Calculate and print the standard deviations of RMSE
stds = calculate_std(file1, file2)
for label, std in stds:
    print(f'{label}: {std:.1f}')


def save_results(rmse_values, std_values, experiment_name):
    # Create a DataFrame with the RMSE and standard deviation results
    results = {
        'label': [rmse[0] for rmse in rmse_values],
        f'{experiment_name}_average_rmse': [rmse[1] for rmse in rmse_values],
        f'{experiment_name}_std_rmse': [std[1] for std in std_values]
    }
    return pd.DataFrame(results)

# Paths to the input and output files
baseline_predictions_file = '/home/mahd/Label-Augmentation-Folder/dataframes/new_output_scaled_predictions.csv'
proposed_predictions_file = '/home/mahd/results/testing/scaled_merged_testing_test_prediction.csv'
ground_truth_file = '/home/mahd/pre_test_EN_cropped.txt'
output_file = '/home/mahd/Label-Augmentation-Folder/dataframes/rmse_std_results_combined.csv'

# Calculate RMSE and standard deviation for baseline
baseline_rmse_values = calculate_rmse(baseline_predictions_file, ground_truth_file)
baseline_std_values = calculate_std(baseline_predictions_file, ground_truth_file)
baseline_results = save_results(baseline_rmse_values, baseline_std_values, 'baseline')

# Calculate RMSE and standard deviation for proposed
proposed_rmse_values = calculate_rmse(proposed_predictions_file, ground_truth_file)
proposed_std_values = calculate_std(proposed_predictions_file, ground_truth_file)
proposed_results = save_results(proposed_rmse_values, proposed_std_values, 'proposed')

# Combine the results
combined_results = pd.merge(baseline_results, proposed_results, on='label')

# Save the combined results to a CSV file
combined_results.to_csv(output_file, index=False)
print(f"Combined results saved to {output_file}")
