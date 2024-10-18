import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

label_names = {
    'label_0': 'femoral head',
    'label_1': 'intertrochanteric region',
    'label_2': 'femoral midshaft',
    'label_3': 'left lateral femoral condyle',
    'label_4': 'medial femoral condyle',
    'label_5': 'right lateral femoral condyle',
    'label_6': 'left lateral third of tibia',
    'label_7': 'medial third of tibia',
    'label_8': 'right lateral third of tibia',
    'label_9': 'tibial midshaft',
    'label_10': 'tibial plafond',
}

# Load the CSV files
baseline_predictions_file = '/home/mahd/Label-Augmentation-Folder/dataframes/new_output_scaled_predictions.csv'
proposed_predictions_file = '/home/mahd/results/testing/scaled_merged_testing_test_prediction.csv'
ground_truth_file = '/home/mahd/pre_test_EN_cropped.txt'

baseline_df = pd.read_csv(baseline_predictions_file)
proposed_df = pd.read_csv(proposed_predictions_file)
ground_truth_df = pd.read_csv(ground_truth_file)

def calculate_rmse(pred_df, gt_df):
    rmse_values = {f'label_{i}': [] for i in range(11)}

    for _, row in pred_df.iterrows():
        image_name = row['image_name']
        gt_row = gt_df[gt_df['image_name'] == image_name]

        if gt_row.empty:
            continue

        for i in range(11):
            label_y_pred = row[f'label_{i}_y']
            label_x_pred = row[f'label_{i}_x']
            
            label_y_gt = gt_row[f'label_{i}_y'].values[0]
            label_x_gt = gt_row[f'label_{i}_x'].values[0]
            
            rmse_y = np.sqrt(mse([label_y_gt], [label_y_pred]))
            rmse_x = np.sqrt(mse([label_x_gt], [label_x_pred]))
            
            # Compute the average RMSE for the label
            avg_rmse = np.mean([rmse_y, rmse_x])
            
            rmse_values[f'label_{i}'].append(avg_rmse)
    
    return rmse_values

baseline_rmse = calculate_rmse(baseline_df, ground_truth_df)
proposed_rmse = calculate_rmse(proposed_df, ground_truth_df)

# Prepare data for the box plot
data = []
colors = ['blue', 'red']  # Colors for baseline and proposed
positions = []
labels = []

for i in range(11):
    label = f'label_{i}'
    label_name = label_names[label]
    data.append(baseline_rmse[label])
    positions.append(i * 2 + 1)
    labels.append(f'{label_name} (Baseline)')
    
    data.append(proposed_rmse[label])
    positions.append(i * 2 + 2)
    labels.append(f'{label_name} (Proposed)')

# Create the box plot
plt.figure(figsize=(25, 18))
box = plt.boxplot(data, positions=positions, patch_artist=True, widths=0.8)

# Color the box plots and make them thicker
for i, box_element in enumerate(box['boxes']):
    box_element.set_facecolor(colors[i % 2])
    box_element.set_linewidth(2)
    
for element in ['whiskers', 'caps', 'medians']:
    plt.setp(box[element], linewidth=2, color='black')

# Create legend
plt.plot([], c='blue', label='Baseline')
plt.plot([], c='red', label='Proposed')
plt.legend(fontsize=35)

# Set x-ticks
xtick_labels = [f'{label_names[f"label_{i}"]}' for i in range(11)]
plt.xticks(ticks=[i * 2 + 1.5 for i in range(11)], labels=xtick_labels, fontsize=30, rotation=50)

# Set y-ticks with larger font size
plt.yticks(fontsize=30)

plt.title('RMSE Comparison: Baseline vs Proposed', fontsize=35)
plt.ylabel('RMSE (pixel distance)', fontsize=35)
plt.tight_layout()
plt.savefig('/home/mahd/Label-Augmentation-Folder/dataframes/box_plot.png')
plt.show()
