import pandas as pd

# Load the CSV file
input_file = '/home/mahd/results/testing/testing_test_prediction.csv'
output_file = '/home/mahd/results/testing/merged_testing_test_prediction.csv'

df = pd.read_csv(input_file)

# Group by image_name and concatenate the label columns row-wise
grouped = df.groupby('image_name').apply(lambda x: x.iloc[:, 2:].values.flatten()).reset_index()

# Reconstruct the DataFrame
merged_data = []
for _, row in grouped.iterrows():
    image_name = row['image_name']
    labels = row[0]
    # Filter out NaNs or empty values
    labels = labels[~pd.isna(labels)]
    # Create a row with the image name and concatenated labels
    merged_row = [image_name, len(labels)//2] + labels.tolist()
    merged_data.append(merged_row)

# Determine the maximum number of labels to create appropriate columns
max_labels = max(len(row) for row in merged_data) - 2
column_names = ['image_name', 'number_of_labels'] + [f'label_{i}_{coord}' for i in range(max_labels//2) for coord in ['y', 'x']]

# Create the final DataFrame
final_df = pd.DataFrame(merged_data, columns=column_names)

# Save the final DataFrame to a new CSV file
final_df.to_csv(output_file, index=False)

print(f"Merged CSV file saved as {output_file}")