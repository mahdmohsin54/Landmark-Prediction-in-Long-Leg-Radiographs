import pandas as pd
from sklearn.model_selection import train_test_split

# Load the txt file
txt_file_path = '/home/mahd/pre_train_EN_cropped.txt'

# Initialize an empty list to hold the data
data = []

# Read the txt file and parse each line
with open(txt_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        image_name = parts[0]
        num_labels = int(parts[1])
        labels = [int(coord) for coord in parts[2:]]
        row = [image_name, num_labels] + labels
        data.append(row)

# Determine the maximum number of labels
max_labels = max(row[1] for row in data)

# Create column names
columns = ['image_name', 'number_of_labels']
for i in range(max_labels):
    columns.extend([f'label_{i}_y', f'label_{i}_x'])

# Create the DataFrame
df = pd.DataFrame(data, columns=columns)

# Fill missing values with 0
df = df.fillna(0)

# Save the DataFrame to a CSV file for reference
df.to_csv('/home/mahd/Label-Augmentation-Folder/dataframes/dataset.csv', index=False)

print(df.head())

# Split the DataFrame
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
# test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the splits to CSV files
train_df.to_csv('/home/mahd/Label-Augmentation-Folder/dataframes/train_dataset.csv', index=False)
val_df.to_csv('/home/mahd/Label-Augmentation-Folder/dataframes/validation_dataset.csv', index=False)
# test_df.to_csv('test_dataset.csv', index=False)

print("Training DataFrame head:")
print(train_df.head())

print("Validation DataFrame head:")
print(val_df.head())

# print("Testing DataFrame head:")
# print(test_df.head())

