import pandas as pd

csv_path = 'C:/Users/peter/OneDrive/Documents/GitHub/Dog_Classifier/dogs.csv'
df = pd.read_csv(csv_path)

# Check unique values in a specific column (e.g., 'labels')
unique_values = df['labels'].unique()

# Print the number of unique values
print(f"Number of unique strings in 'labels': {len(unique_values)}")
print(f"Unique strings: {unique_values}")