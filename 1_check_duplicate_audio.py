import pandas as pd

# Path to the file
file_path = '/scratch/txt'

# Read the file into a DataFrame
data = pd.read_csv(file_path, sep='|')

# Check for duplicates in the 'path' column
duplicate_files = data[data.duplicated('path', keep=False)]

if not duplicate_files.empty:
    print("Duplicate audio files found:")
    print(duplicate_files)
else:
    print("No duplicate audio files found.")
