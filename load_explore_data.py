import pandas as pd

# Load the dataset
file_path = r"C:\\Users\\m.niftali\\Documents\\ASPR_Treatments_Locator_20241225.csv"
data = pd.read_csv(file_path)

# Explore the dataset with better visual output

# Dataset Shape
print("\n" + "="*50)
print(f"Dataset Shape: {data.shape}")
print("="*50)

# Columns in Dataset
print("Columns in Dataset:")
for column in data.columns:
    print(f"- {column}")
print("="*50)

# First 5 Rows
print("First 5 Rows:")
print(data.head())
print("="*50)

# Missing Values
print("Missing Values per Column:")
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]  # Filter columns with missing values
if not missing_values.empty:
    for column, missing in missing_values.items():
        print(f"{column}: {missing} missing values")
else:
    print("No missing values found!")
print("="*50)
