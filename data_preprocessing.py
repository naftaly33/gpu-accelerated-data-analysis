import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
file_path = "C:\\Users\\m.niftali\\Documents\\ASPR_Treatments_Locator_20241225.csv"
data = pd.read_csv(file_path)

# Step 2: Handle missing data (drop rows with missing values)
data = data.dropna()

# Step 3: Explore the columns and check which are numerical for normalization
print("Columns in Dataset:", data.columns)

# Step 4: Select numerical columns for normalization (adjust to your dataset)
numerical_columns = ["Latitude", "Longitude", "Zip"]  # Replace with actual numerical columns
X = data[numerical_columns].values

# Step 5: Visualize the distribution of numerical columns before normalization
plt.figure(figsize=(10, 6))
sns.histplot(data[numerical_columns], kde=True)
plt.title('Distribution of Numerical Columns Before Normalization')
plt.show()

# Step 6: Normalize the numerical data using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Step 7: Visualize the distribution of numerical columns after normalization
plt.figure(figsize=(10, 6))
sns.histplot(X_normalized, kde=True)
plt.title('Distribution of Numerical Columns After Normalization')
plt.show()

# Step 8: Convert target column to binary classes
# Assuming 'Is COVID-19' is your target column (adjust if necessary)
data['Target'] = data['Is COVID-19'].apply(lambda x: 1 if x == True else 0)  # Adjust target column as needed
y = data['Target'].values

# Step 9: Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=data['Target'])
plt.title('Distribution of Target (Is COVID-19)')
plt.show()

# Step 10: Print out the results for inspection
print("Normalized Data:\n", X_normalized[:5])  # Print first 5 rows for inspection
print("Target Values:\n", y[:5])  # Print first 5 target values
