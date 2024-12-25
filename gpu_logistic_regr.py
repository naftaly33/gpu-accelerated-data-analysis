import cupy as cp
from cuml.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Now we are converting the normalized data (X_normalized) and target (y) to CuPy arrays for GPU processing
X_gpu = cp.array(X_normalized)  # Converting normalized features to CuPy array for GPU
y_gpu = cp.array(y)  # Convertin target labels to CuPy array for GPU

# Now we are initializing the cuML Logistic Regression model
model_gpu = LogisticRegression()

# Now we are training the logistic regression model on the GPU
model_gpu.fit(X_gpu, y_gpu)

# Now we are using the trained model to make predictions
y_pred_gpu = model_gpu.predict(X_gpu)

# Now we are converting CuPy arrays back to NumPy arrays for evaluation with sklearn's accuracy_score
y_gpu_np = cp.asnumpy(y_gpu)  # Convert true labels from GPU to CPU
y_pred_gpu_np = cp.asnumpy(y_pred_gpu)  # Convert predicted labels from GPU to CPU

# Now we are calculating and printing the accuracy of the model
accuracy = accuracy_score(y_gpu_np, y_pred_gpu_np)
print(f"GPU Logistic Regression Accuracy: {accuracy * 100:.2f}%")
