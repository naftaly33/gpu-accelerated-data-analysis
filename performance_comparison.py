import time
from sklearn.linear_model import LogisticRegression as CPU_LogisticRegression
from cuml.linear_model import LogisticRegression as GPU_LogisticRegression
import cupy as cp

# Train on CPU
start_cpu = time.time()
model_cpu = CPU_LogisticRegression()
model_cpu.fit(X_normalized, y)
end_cpu = time.time()

# Measure CPU training time
cpu_time = end_cpu - start_cpu
print(f"CPU Training Time: {cpu_time:.2f} seconds")

# Initialize a new GPU model for training on the GPU
model_gpu = GPU_LogisticRegression()

# Train on GPU
start_gpu = time.time()
model_gpu.fit(X_gpu, y_gpu)
end_gpu = time.time()

# Measure GPU training time
gpu_time = end_gpu - start_gpu
print(f"GPU Training Time: {gpu_time:.2f} seconds")

# Speedup
speedup = cpu_time / gpu_time
print(f"Speedup: {speedup:.2f}x")
