import matplotlib.pyplot as plt

# Bar plot for performance comparison
times = [cpu_time, gpu_time]
labels = ['CPU', 'GPU']

# Create a bar plot
plt.figure(figsize=(8, 5))  # Set the figure size
plt.bar(labels, times, color=['blue', 'green'], width=0.5)

# Add labels and title
plt.title('Training Time Comparison (CPU vs GPU)', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xlabel('Platform', fontsize=12)

# Show the exact value on top of each bar
for i, time in enumerate(times):
    plt.text(i, time + 0.1, f'{time:.2f}s', ha='center', fontsize=12)

# Adding grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
