import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Simulated data for predicted vs actual movement (e.g., in 2D space)
actual_x = np.sin(np.linspace(0, 10, 100))
actual_y = np.cos(np.linspace(0, 10, 100))
predicted_x = actual_x + np.random.normal(0, 0.05, 100)  # Reduced noise
predicted_y = actual_y + np.random.normal(0, 0.05, 100)

# Calculate error
error_x = predicted_x - actual_x
error_y = predicted_y - actual_y
error = np.sqrt(error_x**2 + error_y**2)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(np.vstack((actual_x, actual_y)).T,
                                  np.vstack((predicted_x, predicted_y)).T))

# Convert RMSE to accuracy (inverse proportion of error)
accuracy = max(0, 1 - rmse / 0.5) * 100  # Normalized to a max error tolerance of 0.5

# Plot
plt.figure(figsize=(8, 6))
plt.plot(actual_x, actual_y, label='Actual Trajectory')
plt.plot(predicted_x, predicted_y, label='Predicted Trajectory', linestyle='dashed')
sc = plt.scatter(predicted_x, predicted_y, c=error, cmap='viridis', label='Error Magnitude', alpha=0.6)
plt.colorbar(sc, label='Error Magnitude')

plt.title(f'Predicted vs Actual Robotic Movement\nTrajectory Accuracy: {accuracy:.2f}%', fontsize=13)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.tight_layout()
plt.show()
