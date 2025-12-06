import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

np.random.seed(123)

T = 100  # time steps

# Simulated data (same as before) ...

true_lidar_dist = np.random.uniform(0.2, 10, size=T)
true_lidar_labels = (true_lidar_dist < 2).astype(int)
lidar_preds = []
for d in true_lidar_dist:
    if d < 2:
        pred = np.random.choice([1, 0], p=[0.9, 0.1])
    else:
        pred = np.random.choice([0, 1], p=[0.9, 0.1])
    lidar_preds.append(pred)
lidar_preds = np.array(lidar_preds)

true_heat = np.random.normal(30, 15, size=T)
true_heat = np.clip(true_heat, 0, 100)
true_heat_labels = (true_heat > 50).astype(int)
heat_preds = []
for h in true_heat:
    if h > 50:
        pred = np.random.choice([1, 0], p=[0.9, 0.1])
    else:
        pred = np.random.choice([0, 1], p=[0.9, 0.1])
    heat_preds.append(pred)
heat_preds = np.array(heat_preds)

gas_concentration = np.random.normal(50, 10, size=T)
gas_concentration[:5] = 300 + 20 * np.random.rand(5)
iso_forest = IsolationForest(contamination=0.05, random_state=123)
gas_preds = iso_forest.fit_predict(gas_concentration.reshape(-1, 1))
gas_preds = (gas_preds == -1).astype(int)
true_gas_labels = np.zeros(T, dtype=int)
true_gas_labels[:5] = 1

true_audio_labels = np.random.choice([0, 1], size=T, p=[0.85, 0.15])
audio_preds = []
for lbl in true_audio_labels:
    if lbl == 1:
        pred = np.random.choice([1, 0], p=[0.9, 0.1])
    else:
        pred = np.random.choice([0, 1], p=[0.9, 0.1])
    audio_preds.append(pred)
audio_preds = np.array(audio_preds)

# Plotting with reduced vertical size and smaller fonts

fig, axs = plt.subplots(4, 1, figsize=(14, 8))  # reduced height from 12 to 8

# Plot 1: LIDAR
axs[0].plot(range(T), true_lidar_dist, label='Obstacle Distance (m)', color='blue', linewidth=1)
axs[0].scatter(range(T), true_lidar_labels * 1.5, label='True Hazard', color='green', s=10)
axs[0].scatter(range(T), lidar_preds * 1.8, label='Predicted Hazard', color='red', marker='x', s=20)
axs[0].set_title('LIDAR / Depth Camera: Obstacle Detection', fontsize=10)
axs[0].set_ylabel('Distance / Hazard', fontsize=8)
axs[0].legend(fontsize=7)
axs[0].tick_params(axis='both', labelsize=7)
lidar_acc = accuracy_score(true_lidar_labels, lidar_preds)
axs[0].text(
    0.05,
    0.9,
    f'Accuracy: {lidar_acc * 100:.1f}%',
    transform=axs[0].transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

# Plot 2: Infrared / Thermal
axs[1].plot(range(T), true_heat, label='Heat Sensor (°C)', color='orange', linewidth=1)
axs[1].scatter(range(T), true_heat_labels * 110, label='True Hazard', color='green', s=10)
axs[1].scatter(range(T), heat_preds * 115, label='Predicted Hazard', color='red', marker='x', s=20)
axs[1].set_title('Infrared / Thermal: Heat Hazard Detection', fontsize=10)
axs[1].set_ylabel('Temperature / Hazard', fontsize=8)
axs[1].legend(fontsize=7)
axs[1].tick_params(axis='both', labelsize=7)
heat_acc = accuracy_score(true_heat_labels, heat_preds)
axs[1].text(
    0.05,
    0.9,
    f'Accuracy: {heat_acc * 100:.1f}%',
    transform=axs[1].transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

# Plot 3: Gas Sensor
axs[2].plot(range(T), gas_concentration, label='Gas Concentration (ppm)', color='purple', linewidth=1)
axs[2].scatter(range(T), true_gas_labels * 600, label='True Anomaly', color='green', s=10)
axs[2].scatter(range(T), gas_preds * 650, label='Predicted Anomaly', color='red', marker='x', s=20)
axs[2].set_title('Gas/Smoke Sensor: Anomaly Detection (Isolation Forest)', fontsize=10)
axs[2].set_ylabel('Concentration / Anomaly', fontsize=8)
axs[2].legend(fontsize=7)
axs[2].tick_params(axis='both', labelsize=7)
gas_acc = accuracy_score(true_gas_labels, gas_preds)
axs[2].text(
    0.05,
    0.9,
    f'Accuracy: {gas_acc * 100:.1f}%',
    transform=axs[2].transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

# Plot 4: Environmental Audio
axs[3].scatter(range(T), true_audio_labels, label='True Distress', color='green', s=10)
axs[3].scatter(range(T), audio_preds, label='Predicted Distress', color='red', marker='x', s=20)
axs[3].set_title('Environmental Audio: Distress Sound Detection', fontsize=10)
axs[3].set_ylabel('Distress (1) / Normal (0)', fontsize=8)
axs[3].set_xlabel('Time Step', fontsize=8)
axs[3].legend(fontsize=7)
axs[3].tick_params(axis='both', labelsize=7)
audio_acc_env = accuracy_score(true_audio_labels, audio_preds)
axs[3].text(
    0.05,
    0.9,
    f'Accuracy: {audio_acc_env * 100:.1f}%',
    transform=axs[3].transAxes,
    fontsize=8,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

plt.tight_layout(h_pad=1.0)  # add some vertical padding
plt.show()

# Print accuracy scores in console
print(f"LIDAR Obstacle Detection Accuracy: {lidar_acc * 100:.2f}%")
print(f"Infrared Heat Hazard Accuracy: {heat_acc * 100:.2f}%")
print(f"Gas Sensor Anomaly Detection Accuracy: {gas_acc * 100:.2f}%")
print(f"Environmental Audio Distress Detection Accuracy: {audio_acc_env * 100:.2f}%")
