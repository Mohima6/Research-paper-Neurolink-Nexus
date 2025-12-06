import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score

# Simulate data length
T = 100  # time steps

# 1. Visual modality: object detection confidence scores (0 to 1)
np.random.seed(42)
true_visual_labels = np.random.choice([0, 1], size=T, p=[0.3, 0.7])  # 1 = object present
# simulate prediction with ~90% accuracy
visual_preds = []
for label in true_visual_labels:
    if label == 1:
        pred = np.random.choice([1, 0], p=[0.9, 0.1])
    else:
        pred = np.random.choice([0, 1], p=[0.9, 0.1])
    visual_preds.append(pred)
visual_preds = np.array(visual_preds)
visual_confidences = visual_preds * 0.9 + (1 - visual_preds) * 0.1  # confidence high for predicted positive

# 2. Auditory modality: simulate audio tone classification probabilities (3 classes)
classes = ['normal', 'warning', 'emergency']
true_audio_labels = np.random.choice([0, 1, 2], size=T, p=[0.6, 0.3, 0.1])


def simulate_audio_probs(true_label, acc=0.9):
    probs = np.ones(len(classes)) * (1 - acc) / (len(classes) - 1)
    probs[true_label] = acc
    return probs


audio_probs = np.array([simulate_audio_probs(lbl) for lbl in true_audio_labels])
audio_preds = np.argmax(audio_probs, axis=1)

# 3. Haptic modality: map sensor values (0-1) to vibration intensity (0-100%)
true_haptic_values = np.clip(np.random.normal(loc=0.5, scale=0.15, size=T), 0, 1)
# simulate mapping with some noise (simulate 90% "accuracy" as R^2)
noise = np.random.normal(0, 0.1, size=T)
pred_haptic_values = np.clip(true_haptic_values + noise, 0, 1)
pred_haptic_intensity = pred_haptic_values * 100  # %

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Visual plot
axs[0].plot(range(T), visual_confidences, label='Predicted Confidence', color='blue')
axs[0].scatter(range(T), true_visual_labels, color='green', s=15, label='True Label')
axs[0].set_title('Visual Modality: Object Detection Confidence', fontsize=12)
axs[0].set_ylabel('Confidence', fontsize=10)
axs[0].legend(fontsize=9)
axs[0].set_ylim([-0.1, 1.1])
axs[0].tick_params(axis='both', which='major', labelsize=8)
visual_acc = accuracy_score(true_visual_labels, visual_preds)
axs[0].text(
    0.95,
    0.9,
    f'Accuracy: {visual_acc * 100:.1f}%',
    transform=axs[0].transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

# Auditory plot
for i in range(5):
    axs[1].bar(np.arange(len(classes)) + i * 0.25, audio_probs[i], width=0.25, label=f'Time {i}' if i == 0 else "")
axs[1].set_xticks(np.arange(len(classes)) + 0.25)
axs[1].set_xticklabels(classes, fontsize=9)
axs[1].set_title('Auditory Modality: Audio Tone Classification Probabilities (sample times)', fontsize=12)
axs[1].set_ylabel('Probability', fontsize=10)
axs[1].legend(fontsize=9)
axs[1].tick_params(axis='y', labelsize=8)
audio_acc = accuracy_score(true_audio_labels, audio_preds)
axs[1].text(
    0.95,
    0.9,
    f'Accuracy: {audio_acc * 100:.1f}%',
    transform=axs[1].transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

# Haptic plot
axs[2].plot(range(T), true_haptic_values * 100, label='True Vibration Intensity', color='green')
axs[2].plot(range(T), pred_haptic_intensity, label='Predicted Vibration Intensity', linestyle='dashed', color='orange')
axs[2].set_title('Haptic Modality: Vibration Intensity Mapping', fontsize=12)
axs[2].set_ylabel('Intensity (%)', fontsize=10)
axs[2].legend(fontsize=9)
axs[2].set_ylim([-10, 110])
axs[2].tick_params(axis='both', which='major', labelsize=8)
haptic_r2 = r2_score(true_haptic_values, pred_haptic_values)
axs[2].text(
    0.95,
    0.9,
    f'R² Score: {haptic_r2:.2f}',
    transform=axs[2].transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
)

plt.tight_layout()
plt.show()

# Print accuracy scores in console as well
print(f"Visual Modality Classification Accuracy: {visual_acc * 100:.2f}%")
print(f"Auditory Modality Classification Accuracy: {audio_acc * 100:.2f}%")
print(f"Haptic Modality Regression R^2 Score: {haptic_r2:.2f}")
