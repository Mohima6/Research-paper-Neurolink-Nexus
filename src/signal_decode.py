import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from scipy.signal import butter, filtfilt


# === Signal generation and filtering ===

def generate_fake_eeg(duration=30, fs=250):
    """
    Generate fake EEG signal for 'duration' seconds at sampling freq fs.
    Signal is a mix of sine waves + noise simulating EEG.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # simulate 4 classes by mixing different freq components at intervals
    signal = np.zeros_like(t)
    segment_len = int(len(t) / 4)
    freqs = [8, 12, 20, 30]  # example freq for each class

    for i in range(4):
        start = i * segment_len
        end = start + segment_len
        signal[start:end] = (np.sin(2 * np.pi * freqs[i] * t[start:end]) +
                             0.5 * np.random.randn(segment_len))

    return t, signal


def apply_filter(signal, fs=250, lowcut=1.0, highcut=40.0, order=4):
    """
    Bandpass Butterworth filter for EEG (1-40Hz)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


# === Prepare dataset ===

def decode_signal_to_class(filtered_signal, fs=250, window_size=1.0):
    """
    Split filtered EEG into windows and assign class labels based on segment index.
    Classes: 0=Stop, 1=Left, 2=Right, 3=Forward
    """
    samples_per_window = int(fs * window_size)
    num_classes = 4

    X = []
    y = []

    total_samples = len(filtered_signal)
    total_windows = total_samples // samples_per_window

    for i in range(total_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        segment = filtered_signal[start:end]

        X.append(segment)
        # Assign class label depending on quarter segments
        class_label = i * num_classes // total_windows
        y.append(class_label)

    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = to_categorical(y, num_classes=num_classes)
    return X, y


# === CNN Model ===

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(16, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(32, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# === Plot all results in 4-in-1 figure ===

def plot_results(t, raw_signal, filtered_signal, y_true, y_pred, class_names, history):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[INFO] Classification Accuracy: {acc * 100:.2f}%")

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # 1) Raw and filtered signals (first 5 seconds)
    axs[0, 0].plot(t[:5 * 250], raw_signal[:5 * 250], label='Raw Signal', alpha=0.7)
    axs[0, 0].plot(t[:5 * 250], filtered_signal[:5 * 250], label='Filtered Signal', alpha=0.7)
    axs[0, 0].set_title("Raw and Filtered EEG Signal (first 5s)", fontsize=12, pad=10)
    axs[0, 0].set_xlabel("Time (s)", fontsize=10, labelpad=8)
    axs[0, 0].set_ylabel("Amplitude", fontsize=10, labelpad=8)
    axs[0, 0].legend(fontsize=9)
    axs[0, 0].grid(True)

    # 2) Confusion matrix with accuracy
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axs[0, 1], cbar=False)
    axs[0, 1].set_xlabel("Predicted", fontsize=10, labelpad=8)
    axs[0, 1].set_ylabel("True", fontsize=10, labelpad=8)
    axs[0, 1].set_title(f"Confusion Matrix\nAccuracy: {acc * 100:.2f}%", fontsize=12, pad=10)

    # 3) Accuracy & Loss curves
    epochs = range(1, len(history.history['accuracy']) + 1)
    axs[1, 0].plot(epochs, history.history['accuracy'], label='Train Acc')
    axs[1, 0].plot(epochs, history.history['val_accuracy'], label='Val Acc')
    axs[1, 0].plot(epochs, history.history['loss'], '--', label='Train Loss')
    axs[1, 0].plot(epochs, history.history['val_loss'], '--', label='Val Loss')
    axs[1, 0].set_title("Accuracy & Loss", fontsize=12, pad=10)
    axs[1, 0].set_xlabel("Epochs", fontsize=10, labelpad=8)
    axs[1, 0].set_ylabel("Value", fontsize=10, labelpad=8)
    axs[1, 0].legend(fontsize=9)
    axs[1, 0].grid(True)

    # 4) Bar plot of predicted class distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    axs[1, 1].bar([class_names[i] for i in unique], counts, color='orange')
    axs[1, 1].set_title("Decoded Control Command Distribution", fontsize=12, pad=10)
    axs[1, 1].set_xlabel("Command", fontsize=10, labelpad=8)
    axs[1, 1].set_ylabel("Count", fontsize=10, labelpad=8)

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.3, hspace=0.4)
    plt.show()



# === Main ===

def main():
    class_names = ['Stop', 'Left', 'Right', 'Forward']

    # Generate and filter signal (longer duration for enough samples)
    t, raw_signal = generate_fake_eeg(duration=30)
    filtered_signal = apply_filter(raw_signal)

    # Prepare data for CNN
    X, y = decode_signal_to_class(filtered_signal)
    print(f"Total samples: {X.shape[0]}")  # Should be enough for stratify

    # Split with stratify to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42)

    # Create and train model
    model = create_cnn_model(input_shape=(X.shape[1], 1), num_classes=y.shape[1])
    history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test), verbose=2)

    # Predict and evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Plot all results in one figure with 4 subplots
    plot_results(t, raw_signal, filtered_signal, y_true, y_pred, class_names, history)


if __name__ == "__main__":
    main();