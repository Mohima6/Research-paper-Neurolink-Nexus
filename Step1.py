import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape, Input
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate random EEG-like data
n_samples = 1000  # Number of samples
n_channels = 64   # Number of EEG channels
n_timesteps = 256 # Number of time steps per channel

# Random EEG data: shape (n_samples, n_channels, n_timesteps)
X = np.random.randn(n_samples, n_channels, n_timesteps)

# Adding noise to the EEG data (simulating noisy signal)
noise_factor = 0.5
X_noisy = X + noise_factor * np.random.randn(*X.shape)

# Random binary labels (for classification)
y = np.random.randint(0, 2, size=(n_samples,))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

# Building Autoencoder for Denoising
input_layer = Input(shape=(n_channels, n_timesteps))
encoded = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
encoded = Conv1D(64, 3, activation='relu', padding='same')(encoded)
encoded = Flatten()(encoded)
encoded = Dense(128, activation='relu')(encoded)

# Decoder for denoising
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(n_channels * n_timesteps, activation='sigmoid')(decoded)
decoded = Reshape((n_channels, n_timesteps))(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder to learn denoising
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))

# Use the trained autoencoder for denoising
X_train_denoised = autoencoder.predict(X_train)
X_test_denoised = autoencoder.predict(X_test)

# Building CNN model for classification using the denoised features
cnn_input = Input(shape=(X_train_denoised.shape[1], X_train_denoised.shape[2]))  # Shape after denoising
cnn_layer = Conv1D(64, 3, activation='relu')(cnn_input)
cnn_layer = Conv1D(128, 3, activation='relu')(cnn_layer)
cnn_layer = Flatten()(cnn_layer)
cnn_layer = Dense(64, activation='relu')(cnn_layer)
cnn_output = Dense(1, activation='sigmoid')(cnn_layer)

cnn_model = Model(cnn_input, cnn_output)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model on the denoised data
cnn_model.fit(X_train_denoised, y_train, epochs=10, batch_size=32, validation_data=(X_test_denoised, y_test))

# Evaluate the CNN model
y_pred = cnn_model.predict(X_test_denoised)
y_pred_classes = (y_pred > 0.5).astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plotting 4-in-1 graph
plt.figure(figsize=(10, 6))

# Plot 1: Noisy EEG signal (sample from training set)
plt.subplot(2, 2, 1)
plt.plot(X_train[0, 0, :])  # Plot one sample from one channel (noisy)
plt.title('Noisy EEG Signal')

# Plot 2: Denoised EEG signal (Autoencoder output)
plt.subplot(2, 2, 2)
plt.plot(X_train_denoised[0, 0, :])  # Denoised data
plt.title('Denoised EEG Signal (Autoencoder)')

# Plot 3: Clean EEG signal (original signal)
plt.subplot(2, 2, 3)
plt.plot(X_train[0, 0, :])  # Clean EEG data (original data)
plt.title('Clean EEG Signal')

# Plot 4: Confusion Matrix
plt.subplot(2, 2, 4)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')

# Show all plots
plt.tight_layout()
plt.show()
plt.tight_layout()
