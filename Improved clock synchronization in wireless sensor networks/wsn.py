
!pip install tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE


# Loading the dataset
df = pd.read_csv("/content/WSN-DS.csv")
print(df.head())

# Displaying the basic statistics for each column
print(df.describe())

# Display the data types of each column
print(df.dtypes)

# Display the number of missing values in each column
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Attack type'] = label_encoder.fit_transform(df['Attack type'])
class_counts = df['Attack type'].value_counts()
print(class_counts)

X = df.drop('Attack type', axis=1)
y = df['Attack type']

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Random oversampling
random_oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled_over, y_resampled_over = random_oversampler.fit_resample(X, y)

# Random undersampling
random_undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_resampled_under, y_resampled_under = random_undersampler.fit_resample(X, y)

# Split the data into training and testing sets for oversampled data
X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X_resampled_over, y_resampled_over, test_size=0.2, random_state=42)

# Split the data into training and testing sets for undersampled data
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_resampled_under, y_resampled_under, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_lstm_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_lstm_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
y_train_encoded = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test_encoded = to_categorical(y_test, num_classes=len(label_encoder.classes_))

from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame containing the 'Attack type' column
label_encoder = LabelEncoder()
df['Attack type'] = label_encoder.fit_transform(df['Attack type'])

# Print the mapping of original labels to numerical values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Mapping of Attack type:")
print(label_mapping)

#Clock values before using Kalman Filter
print("Noisy Measurements (Before Kalman Filtering):")
for step, measurement in enumerate(noisy_measurements):
    print(f"Clock value at step {step}: {measurement}")

# Kalman Filter Loop
estimated_clock = np.zeros(len(noisy_measurements))
for k in range(len(noisy_measurements)):
    kf.predict()
    kf.update(noisy_measurements[k])
    estimated_clock[k] = kf.x_hat
    print(f"Estimated Clock at step {k}: {estimated_clock[k]}")

# Reshape for LSTM model
X_lstm_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_lstm_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# One-hot encode the target variable
y_train_encoded = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_test_encoded = to_categorical(y_test, num_classes=len(label_encoder.classes_))

# Kalman Filter for clock synchronization
class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=0.1, initial_offset=0.1, initial_covariance=1.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.A = np.array([[1]])
        self.H = np.array([[1]])
        self.x_hat = initial_offset
        self.P = initial_covariance

    def predict(self):
        self.x_hat_minus = self.A.dot(self.x_hat)
        self.P_minus = self.A.dot(self.P).dot(self.A.T) + self.Q

    def update(self, measurement):
        K = self.P_minus.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P_minus).dot(self.H.T) + self.R))
        self.x_hat = self.x_hat_minus + K.dot(measurement - self.H.dot(self.x_hat_minus))
        self.P = (1 - K.dot(self.H)).dot(self.P_minus)

def simulate_network_clock(true_offset, true_skew, process_noise, measurement_noise, num_steps):
    true_clock = true_offset + true_skew * np.arange(num_steps)
    noisy_measurements = true_clock + np.sqrt(measurement_noise) * np.random.randn(num_steps)
    return true_clock, noisy_measurements

# Kalman Filter Loop
# kf = KalmanFilter(process_noise=1e-5, measurement_noise=0.1, initial_offset=0.1, initial_covariance=1.0)
kf = KalmanFilter()
true_clock, noisy_measurements = simulate_network_clock(true_offset=0.1, true_skew=1.01,process_noise=1e-5, measurement_noise=0.1, num_steps=100)

estimated_clock = np.zeros(len(noisy_measurements))
for k in range(len(noisy_measurements)):
    kf.predict()
    kf.update(noisy_measurements[k])
    estimated_clock[k] = kf.x_hat
    print(f"Estimated Clock at step {k}: {estimated_clock[k]}")

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(true_clock, label="True Clock")
plt.plot(noisy_measurements, 'rx', label="Noisy Measurements")
plt.title("Before Kalman Filtering")
plt.xlabel("Time Steps")
plt.ylabel("Clock Offset")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(true_clock, label="True Clock")
plt.plot(estimated_clock, label="Kalman Filter Estimate")
plt.title("After Kalman Filtering")
plt.xlabel("Time Steps")
plt.ylabel("Clock Offset")
plt.legend()

plt.tight_layout()
plt.show()

# LSTM model creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical

def lr_scheduler(epoch, lr):
    if epoch > 10:
        return 0.001
    else:
        return lr

model_lstm = Sequential()
model_lstm.add(LSTM(64, activation='relu', input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
model_lstm.add(Dense(len(label_encoder.classes_), activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lr_schedule_lstm = LearningRateScheduler(lr_scheduler)
history_lstm = model_lstm.fit(X_lstm_train, y_train_encoded, epochs=10, batch_size=64,
                              validation_data=(X_lstm_test, y_test_encoded),
                              callbacks=[lr_schedule_lstm], verbose=2)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Classification evaluation metrics for LSTM model
y_pred_prob_lstm = model_lstm.predict(X_lstm_test)
y_pred_labels_lstm = np.argmax(y_pred_prob_lstm, axis=1)

# Check for NaN values in predictions or target data
nan_indices = np.isnan(y_pred_prob_lstm).any() or np.isnan(y_pred_labels_lstm).any()

if nan_indices:
    # Handle NaN values
    print("NaN values detected in predictions or labels. Handle accordingly.")
else:
    confusion_mat_lstm = confusion_matrix(y_test, y_pred_labels_lstm)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat_lstm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (LSTM Model)')
    plt.show()

    # Additional evaluation metrics for the LSTM model
    print("Additional Evaluation Metrics (LSTM Model):")
    target_names = [str(c) for c in label_encoder.classes_]
    print(classification_report(y_test, y_pred_labels_lstm, labels=label_encoder.classes_, target_names=target_names))

    # Convert multiclass labels to binary labels
    y_test_binary = label_binarize(y_test, classes=label_encoder.classes_)

    # ROC AUC for the LSTM model
    fpr_lstm = {}
    tpr_lstm = {}
    roc_auc_lstm = {}

    for i in range(len(label_encoder.classes_)):
        fpr_lstm[i], tpr_lstm[i], _ = roc_curve(y_test_binary[:, i], y_pred_prob_lstm[:, i])
        roc_auc_lstm[i] = auc(fpr_lstm[i], tpr_lstm[i])
        print(f'ROC AUC for Class {i} ({label_encoder.classes_[i]}): {roc_auc_lstm[i]:.4f}')

    # Plot ROC curves for each class for the LSTM model
    plt.figure(figsize=(10, 8))
    for i in range(len(label_encoder.classes_)):
        plt.plot(fpr_lstm[i], tpr_lstm[i], label=f'Class {i} ({label_encoder.classes_[i]}) - AUC: {roc_auc_lstm[i]:.2f}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class (LSTM Model)')
    plt.legend()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Kalman Filter
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_variance, measurement_variance):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self):
        self.state = self.state
        self.covariance += self.process_variance

    def update(self, measurement):
        kalman_gain = self.covariance / (self.covariance + self.measurement_variance)
        self.state += kalman_gain * (measurement - self.state)
        self.covariance = (1 - kalman_gain) * self.covariance

# Check available column names in the dataset
print(df.columns)

if 'Attack type' in df.columns:
    noisy_measurement = df["Attack type"].values
    filtered_estimates = []
    initial_state = 0.0
    initial_covariance = 1.0
    process_variance = 0.1
    measurement_variance = 1.0

    kalman_filter = KalmanFilter(initial_state, initial_covariance, process_variance, measurement_variance)

    for measurement in noisy_measurement:
        kalman_filter.predict()
        kalman_filter.update(measurement)
        filtered_estimates.append(kalman_filter.state)

    # Plotting results of Kalman Filter
    plt.plot(df.index, noisy_measurement, label='Noisy Measurement')
    plt.plot(df.index, filtered_estimates, label='Filtered Estimate')
    plt.xlabel('Time')
    plt.ylabel('Attack type')
    plt.legend()
    plt.title('Kalman Filter Example')
    plt.show()
else:
    print("Column 'Dist_To_CH' not found in the dataset.")

import numpy as np

class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=0.1, initial_covariance=1.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.A = np.array([[1]])
        self.H = np.array([[1]])
        self.x_hat = 0  # Initial state estimate
        self.P = initial_covariance

    def predict(self):
        self.x_hat_minus = self.A.dot(self.x_hat)
        self.P_minus = self.A.dot(self.P).dot(self.A.T) + self.Q

    def update(self, measurement):
        K = self.P_minus.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P_minus).dot(self.H.T) + self.R))
        self.x_hat = self.x_hat_minus + K.dot(measurement - self.H.dot(self.x_hat_minus))
        self.P = (1 - K.dot(self.H)).dot(self.P_minus)

class DualKalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_state_covariance):
        self.kf1 = KalmanFilter(process_noise, measurement_noise, initial_state_covariance)
        self.kf2 = KalmanFilter(process_noise, measurement_noise, initial_state_covariance)

    def predict(self):
        self.kf1.predict()
        self.kf2.predict()

    def update(self, measurement1, measurement2):
        self.kf1.update(measurement1)
        self.kf2.update(measurement2)

def simulate_dual_network_clock(true_offset, true_skew, process_noise, measurement_noise, num_steps):
    true_clock1 = true_offset + true_skew * np.arange(num_steps)
    true_clock2 = true_offset + true_skew * np.arange(num_steps)

    noisy_measurements1 = true_clock1 + np.sqrt(measurement_noise) * np.random.randn(num_steps)
    noisy_measurements2 = true_clock2 + np.sqrt(measurement_noise) * np.random.randn(num_steps)

    return true_clock1, true_clock2, noisy_measurements1, noisy_measurements2

# Dual Kalman Filter Loop
process_noise = 1e-5
measurement_noise = 0.1
initial_state_covariance = 1.0

dual_kf = DualKalmanFilter(process_noise, measurement_noise, initial_state_covariance)
true_clock1, true_clock2, noisy_measurements1, noisy_measurements2 = simulate_dual_network_clock(true_offset=0.1, true_skew=1.01, process_noise=process_noise, measurement_noise=measurement_noise, num_steps=100)

estimated_clock1 = np.zeros(len(noisy_measurements1))
estimated_clock2 = np.zeros(len(noisy_measurements2))

for k in range(len(noisy_measurements1)):
    dual_kf.predict()
    dual_kf.update(noisy_measurements1[k], noisy_measurements2[k])
    estimated_clock1[k] = dual_kf.kf1.x_hat
    estimated_clock2[k] = dual_kf.kf2.x_hat

    print(f"Estimated Clock 1 at step {k}: {estimated_clock1[k]}")
    print(f"Estimated Clock 2 at step {k}: {estimated_clock2[k]}")

import matplotlib.pyplot as plt
# Plotting the estimated clocks
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(estimated_clock1)), estimated_clock1, label='Estimated Clock 1', color='blue')
plt.plot(np.arange(len(true_clock1)), true_clock1, label='True Clock 1', linestyle='--', color='red')
plt.xlabel('Steps')
plt.ylabel('Clock Value')
plt.title('Estimated Clock 1 vs True Clock 1')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(estimated_clock2)), estimated_clock2, label='Estimated Clock 2', color='green')
plt.plot(np.arange(len(true_clock2)), true_clock2, label='True Clock 2', linestyle='--', color='orange')
plt.xlabel('Steps')
plt.ylabel('Clock Value')
plt.title('Estimated Clock 2 vs True Clock 2')
plt.legend()

plt.tight_layout()
plt.show()

