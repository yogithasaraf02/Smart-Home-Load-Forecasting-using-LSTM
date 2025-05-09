import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error


file_path = '/datasetLstm.csv'
df = pd.read_csv(file_path, header=None)
print(f"Data shape: {df.shape}")

load_values = df.iloc[:, 1].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(load_values)

X, y = [], []
time_steps = 24
for i in range(time_steps, len(data_scaled)):
    X.append(data_scaled[i-time_steps:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train_full, X_test3, y_train_full, y_test3 = train_test_split(X, y, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, shuffle=False)
X_test1, X_test2 = X_test3[:3421], X_test3[3421:]
y_test1, y_test2 = y_test3[:3421], y_test3[3421:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

model.save('load_prediction_model.h5')

def evaluate_and_plot(X_test, y_test, label):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"The mean squared error (MSE) on the {label} dataset is {round(mse, 3)} over {len(y_test)} test samples.")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], label='Actual', linewidth=2)
    plt.plot(y_pred[:100], label='Predicted', linestyle='--')
    plt.title(f'Actual vs Predicted Load ({label})')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Load')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{label}_prediction_plot.png')  
    plt.show()


evaluate_and_plot(X_test1, y_test1, "Test1")
evaluate_and_plot(X_test2, y_test2, "Test2")
evaluate_and_plot(X_test3, y_test3, "Test3")
