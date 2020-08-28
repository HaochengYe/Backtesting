import yfinance as yf
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft

from datetime import datetime
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, iqr
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import arch

# ML imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import keras.backend as K
from keras.callbacks import EarlyStopping

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def coeff_deter(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


baba = yf.Ticker('V')

START = datetime(2010, 1, 1)
END = datetime(2020, 8, 21)

hist = baba.history(start=START, end=END).dropna()

X = []
Y1 = []
Y2 = []

for i in range(len(hist)-34):
    x = hist.Close.iloc[i:i+30].values
    y1 = hist.Close.iloc[i+30]
    y2 = hist.Close.iloc[i+34]
    X.append(x)
    Y1.append(y1)
    Y2.append(y2)

X = np.array(X)
Y1 = np.array(Y1).reshape(-1,1)
Y2 = np.array(Y2).reshape(-1,1)

print(X.shape, Y1.shape, Y2.shape)

cutoff = int(X.shape[0] * 0.8)

mm_scaler_x = MinMaxScaler()
tempx = X[:cutoff,:]

mm_scaler_y = MinMaxScaler()
tempy = Y1[:cutoff]

mm_scaler_x = mm_scaler_x.fit(tempx)
mm_scaler_y = mm_scaler_y.fit(tempy)

scale_X = mm_scaler_x.transform(X)
scale_Y = mm_scaler_y.transform(Y1)

X_train, X_test, Y_train, Y_test = train_test_split(scale_X, scale_Y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1 , X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


model_1 = Sequential()
model_1.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_1.add(Dropout(0.4))
model_1.add(LSTM(50))
model_1.add(Dropout(0.4))
model_1.add(Dense(1))
model_1.compile(loss='mae', optimizer='adam', metrics=[coeff_deter])

es = EarlyStopping(monitor='val_coeff_deter', mode='max', patience=5)


model_1.fit(X_train, Y_train,
            batch_size=32,
            validation_data=(X_test, Y_test),
            epochs=50,
            callbacks=[es])


cutoff = int(X.shape[0] * 0.8)

mm_scaler_x = MinMaxScaler()
tempx = X[:cutoff,:]

mm_scaler_y = MinMaxScaler()
tempy = Y2[:cutoff]

mm_scaler_x = mm_scaler_x.fit(tempx)
mm_scaler_y = mm_scaler_y.fit(tempy)

scale_X = mm_scaler_x.transform(X)
scale_Y = mm_scaler_y.transform(Y2)

X_train, X_test, Y_train, Y_test = train_test_split(scale_X, scale_Y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1 , X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


model_5 = Sequential()
model_5.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model_5.add(Dropout(0.4))
model_5.add(LSTM(50))
model_5.add(Dropout(0.4))
model_5.add(Dense(1))
model_5.compile(loss='mae', optimizer='adam', metrics=[coeff_deter])

es = EarlyStopping(monitor='val_coeff_deter', mode='max', patience=5)


model_5.fit(X_train, Y_train,
            batch_size=32,
            validation_data=(X_test, Y_test),
            epochs=50,
            callbacks=[es])

# trading

ASSET = 10000

t = 0

while t < X_test.shape[0]:
    base = X_train.shape[0]





