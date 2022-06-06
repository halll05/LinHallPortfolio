from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score

df = read_csv('data/nonrigid_rollall_processed.csv')
df = df.dropna()
print('check if y has strong connections with your X:')
print(df.corr())

init_values = df.values

n_records_train = int(len(init_values) * 0.70)

itrain = init_values[:n_records_train, :]
itest = init_values[n_records_train:, :]

# split in inputs and outputs
itrain_X, itrain_y = itrain[:, 1:], itrain[:, 0]
itest_X, itest_y = itest[:, 1:], itest[:, 0]
# reshape input to be 3D [samples, timesteps, features]
itrain_X = itrain_X.reshape((itrain_X.shape[0], 1, itrain_X.shape[1]))
itest_X = itest_X.reshape((itest_X.shape[0], 1, itest_X.shape[1]))

ground_truth = itest_y

values = df.values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

reframed = DataFrame(scaled)

print(reframed.head())

values = reframed.values
n_records_train = int(len(values) * 0.70)

train = values[:n_records_train, :]
test = values[n_records_train:, :]

# split in inputs and outputs
train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 0:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

r_squared = r2_score(inv_y, inv_yhat)
print('Test R Squared: %.3f' % r_squared)

pyplot.plot(ground_truth, label='ground truth')
pyplot.plot(inv_yhat, label='prediction')
pyplot.legend()
pyplot.show()