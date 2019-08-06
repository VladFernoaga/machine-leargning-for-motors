import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


# Load dataset
url = "./train_data.xlsx"
names = ['sarcina', 'turatie','p', 'fum', 'pk', 'tge', 'cs',  'rezultat']
dataset_load = pd.read_excel(url, sheet_name="unified", names=names)
dataset= dataset_load.values

x=dataset[:,0:7]
y=dataset[:,7]
y=np.reshape(y, (-1,1))
scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))
xscale=scaler.transform(x)
yscale=scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(23, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(11, activation='relu'))
# model.add(Activation('softmax'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mse', optimizer='sgd', metrics=['mse','mae'])


history = model.fit(X_train, y_train, epochs=150, batch_size=10,  verbose=1, validation_split=0.2)

scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

# "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

Xnew = np.array([[1.00, 10.00, 60.00, 69.0, 175.0, 520.0, 224.4],[1.00, 12.00, 270.00, 70.0, 255.0, 530.0, 217.6],[1.00, 22.00, 1000.00, 13.0, 580.0, 520.0, 223.7] ])
XnewScale=scaler.transform(Xnew)
print(XnewScale)

ynew=model.predict(Xnew)
print("X=%s, Predicted=%s, Actual=72.06" % (Xnew[0], ynew[0]))
print("X=%s, Predicted=%s, Actual=95.00" % (Xnew[1], ynew[1]))
print("X=%s, Predicted=%s, Actual=172" % (Xnew[2], ynew[2]))

