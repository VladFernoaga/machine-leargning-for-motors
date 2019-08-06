import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


# Load dataset
url = "./train_data.xlsx"
names = ['sarcina', 'turatie', 'p', 'rezultat']
dataset_load = pd.read_excel(url, sheet_name="pe", names=names)
dataset= dataset_load.values

x=dataset[:,0:3]
y=dataset[:,3]
y=np.reshape(y, (-1,1))
scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))
xscale=scaler.transform(x)
yscale=scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(10, input_dim=3, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(50, activation='relu'))
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

Xnew = np.array([[0.30, 10, 280],[0.30,12,340],[1,20,2700] 
 ])
XnewScale=scaler.transform(Xnew)
print(XnewScale)

ynew=model.predict(Xnew)
print("X=%s, Predicted=%s, Actual=17.05" % (Xnew[0], ynew[0]))
print("X=%s, Predicted=%s, Actual=28.48" % (Xnew[1], ynew[1]))
print("X=%s, Predicted=%s, Actual=157.9" % (Xnew[2], ynew[2]))

