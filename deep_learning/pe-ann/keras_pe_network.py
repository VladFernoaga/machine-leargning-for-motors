
# LOAD DATA
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas
from sklearn import model_selection

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# Load dataset
url = "./train_data.xlsx"
names = ['sarcina', 'turatie', 'p', 'rezultat']
dataset_pe = pandas.read_excel(url, sheet_name="pe-normalizated", names=names)

# Split-out validation dataset
print("Split input features from the expected results")
array = dataset_pe.values
inputFeatures = array[:,0:3]
expectedResult = array[:,3]

# print ("split train stet from validation set")
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(inputFeatures, expectedResult, test_size=validation_size, random_state=seed)
# print (Y_train)


# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(25, input_dim=3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='linear'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
	return model


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, inputFeatures, expectedResult, cv=kfold)
print(results)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# # create model
# model = Sequential()
# model.add(Dense(13, input_dim=3, activation='relu'))
# # model.add(Dense(2, activation='relu'))
# model.add(Dense(1, activation='softmax'))

# # Compile model
# model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mse', 'mae', 'mape', 'cosine'])

# Fit the model
# model.fit(X_train, Y_train, epochs=150, batch_size=10,  verbose=2)

# evaluate the model
# scores = model.evaluate(X_train, Y_train)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
# predictions = model.predict(X_validation, verbose=2)
# round predictions
#rounded = [round(x[0]) for x in predictions]
# print(predictions)