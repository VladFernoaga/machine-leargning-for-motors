

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "./train_data.xlsx"
names = ['sarcina', 'turatie', 'p', 'rezultat']
dataset_pe = pandas.read_excel(url, sheet_name="pe", names=names)

# shape
print("Rows, attributes: ")
print(dataset_pe.shape)

print()

# descriptions
print(dataset_pe.describe())

# class distribution
# print(dataset_pe.groupby('rezultat').size())

# scatter plot matrix
scatter_matrix(dataset_pe)
plt.show()