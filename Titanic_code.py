# import packages
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

# extract data
titanic_data = pd.read_csv("C:\\To Read\\Data_sets\\titanic.csv")
titanic_data

# preprocessing
input_data = titanic_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Parch', 'Ticket', 'SibSp', 'Cabin', 'Embarked'])
output_data = titanic_data['Survived']
decode = LabelEncoder()
input_data['sex'] = decode.fit_transform(input_data['Sex'])
input_dataa = input_data.drop(columns=['Sex'])
input_dataa.isnull().sum()
input_data['Age'].fillna(0, inplace = True)

# graph
pd.crosstab(input_data['Age'], output_data).head(10).plot(kind='bar')

# training input and output
input_dataa_train, input_data_test, output_data_train, output_data_test = train_test_split(input_dataa, output_data, train_size=0.8)
model = DecisionTreeClassifier()
model.fit(input_dataa_train, output_data_train)
prediction = model.predict(input_data_test)

# accuracy
print('The accuracy is ',int((model.score(input_dataa_train, output_data_train))*100),'%')